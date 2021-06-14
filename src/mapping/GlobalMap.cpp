/*
* This file is part of the EDS: Event-aided Direct Sparse Odometry
* (https://rpg.ifi.uzh.ch/eds.html)
*
* Copyright (c) 2022 Javier Hidalgo-Carrió, Robotics and Perception
* Group (RPG) University of Zurich.
*
* EDS is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* EDS is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "GlobalMap.hpp"
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <pcl/filters/radius_outlier_removal.h>

using namespace eds::mapping;


GlobalMap::GlobalMap(const ::eds::mapping::Config &config,
                  const ::eds::calib::CameraInfo &cam_info,
                  const ::eds::bundles::Config &bundles_config,
                  const double &percent_tracking_points)
{
    this->config = config;
    this->bundles_config = bundles_config;
    this->setNumPointsInTracking(cam_info.height, cam_info.width, percent_tracking_points, bundles_config.window_size);

    /** the initial value is to use all the points from tracking **/
    this->active_points_per_kf = this->num_points_in_tracking;

    this->last_kf_id = NONE_KF;
    this->prev_last_kf_id = NONE_KF;

    std::cout<<"[GLOBAL_MAP] CONSTRUCTOR TOTAL_ACTIVE_POINTS["<<this->total_active_points<<"] "<<this->active_points_per_kf<<" PER KF"<<std::endl;
}

void GlobalMap::insert(std::shared_ptr<eds::tracking::KeyFrame> kf,
                    const ::eds::mapping::POINTS_SELECT_METHOD &points_select_method)
{

    /** Insert the keyframe information in the global map **/
    Eigen::Matrix3d K;
    cv::cv2eigen(kf->K_ref, K);
    auto trans_quater = kf->getTransQuater();
    eds::mapping::KeyFrameInfo kf_info(kf->time,
            trans_quater.first, trans_quater.second, // pose
            ::base::Vector6d::Zero(), //velocity
            kf->img_data, //image
            kf->img.rows, kf->img.cols, // height and width
            K, //intrinsics
            0.0, 0.0); // affine brightness transfer parameters a and b
    this->camera_kfs.insert({kf->idx, kf_info});

    /** Insert the points information in the global map **/
    std::vector<PointInfo> points;
    auto it_c = kf->coord.begin();
    auto it_nc = kf->norm_coord.begin();
    auto it_ph = kf->bundle_patches.begin();
    auto it_id = kf->inv_depth.begin();
    auto it_res = kf->residuals.begin();
    auto it_grad = kf->grad.begin();
    for (;it_c != kf->coord.end(); ++it_c, ++it_nc, ++it_ph, ++it_id, ++it_res, ++it_grad)
    {
        PointInfo info (
            ::eds::mapping::Point2d(*it_c), //pixel coord
            ::eds::mapping::Point2d(*it_nc), //norm coord
            *it_ph, //d x d patch
            ::eds::mapping::mu(*it_id), //inverse depth
            kf->img.at<double>(*it_c), // intensity value
            *it_res, //point residual
            cv::norm(*it_grad)); //gradient magnitude
            /** push the point **/
            points.push_back(info);
    }

    /** Order the keypoints according to the method **/
    if (points_select_method != POINTS_SELECT_METHOD::NONE)
    {
        /** Update the number of active points **/
        this->updateNumberActivePoints(this->camera_kfs.size());

        /** Select the points to use in PBA: if method is NONE no selection and uses all points  **/
        this->orderPoints(points, points_select_method, this->active_points_per_kf);
    }

    /**  Insert all the new points to its host keyframe **/
    this->point_kfs.insert({kf->idx, points});

    /** The two last Keyframe ids **/
    this->prev_last_kf_id = this->last_kf_id;
    this->last_kf_id = kf->idx;
    std::cout<<"[GLOBAL_MAP] INSERTED KF["<<kf->idx<<"] trans:["<<kf_info.t[0]<<","
    <<kf_info.t[1]<<","<<kf_info.t[2]<<"] quater:["<<kf_info.q.x()<<","
    <<kf_info.q.y()<<","<<kf_info.q.z()<<","<<kf_info.q.w()<<"] height: "
    <<kf_info.height<<" width: "<<kf_info.width<<"\nK:\n"<<kf_info.K<<std::endl;

    std::cout << "[GLOBAL_MAP] INSERTED LAST KF["<<this->last_kf_id <<"] with "<<points.size()
            <<" points PREV LAST KF ["<<this->prev_last_kf_id<<"] global_map SIZE ["<<this->size()<<"]"<<std::endl;

}

::base::Transform3d GlobalMap::getKFTransform(const uint64_t &kf_idx)
{
    std::cout << "[GLOBAL_MAP] Requested T_w_kf["<<kf_idx<<"]"<<std::endl;
    /** Key is not present **/
    if (this->camera_kfs.find(kf_idx) == this->camera_kfs.end())
        return ::base::Transform3d(::Eigen::Affine3d::Identity().matrix() * ::base::NaN<double>());
    ::base::Transform3d T_w_kf(this->camera_kfs.at(kf_idx).q);
    T_w_kf.translation() = this->camera_kfs.at(kf_idx).t;
    return T_w_kf;
}

bool GlobalMap::isKFinMap(const uint64_t &kf_idx)
{
    return this->camera_kfs.find(kf_idx)!=this->camera_kfs.end();
}

bool GlobalMap::removeKeyFrame(const uint64_t &kf_id)
{
    /** Remove a keyframe and its points
    from the map **/
    if ((this->camera_kfs.find(kf_id)!=this->camera_kfs.end()) and (this->point_kfs.find(kf_id) != this->point_kfs.end()))
    {
        this->camera_kfs.erase(kf_id);
        this->point_kfs.erase(kf_id);
        std::cout<<"[GLOBAL MAP] KF ID["<<kf_id<<"] to marginalize. new window size: "<<this->camera_kfs.size()<<std::endl;
        return true;
    }
    return false;
}

cv::Mat GlobalMap::vizKeyFrame(const uint64_t &kf_id)
{
    eds::mapping::KeyFrameInfo &kf = this->camera_kfs.at(kf_id);

    /** Get the inverse depth in floating point (0-1) **/
    std::vector<double>inv_depth;
    double min = base::infinity<double>(); double max = 0;
    auto point_end = this->point_kfs.at(kf_id).begin();
    std::advance(point_end, std::min((int)this->point_kfs.at(kf_id).size(), this->active_points_per_kf));
    for (auto it = this->point_kfs.at(kf_id).begin(); it != point_end; ++it)
    {
        inv_depth.push_back(it->inv_depth);
        min = std::min(min, it->inv_depth);
        max = std::max(max, it->inv_depth);
    }

    /** Between (0-1) **/
    if (min != max)
    {
        for (auto &it : inv_depth)
        {
            it = (it - min) / (max - min);
        }
    }

    /** Create the keyframe image **/
    cv::Mat img = cv::Mat(kf.height, kf.width, CV_64FC1, cv::Scalar(0));
    memcpy(img.data, kf.img.data(), kf.img.size()*sizeof(double));
    cv::Mat img_color; img.convertTo(img_color, CV_8UC1, 255, 0);

    /** Draw the points in the keyframe image **/
    cv::cvtColor(img_color, img_color, cv::COLOR_GRAY2RGB);
    auto it_inv = inv_depth.begin();
    for (auto it = this->point_kfs.at(kf_id).begin(); it != point_end; ++it, ++it_inv)
    {
        cv::Vec3b color = ::eds::utils::valueToColor(*it_inv);
        cv::circle(img_color, cv::Point2d(it->coord), 2.0, color, cv::FILLED, cv::LINE_AA);
    }

    //double min_ = * std::min_element(std::begin(inv_depth), std::end(inv_depth));
    //double max_ = * std::max_element(std::begin(inv_depth), std::end(inv_depth));
    //std::cout<<"[VIZ_KF: "<<kf_id<<"] min: "<<min<<" max: "<<max<<std::endl;
    //std::cout<<"[VIZ_KF: "<<kf_id<<"] norm_min: "<<min_<<" norm_max: "<<max_<<std::endl;

    /** Write KF ID text **/
    const std::string text = "#" + std::to_string(kf_id);
    cv::putText(img_color, text, cv::Point(5, img_color.rows-5), 
            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0,0,255), 0.1, cv::LINE_AA);
    //const std::string text_min = "min:"+std::to_string(min);
    //cv::putText(img_color, text_min, cv::Point(5, 20), 
    //        cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0,0,0), 0.1, cv::LINE_AA);
    //const std::string text_max = "max:"+std::to_string(max);
    //cv::putText(img_color, text_max, cv::Point(5, 40), 
    //        cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0,0,0), 0.1, cv::LINE_AA);

    return img_color;
}

cv::Mat GlobalMap::residualsViz(const uint64_t &kf_id)
{
    eds::mapping::KeyFrameInfo &kf = this->camera_kfs.at(kf_id);

    /** Get the residuals in floating point (0-1) **/
    std::vector<double>residuals;
    double min = base::infinity<double>(); double max = 0;
    auto point_end = this->point_kfs.at(kf_id).begin();
    std::advance(point_end, std::min((int)this->point_kfs.at(kf_id).size(), this->active_points_per_kf));
    for (auto it = this->point_kfs.at(kf_id).begin(); it != point_end; ++it)
    {
        residuals.push_back(it->residual);
        min = std::min(min, it->residual);
        max = std::max(max, it->residual);
    }

    /** Between (0-1) **/
    if (min != max)
    {
        for (auto &it : residuals)
        {
            it = (it - min) / (max - min);
        }
    }
    //double min_ = * std::min_element(std::begin(residuals), std::end(residuals));
    //double max_ = * std::max_element(std::begin(residuals), std::end(residuals));
    //std::cout<<"[RESIDUAL VIZ] KF: "<<kf_id<<"] min: "<<min<<" max: "<<max<<" size: "<<residuals.size()<<std::endl;
    //std::cout<<"[RESIDUAL VIZ] KF: "<<kf_id<<"] norm_min: "<<min_<<" norm_max: "<<max_<<" size: "<<residuals.size()<<std::endl;

    /** Create the keyframe image **/
    cv::Mat img = cv::Mat(kf.height, kf.width, CV_64FC1, cv::Scalar(0));
    memcpy(img.data, kf.img.data(), kf.img.size()*sizeof(double));
    cv::Mat img_color; img.convertTo(img_color, CV_8UC1, 255, 0);

    /** Draw the residuals points in the keyframe image **/
    cv::cvtColor(img_color, img_color, cv::COLOR_GRAY2RGB);
    auto it_res = residuals.begin();
    for (auto it = this->point_kfs.at(kf_id).begin(); it != point_end; ++it, ++it_res)
    {
        cv::Vec3b color = ::eds::utils::valueToColor(*it_res);
        cv::circle(img_color, cv::Point2d(it->coord), 2.0, color, cv::FILLED, cv::LINE_AA);
    }

    /** Write KF ID text **/
    const std::string text = "#" + std::to_string(kf_id);
    cv::putText(img_color, text, cv::Point(5, img_color.rows-5),
            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0,0,255), 0.1, cv::LINE_AA);
    return img_color;
}

cv::Mat GlobalMap::vizMosaic(const int &bundles_window_size)
{
    /** Get the mosaic vector part **/
    std::vector<cv::Mat> img_vector;
    for (auto kf : this->camera_kfs)
    {
        cv::Mat img;
        cv::resize(this->vizKeyFrame(kf.first), img, cv::Size(kf.second.width/2, kf.second.height/2), cv::INTER_CUBIC);
        img_vector.push_back(img);
    }

    /** Get the last Keyframe with all the map points (all points in the sliding window) **/
    eds::mapping::KeyFrameInfo &last_kf = this->camera_kfs.at(this->last_kf_id);

    /** Get the map **/
    ::eds::mapping::IDepthMap2d last_kf_depth_map;
    this->getIDepthMap(this->last_kf_id, last_kf_depth_map, true);

    /** Normalized between (0-1) the inverse depth **/
    double min = * std::min_element(std::begin(last_kf_depth_map.idepth), std::end(last_kf_depth_map.idepth));
    double max = * std::max_element(std::begin(last_kf_depth_map.idepth), std::end(last_kf_depth_map.idepth));
    if (min != max)
    {
        for (auto &it:last_kf_depth_map.idepth)
        {
            it = (it - min)/(max - min);
        }
    }
    //std::cout<<"[VIZ_MOSAIC: "<<this->last_kf_id<<"] min: "<<min<<" max: "<<max<<std::endl;
    min = * std::min_element(std::begin(last_kf_depth_map.idepth), std::end(last_kf_depth_map.idepth));
    max = * std::max_element(std::begin(last_kf_depth_map.idepth), std::end(last_kf_depth_map.idepth));
    //std::cout<<"[VIZ_MOSAIC: "<<this->last_kf_id<<"] norm_min: "<<min<<" norm_max: "<<max<<std::endl;

    /** Get the KF grayscale image **/
    cv::Mat last_kf_img = cv::Mat(last_kf.height, last_kf.width, CV_64FC1, cv::Scalar(0));
    memcpy(last_kf_img.data, last_kf.img.data(), last_kf.img.size()*sizeof(double));

    /** Draw the points in the keyframe image **/
    cv::Mat kf_color; last_kf_img.convertTo(kf_color, CV_8UC1, 255, 0);
    cv::cvtColor(kf_color, kf_color, cv::COLOR_GRAY2RGB);
    auto it_c = last_kf_depth_map.coord.begin();
    auto it_inv = last_kf_depth_map.idepth.begin();
    for (; it_c != last_kf_depth_map.coord.end(); ++it_c, ++it_inv)
    {
        cv::Vec3b color = ::eds::utils::valueToColor(*it_inv);
        cv::circle(kf_color, cv::Point2d(*it_c), 2.0, color, cv::FILLED, cv::LINE_AA);
    }

    /** Write the text **/
    const std::string text = "LAST KF";
    cv::putText(kf_color, text, cv::Point(5, kf_color.rows-5), 
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0,0,255), 0.1, cv::LINE_AA);

    /** Create the mosaic **/
    int n = std::ceil(bundles_window_size/2.0);
    cv::Mat mosaic = cv::Mat((n * last_kf.height/2 ) + last_kf.height, last_kf.width, CV_8UC3, cv::Scalar(0));

    /** Fill the mosaic **/
    kf_color.copyTo(mosaic(cv::Rect(0, 0, kf_color.cols, kf_color.rows)));
    for (size_t i=0; i<img_vector.size(); ++i)
    {
        img_vector[i].copyTo(mosaic(cv::Rect((i%2)*img_vector[i].cols,
                                    kf_color.rows + ((i/2)*img_vector[i].rows),
                                    img_vector[i].cols, img_vector[i].rows)));
    }

    return mosaic;
}

std::vector<::eds::mapping::Point3d> GlobalMap::getMap(const bool &only_active_points, const bool &remove_outliers)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    std::map<uint64_t, std::pair<uint64_t, int> > points_dict;

    /** Get the 3D points in the world frame **/
    uint64_t unique_id = 0;
    for (auto kf=this->camera_kfs.begin(); kf!=this->camera_kfs.end(); ++kf)
    {
        /** Host key frame pose **/
        Eigen::Vector3d &t_w_kf = kf->second.t;
        Eigen::Quaterniond &q_w_kf = kf->second.q;
        ::base::Transform3d T_w_kf(q_w_kf);
        T_w_kf.translation() = t_w_kf;

        /** Vector of points**/
        std::vector<eds::mapping::PointInfo> &points_vector = this->point_kfs.at(kf->first);

        /** For all the points in this keyframe **/
        int N = (only_active_points)? std::min((int)points_vector.size(), this->active_points_per_kf) : (int)points_vector.size();
        for (int i=0; i<N; ++i)
        {
            ::eds::mapping::Point2d &norm_coord = points_vector[i].norm_coord;
            double depth = 1.0/points_vector[i].inv_depth;
            Eigen::Vector3d point_3d(norm_coord.x() * depth, norm_coord.y() * depth, depth);

            /** Transform the point in the world frame **/
            Eigen::Vector3d p_world = T_w_kf * point_3d;

            /** Push the point in the map **/
            pcl::PointXYZ p; p.x = p_world[0]; p.y = p_world[1]; p.z = p_world[2];
            point_cloud->push_back(p);
            points_dict.insert({unique_id, std::make_pair(kf->first, i)});
            unique_id++;
        }
    }

    /** Outlier removal **/
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    if (this->config.sor_active)
    {
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> outlier_rm;
        outlier_rm.setInputCloud(point_cloud);
        outlier_rm.setRadiusSearch(this->config.sor_radius);
        outlier_rm.setMinNeighborsInRadius(this->config.sor_nb_points);
        outlier_rm.setKeepOrganized(true);
        outlier_rm.filter(*point_cloud_filtered);
    }
    else
    {
        point_cloud_filtered = point_cloud;
    }

    unique_id = 0;
    std::vector<eds::mapping::Point3d> points_map;
    for (auto &point : *point_cloud_filtered)
    {
        if (std::isnan(point.x))
        {
            if (remove_outliers)
            {
                /** Mark the points with infinite residual since it is an outlier **/
                std::pair<uint64_t, int> &outlier = points_dict.at(unique_id);
                std::vector<eds::mapping::PointInfo> &points_vector = this->point_kfs.at(outlier.first);
                auto it_p = points_vector.begin();
                std::advance(it_p, outlier.second);
                it_p->residual = base::NaN<double>();
            }
        }
        else
            points_map.push_back(eds::mapping::Point3d(point.x, point.y, point.z));
        unique_id++;
    }

    /** In case we want to remove outlier points **/
    if (remove_outliers)
    {
        for (auto &it : this->point_kfs)
        {
            std::vector<eds::mapping::PointInfo> &points_vector = it.second;
            this->eraseNaNResiduals(points_vector, only_active_points);
        }
    }

    return points_map;
}

void GlobalMap::getMap(base::samples::Pointcloud &points_map, const bool &only_active_points, const bool &remove_outliers)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    std::map<uint64_t, std::pair<uint64_t, int> > points_dict;

    /** Get the 3D points in the world frame **/
    uint64_t unique_id = 0;
    for (auto kf=this->camera_kfs.begin(); kf!=this->camera_kfs.end(); ++kf)
    {
        /** Host key frame pose **/
        Eigen::Vector3d &t_w_kf = kf->second.t;
        Eigen::Quaterniond &q_w_kf = kf->second.q;
        ::base::Transform3d T_w_kf(q_w_kf);
        T_w_kf.translation() = t_w_kf;

        /** Vector of points**/
        std::vector<eds::mapping::PointInfo> &points_vector = this->point_kfs.at(kf->first);

        /** For all the points in this keyframe **/
        int N = (only_active_points)? std::min((int)points_vector.size(), this->active_points_per_kf) : (int)points_vector.size();
        for (int i=0; i<N; ++i)
        {
            ::eds::mapping::Point2d &norm_coord = points_vector[i].norm_coord;
            double &color = points_vector[i].intensity;
            double depth = 1.0/points_vector[i].inv_depth;
            Eigen::Vector3d point_3d(norm_coord.x() * depth, norm_coord.y() * depth, depth);

            /** Transform the point in the world frame **/
            Eigen::Vector3d p_world = T_w_kf * point_3d;

            /** Points with intensity **/
            pcl::PointXYZI p; p.x = p_world[0]; p.y = p_world[1]; p.z = p_world[2];
            p.intensity = color;

            /** Push the point in the map **/
            point_cloud->push_back(p);
            /** Push the id in the dictionary **/
            points_dict.insert({unique_id, std::make_pair(kf->first, i)});
            unique_id++;
        }
    }

    /** Oulier removal **/
    pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);
    if (this->config.sor_active)
    {
        pcl::RadiusOutlierRemoval<pcl::PointXYZI> outlier_rm;
        outlier_rm.setInputCloud(point_cloud);
        outlier_rm.setRadiusSearch(this->config.sor_radius);
        outlier_rm.setMinNeighborsInRadius(this->config.sor_nb_points);
        outlier_rm.setKeepOrganized(true);
        outlier_rm.filter(*point_cloud_filtered);
        //std::cout<<"PCL size: "<<point_cloud->size()<<" FILTERED PCL size: "<<point_cloud_filtered->size()<<std::endl;
    }
    else
    {
        point_cloud_filtered = point_cloud;
    }

    /** Copy to the argumnet map type **/
    unique_id = 0;
    points_map.points.clear(); points_map.colors.clear();
    for (auto &point : *point_cloud_filtered)
    {
        if (std::isnan(point.x))
        {
            if (remove_outliers)
            {
                /** Mark the points with infinite residual since it is an outlier **/
                std::pair<uint64_t, int> &outlier = points_dict.at(unique_id);
                std::vector<eds::mapping::PointInfo> &points_vector = this->point_kfs.at(outlier.first);
                auto it_p = points_vector.begin();
                std::advance(it_p, outlier.second);
                it_p->residual = base::NaN<double>();
                //std::cout<<"[REMOVE_POINTS] MARKED KF: "<<outlier.first<<" #point: "<<outlier.second<<"["<<it_p->residual<<"]"<<std::endl;
            }
        }
        else
        {
            /** Push the point in the map **/
            points_map.points.push_back(::base::Vector3d(point.x, point.y, point.z));
            /** Push the color of the point **/
            points_map.colors.push_back(::base::Vector4d(point.intensity, point.intensity, point.intensity, 1.0));
        }
        unique_id++;
    }

    /** In case we want to remove outlier points **/
    if (remove_outliers)
    {
        for (auto &it : this->point_kfs)
        {
            std::vector<eds::mapping::PointInfo> &points_vector = it.second;
            this->eraseNaNResiduals(points_vector, only_active_points);
        }
    }
    /** Set map time for the last KF timestamp **/
    auto kf = this->camera_kfs.rbegin();
    points_map.time = kf->second.ts;
}
 
void GlobalMap::getIDepthMap(const uint64_t &kf_id, ::eds::mapping::IDepthMap2d &depthmap, const bool &only_active_points, const bool &remove_outliers)
{
    /** Clean depthmap arguments **/
    depthmap.clear();

    /** Check if the kf id is a valid id **/
    if (this->camera_kfs.find(kf_id) == this->camera_kfs.end())
        return;

    /** Get keyframe pose **/
    eds::mapping::KeyFrameInfo &kf = this->camera_kfs.at(kf_id);
    ::base::Transform3d T_kf_w(kf.q); T_kf_w.translation() = kf.t;
    T_kf_w = T_kf_w.inverse();

    /** Get keyframe intrinsics [fx, fy, cx, cy] **/
    std::vector<double> intrinsics{kf.K(0, 0), kf.K(1, 1), kf.K(0, 2), kf.K(1, 2)};

    /** Image dimension **/
    cv::Size kf_size = cv::Size(kf.width, kf.height);

    /** Compute the inverse depth 2D Map in the keyframe coordinates **/
    this->getIDepthMap(T_kf_w, intrinsics, kf_size, depthmap, only_active_points, remove_outliers);
}

void GlobalMap::getIDepthMap(const ::base::Transform3d &T_kf_w, const std::vector<double> &intrinsics, const cv::Size &img_size, ::eds::mapping::IDepthMap2d &depthmap, const bool &only_active_points, const bool &remove_outliers)
{
    /** Get the map **/
    std::vector<eds::mapping::Point3d> map_3d = this->getMap(only_active_points, remove_outliers);

    /** Get keyframe intrinsics **/
    const double &fx = intrinsics[0]; const double &fy = intrinsics[1];
    const double &cx = intrinsics[2]; const double &cy = intrinsics[3];

    /** For all the points in map **/
    for (auto &it : map_3d)
    {
        /** Point in the keyframe frame **/
        Eigen::Vector3d kp = T_kf_w * Eigen::Vector3d(it.x(), it.y(), it.z());

        /** Project the point on the frame. x-y pixel coord, z is inverse depth **/
        cv::Point3d px(fx * (kp[0]/kp[2]) + cx, fy * (kp[1]/kp[2]) + cy, 1.0/kp[2]);

        /** Check if the projected point is in the frame **/
        bool inlier = ((px.x >= 0.0) and (px.x < img_size.width)) and ((px.y >= 0.0) and (px.y < img_size.height));

        /** Push the point in to the vector **/
        if (inlier)
        {
            depthmap.coord.push_back(::eds::mapping::Point2d(px.x, px.y)); //pixel coordinates
            depthmap.idepth.push_back(px.z); //inverse depth
        }
    }
}

void GlobalMap::getKFPoses(std::vector<::base::samples::RigidBodyState> &poses)
{
    poses.clear();
    for (auto &it : this->camera_kfs)
    {
        ::base::samples::RigidBodyState rbs;
        rbs.time = it.second.ts;
        rbs.orientation = it.second.q;
        rbs.position = it.second.t;
        rbs.targetFrame = "world";
        rbs.sourceFrame = std::to_string(it.first);
        poses.push_back(rbs);
    }
}

std::vector<cv::Point3d> GlobalMap::projectMapOnKF(const std::vector<::eds::mapping::Point3d> &map, const uint64_t &kf_id)
{
    std::vector<cv::Point3d> coord; // x-y pixel coord and inverse depth

    /** Check if the kf id is a valid id **/
    if (this->camera_kfs.find(kf_id) == this->camera_kfs.end())
        return coord;

    /** Get keyframe pose **/
    eds::mapping::KeyFrameInfo &kf = this->camera_kfs.at(kf_id);
    ::base::Transform3d T_kf_w(kf.q); T_kf_w.translation() = kf.t;
    T_kf_w = T_kf_w.inverse();

    /** Get keyframe intrinsics **/
    double &fx = kf.K(0, 0); double &fy = kf.K(1, 1);
    double &cx = kf.K(0, 2); double &cy = kf.K(1, 2);
 
    /** For all the points in map **/
    for (auto it : map)
    {
        /** Point in the keyframe frame **/
        Eigen::Vector3d kp = T_kf_w * Eigen::Vector3d(it.x(), it.y(), it.z());

        /** Project the point on the frame. x-y pixel coord, z is inverse depth **/
        cv::Point3d px(fx * (kp[0]/kp[2]) + cx, fy * (kp[1]/kp[2]) + cy, 1.0/kp[2]);

        /** Check if the projected point is in the frame **/
        bool inlier = ((px.x > -1.0) and (px.x < kf.width)) and ((px.y > -1.0) and (px.y < kf.height));

        /** Push the point in to the vector **/
        if (inlier) coord.push_back(px);
    }

    return coord;
}

void GlobalMap::setNumPointsInTracking(const uint16_t &height, const uint16_t &width, const double &percent_tracking_points, const size_t &window_size)
{
    if (percent_tracking_points > 0)
    {
        this->num_points_in_tracking = (percent_tracking_points/100.0) * (height * width);
        this->total_active_points = (this->bundles_config.percent_points/100.0) * this->num_points_in_tracking;
    }
    else
    {
        this->total_active_points = 2000.0; this->num_points_in_tracking = 2000.0;
        this->active_points_per_kf = this->total_active_points / window_size;
    }
    std::cout<<"[GLOBAL_MAP] COMPUTING TOTAL_ACTIVE_POINTS["<<this->total_active_points<<"]"<<std::endl;
}

void GlobalMap::updateNumberActivePoints(const size_t &window_size)
{
    this->active_points_per_kf = this->total_active_points / window_size;
    std::cout<<"[GLOBAL_MAP] WINDOWS SIZE: "<<window_size<<" TOTAL_ACTIVE_POINTS["<<this->total_active_points<<"] "<<this->active_points_per_kf<<" PER KF"<<std::endl;
}

void GlobalMap::orderPoints(std::vector<PointInfo> &points, const ::eds::mapping::POINTS_SELECT_METHOD &method,  const int &num_points, const int &block_size)
{
    switch (method)
    {
    case RANDOM:
    {
        std::cout<<"[GLOBAL_MAP] RANDOM SELECT"<<std::endl;
        /** Random unique N points in the KeyFrame [N = this->active_points_per_kf] **/
        ::eds::utils::random_unique(points.begin(), points.end(), num_points);
        break;
    }

    case RESIDUALS:
    {
        std::cout<<"[GLOBAL_MAP] RESIDUALS SELECT"<<std::endl;
        std::sort(points.begin(), points.end(), ::eds::mapping::PointInfo::residual_order);
        std::cout<<"min: "<<points[0].residual<<" max: "<<points[points.size()-1].residual<<std::endl;
        break;
    }
    case GRADIENT:
    {
        std::cout<<"[GLOBAL_MAP] GRADIENT SELECT"<<std::endl;
        std::sort(points.begin(), points.end(), ::eds::mapping::PointInfo::gradient_order);
        std::cout<<"max: "<<points[0].gradient<<" min: "<<points[points.size()-1].gradient<<std::endl;
        break;
    }

    case SPARSE_GRADIENT:
    {
        /** Points are already in sparse gradient
         * but we need to sort per each block **/
        std::cout<<"[GLOBAL_MAP] SPARSE GRADIENT SELECT"<<std::endl;
        int num_blocks = points.size()/block_size;
        int points_per_block = std::max(1, num_points/num_blocks);//at least on epoint per block
        std::vector<PointInfo> new_points, rest_points;//(points.size());
        std::cout<<"total points: "<<points.size()<<", block_size: "<<block_size<<", num_blocks: "<<num_blocks
        <<", points_per_block: "<<points_per_block<<std::endl;

        std::vector<PointInfo>::iterator it_begin = points.begin();
        for (int i=0; i<num_blocks+1; ++i, it_begin = it_begin+block_size)
        {
            std::vector<PointInfo>::iterator it_end = it_begin;
            int step_advance = std::min(block_size, (int)std::distance(it_begin, points.end()));
            //std::cout<<"["<<i<<"]step_advance: "<<step_advance<<std::endl;
            std::advance(it_end, step_advance);
            std::sort(it_begin, it_end, ::eds::mapping::PointInfo::gradient_order);
            int step_copy = std::min(points_per_block, step_advance);
            //std::cout<<"step_copy: "<<step_copy<<std::endl;
            std::copy_n(it_begin, step_copy, std::back_inserter(new_points));
            int rest = std::max(0, step_advance - step_copy);
            //std::cout<<"rest: "<<rest<<std::endl;
            std::copy_n(it_begin+step_copy, rest, std::back_inserter(rest_points));
            //std::copy_backward(it_begin+step_copy, it_begin+(step_copy+rest), new_points.end()-(i*points_per_block));
        }

        /** Make the new ordered vector of points **/
        new_points.insert(
            new_points.end(),
              std::make_move_iterator(rest_points.begin()),
              std::make_move_iterator(rest_points.end())
        );
        /** Rewrite the vector of points **/
        points = new_points;

        break;
    }

    default:
        break;
    }
}


void GlobalMap::depthCompletion(std::vector<PointInfo> &points)
{
    /** If the number of active points is the total size
     * there is not depth to complete **/
    if (this->active_points_per_kf >= (int)points.size())
        return;

    /** Copy the pixel coordinates **/
    std::vector<::eds::mapping::Point2d> coord;
    for (int i=0; i<this->active_points_per_kf; ++i)
        coord.push_back(points[i].coord);

    /** Create the KDTree to search **/
    ::eds::mapping::KDTree<eds::mapping::Point2d> kdtree(coord);

    auto it_point = points.begin();
    std::advance(it_point, std::min((int)points.size(), this->active_points_per_kf));

    /** Iterate for the points not use in PBA optimization **/
    for (; it_point != points.end(); ++it_point)
    {
	    const int idx = kdtree.nnSearch(it_point->coord);
        it_point->inv_depth = points[idx].inv_depth;
    }
}

void GlobalMap::outliersRemoval(const double &threshold, const bool &only_active_points)
{
    /** Iterates all the points and remove the ones with residual
     * higher than the selected threshold. normaly mean + 3*sigma **/

    for (auto &it : this->point_kfs)
    {
        std::vector<eds::mapping::PointInfo> &points_vector = it.second;
        if (points_vector.size() < this->total_active_points/this->point_kfs.size())
        {
            std::cout<<"[OUTLIER REMOVAL] ABORT KF["<<it.first<<"] MIN NUMBER OF POINTS PER KF REACHED: "<<this->total_active_points/this->point_kfs.size()<<std::endl;
            continue;
        }
        int N = (only_active_points)? std::min((int)points_vector.size(), this->active_points_per_kf) : (int)points_vector.size();
        auto end_point = points_vector.begin();
        std::advance(end_point, N);

        int i=0;
        int removed_points = 0;
        for (auto point = points_vector.begin(); point != end_point && i < N;)
        {
            if (fabs(point->residual) > threshold)
            {
                point = points_vector.erase(point);
                removed_points++;
            }
            else
            {
                point++;
            }
            i++;
        }
        std::cout<<"[GLOBAL_MAP] OUTLIER REMOVAL KF["<<it.first<<"] REMOVED "<<removed_points<<" points"<<std::endl;
    }
}

void GlobalMap::eraseNaNResiduals(std::vector<eds::mapping::PointInfo> &points_vector, const bool &only_active_points)
{
    int N = (only_active_points)? std::min((int)points_vector.size(), this->active_points_per_kf) : (int)points_vector.size();
    auto end_point = points_vector.begin();
    std::advance(end_point, N);
    int i=0;
    for (auto point = points_vector.begin(); point != end_point && i < N;)
    {
        if (base::isNaN<double>(point->residual))
        {
            point = points_vector.erase(point);
        }
        else
        {
            point++;
        }
        i++;
    }
}

void GlobalMap::cleanMap(const uint8_t &num_points_pixel, const bool &only_active_points)
{
    /** GOAL: Project the points into the last KF
     * We get the point in 3D and projected in the last KF, stored a pointer to the PointInfo **/

    /** Get T_last_kf_w transformation**/
    eds::mapping::KeyFrameInfo &last_kf = this->camera_kfs.at(this->last_kf_id);
    ::base::Transform3d T_last_kf_w(last_kf.q); T_last_kf_w.translation() = last_kf.t;
    T_last_kf_w = T_last_kf_w.inverse();

    /** Get last keyframe intrinsics **/
    double &fx = last_kf.K(0, 0); double &fy = last_kf.K(1, 1);
    double &cx = last_kf.K(0, 2); double &cy = last_kf.K(1, 2);

    /** The dictionary to look up with H * W elements **/
    std::map<uint32_t, std::vector<eds::mapping::PointShortInfo> > projected_points;
    for (uint32_t i=0; i<last_kf.width * last_kf.height; ++i)
    {
        std::vector<eds::mapping::PointShortInfo> points;
        projected_points.insert({i, points});
    }

    /** For all the points in each keyframe **/
    for (auto kf=this->camera_kfs.begin(); kf!=this->camera_kfs.end(); ++kf)
    {
        /** Host key frame pose **/
        Eigen::Vector3d &t_w_kf = kf->second.t;
        Eigen::Quaterniond &q_w_kf = kf->second.q;
        ::base::Transform3d T_w_kf(q_w_kf);
        T_w_kf.translation() = t_w_kf;

        /** Host KF to last KF transformation **/
        ::base::Transform3d T_last_kf_kf = T_last_kf_w * T_w_kf;

        /** Vector of points**/
        std::vector<eds::mapping::PointInfo> &points_vector = this->point_kfs.at(kf->first);

        /** For all the points in this keyframe **/
        int N = (only_active_points)? std::min((int)points_vector.size(), this->active_points_per_kf) : (int)points_vector.size();
        for (int i=0; i<N; ++i)
        {
            /** 3D point in keyframe **/
            ::eds::mapping::Point2d &norm_coord = points_vector[i].norm_coord;
            double depth = 1.0/points_vector[i].inv_depth;
            Eigen::Vector3d point_3d(norm_coord.x() * depth, norm_coord.y() * depth, depth);

            /** Transform the point in the last keyframe coordinate **/
            Eigen::Vector3d p = T_last_kf_kf * point_3d;

            /** Project the point on the last frame. x-y pixel coord, z is inverse depth **/
            cv::Point3d px(fx * (p[0]/p[2]) + cx, fy * (p[1]/p[2]) + cy, 1.0/p[2]);

            /** Check if the projected point is in the frame **/
            bool inlier = ((px.x >= 0.0) and (px.x < last_kf.width)) and ((px.y >= 0.0) and (px.y < last_kf.height));

            if (inlier)
            {
                /** Insert the point into the lookup map **/
                uint32_t unique_id = (int)px.x + ((int)px.y * last_kf.width);
                std::vector<eds::mapping::PointShortInfo> &points_ref = projected_points.at(unique_id);
                points_ref.push_back(eds::mapping::PointShortInfo(kf->first, i, points_vector[i].residual));//pair kf_id, point_position in avector
            }
        }
    }

    for (auto it_px=projected_points.begin(); it_px!=projected_points.end(); ++it_px)
    {
        std::vector<eds::mapping::PointShortInfo> &points = it_px->second;
        if (points.size() > 1)
        {
            //std::cout<<"[GLOBAL_MAP] CLEAN POINT_ID["<<it_px->first<<"] vector size: "<<points.size()<<std::endl;
            std::sort(points.begin(), points.end(), ::eds::mapping::PointShortInfo::residual_order);
            for (uint i=1; i<points.size(); ++i)
            {
                /** Get the KF of the point **/
                uint64_t &kf_id = points[i].kf_id;

                /** Get the vector of points for this keyframe **/
                std::vector<eds::mapping::PointInfo> &points_vector = this->point_kfs.at(kf_id);

                /** Mark the point as NaN residual **/
                uint32_t &point_id = points[i].id;
                auto it_p = points_vector.begin()+point_id;
                //std::cout<<"\tKF["<<kf_id<<"] REMOVED: "<<i <<" with residual: "<<points[i].residual <<":"<<it_p->residual<<std::endl;
                it_p->residual = ::base::NaN<double>();
            }
        }
    }

    /** Remove the points that has residual equal NaN **/
    for (auto kf=this->camera_kfs.begin(); kf!=this->camera_kfs.end(); ++kf)
    {
        /** Vector of points**/
        std::vector<eds::mapping::PointInfo> &points_vector = this->point_kfs.at(kf->first);
        this->eraseNaNResiduals(points_vector, only_active_points);
    }

}
