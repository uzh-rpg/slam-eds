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

#include "KeyFrame.hpp"

#include <eds/utils/Colormap.hpp>
#include <opencv2/features2d.hpp>

using namespace eds::tracking;

KeyFrame::KeyFrame(const ::eds::calib::Camera &cam, const ::eds::calib::Camera &newcam, const std::string &distortion_model)
{
    this->K = cam.K.clone();
    this->D = cam.D.clone();
    this->R_rect = newcam.R * cam.R.t();
    this->K_ref =  newcam.K.clone();
    this->distortion_model = distortion_model;

    if (distortion_model.compare("equidistant") != 0)
    {
        cv::initUndistortRectifyMap(this->K, this->D,
                            this->R_rect, this->K_ref,
                            newcam.size,//undistorted image size
                            CV_32FC1,
                            this->mapx, this->mapy);
    }
    else
    {
        cv::fisheye::initUndistortRectifyMap(this->K, this->D,
                            this->R_rect, this->K_ref,
                            newcam.size,//undistorted image size
                            CV_32FC1,
                            this->mapx, this->mapy);
    }

    /* Scale to one (no resize) **/
    this->out_scale[0] = 1.0; this->out_scale[1] = 1.0;

    /** Check if the input image should be downscaled **/
    if ((newcam.out_size.height != 0 || newcam.out_size.width != 0) || (newcam.out_size.height != newcam.size.height || newcam.out_size.width != newcam.size.width))
    {
        /** Downrescale the input **/
        this->out_scale[0] = (double)newcam.size.width / (double)newcam.out_size.width;
        this->out_scale[1] = (double)newcam.size.height / (double)newcam.out_size.height;
        this->K.at<double>(0,0) /=  this->out_scale[0]; this->K.at<double>(1,1) /=  this->out_scale[1];
        this->K.at<double>(0,2) /=  this->out_scale[0]; this->K.at<double>(1,2) /=  this->out_scale[1];
        this->K_ref.at<double>(0,0) /=  this->out_scale[0]; this->K_ref.at<double>(1,1) /=  this->out_scale[1];
        this->K_ref.at<double>(0,2) /=  this->out_scale[0]; this->K_ref.at<double>(1,2) /=  this->out_scale[1];
    }

    std::cout<<"** KEYFRAME: CAMERA CALIB: **"<<std::endl;
    std::cout<<"Model: "<<distortion_model<<std::endl;
    std::cout<<"Size: "<<newcam.size<<std::endl;
    std::cout<<"Out Size: "<<newcam.out_size<<std::endl;
    std::cout<<"K:\n"<<this->K<<std::endl;
    std::cout<<"D:\n"<<this->D<<std::endl;
    std::cout<<"R:\n"<<this->R_rect<<std::endl;
    std::cout<<"K_ref:\n"<<this->K_ref<<std::endl;
    std::cout<<"OUT SCALE ["<<this->out_scale[0]<<","<<this->out_scale[1]<<"]"<<std::endl;

    std::cout<<"mapx: "<<this->mapx.rows<<" x "<<this->mapx.cols<<std::endl;
    std::cout<<"mapy: "<<this->mapy.rows<<" x "<<this->mapy.cols<<std::endl;
}

KeyFrame::KeyFrame(const uint64_t &idx, const ::base::Time &time, cv::Mat &img, ::eds::mapping::IDepthMap2d &depthmap,
                   const ::eds::calib::CameraInfo &cam_info, const ::eds::mapping::Config &map_info, const float &percent_points,
                   const ::base::Affine3d &T, const cv::Size &out_size)
{
    cv::Mat K, D, R_rect, P;
    R_rect  = cv::Mat_<double>::eye(3, 3);
    K = cv::Mat_<double>::eye(3, 3);
    K.at<double>(0,0) = cam_info.intrinsics[0];
    K.at<double>(1,1) = cam_info.intrinsics[1];
    K.at<double>(0,2) = cam_info.intrinsics[2];
    K.at<double>(1,2) = cam_info.intrinsics[3];

    D = cv::Mat_<double>::zeros(4, 1);
    for (size_t i=0; i<cam_info.D.size(); ++i)
    {
        D.at<double>(i, 0) = cam_info.D[i];
    }

    if (cam_info.P.size() == 12)
    {
        P = cv::Mat_<double>::zeros(4, 4);
        for (auto row=0; row<P.rows; ++row)
        {
            for (auto col=0; col<P.cols; ++col)
            {
                P.at<double>(row, col) = cam_info.P[(P.cols*row)+col];
            }
        }
    }

    if (cam_info.R.size() == 9)
    {
        for (auto row=0; row<R_rect.rows; ++row)
        {
            for (auto col=0; col<R_rect.cols; ++col)
            {
                R_rect.at<double>(row, col) = cam_info.R[(R_rect.cols*row)+col];
            }
        }
    }

    CANDIDATE_POINT_METHOD points_selection_method = MEDIAN;
    if (percent_points > 0.0 && percent_points < 100.0) points_selection_method = MAX;

    (*this) = KeyFrame(idx, time, img, depthmap, K, D, R_rect, P,
                    cam_info.distortion_model, points_selection_method,
                    map_info.min_depth, map_info.max_depth,
                    map_info.convergence_sigma2_thresh, percent_points,
                    T, out_size);

}

KeyFrame::KeyFrame(const uint64_t &idx, const ::base::Time &time, cv::Mat &img,
            ::eds::mapping::IDepthMap2d &depthmap,
            cv::Mat &K, cv::Mat &D, cv::Mat &R_rect, cv::Mat &P,
            const std::string &distortion_model,
            const CANDIDATE_POINT_METHOD points_selection_method,
            const double &min_depth, const double &max_depth, const double &convergence_sigma2_thresh,
            const float &percent_points,
            const ::base::Affine3d &T,
            const cv::Size &out_size)
            :idx(idx), time(time), distortion_model(distortion_model), percent_points(percent_points), T_w_kf(T)
{
    std::cout<<"[KEY_FRAME] IDX: "<<idx<<" TIME: "<<time.toMicroseconds() <<std::endl;
    std::cout<<"[KEY_FRAME] img.type: "<<eds::utils::type2str(img.type()) <<std::endl;

    if (P.total()>0)
        this->K_ref = P(cv::Rect(0,0,3,3));
    if (R_rect.total()>0)
        this->R_rect = R_rect.clone();

    cv::Size size = img.size();

    /** Distortion matrices and model **/
    if (K.total() > 0 && D.total() > 0)
    {
        this->K = K.clone();
        this->D = D.clone();
        if (this->K_ref.total() == 0)
        {
            if (distortion_model.compare("equidistant") != 0)
            {
                /** radtan model **/
                this->K_ref = cv::getOptimalNewCameraMatrix(this->K, this->D, cv::Size(size.width, size.height), 0.0);
            }
            else
            {
                /** Kalibr equidistant model is opencv fisheye **/
                cv::fisheye::estimateNewCameraMatrixForUndistortRectify(this->K, this->D, cv::Size(size.width, size.height), this->R_rect, this->K_ref);
            }
        }

        if (distortion_model.compare("equidistant") != 0)
        {
            /** radtan model **/
            std::cout<<"[KEY_FRAME] undistort radtan model"<<std::endl;
            cv::undistort(img, this->img, this->K, this->D, this->K_ref);
        }
        else
        {
            /** Kalibr equidistant model is opencv fisheye **/
            std::cout<<"[KEY_FRAME] undistort equidistant model"<<std::endl;
            cv::fisheye::undistortImage(img, this->img, this->K, this->D, this->K_ref);
        }
    }
    else
    {
        this->K_ref = K.clone();
    }

    /** Check if the input image should be rescale **/
    if ((out_size.height == 0 || out_size.width == 0) || (out_size.height > size.height || out_size.width > size.width))
    {
        this->out_scale[0] = 1.0; this->out_scale[1] = 1.0;
        std::cout<<"[KEY_FRAME] out_scale["<<this->out_scale[0]<<","<<this->out_scale[1]<<"] out_size: "<<size<<std::endl;
    }
    else
    {
        /** Rescale the input **/
        this->out_scale[0] = (double)size.width / (double)out_size.width;
        this->out_scale[1] = (double)size.height / (double)out_size.height;
        std::cout<<"[KEY_FRAME] out_scale["<<this->out_scale[0]<<","<<this->out_scale[1]<<"] out_size: "<<out_size<<std::endl;
        this->K.at<double>(0,0) /=  this->out_scale[0]; this->K.at<double>(1,1) /=  this->out_scale[1];
        this->K.at<double>(0,2) /=  this->out_scale[0]; this->K.at<double>(1,2) /=  this->out_scale[1];
        this->K_ref.at<double>(0,0) /=  this->out_scale[0]; this->K_ref.at<double>(1,1) /=  this->out_scale[1];
        this->K_ref.at<double>(0,2) /=  this->out_scale[0]; this->K_ref.at<double>(1,2) /=  this->out_scale[1];
        cv::resize(this->img, this->img, out_size, cv::INTER_CUBIC);
        size = this->img.size();
    }

    /** Store the color image in uint8_t format **/
    this->uint8_img = this->img.clone();

    /** Convert the grayscale in case of color and normalize the image **/
    if (this->img.channels() > 1) cv::cvtColor(this->img, this->img, cv::COLOR_RGB2GRAY);

    /** Convert image to float **/
    this->img.convertTo(this->img, CV_64FC1);
    double min, max; cv::minMaxLoc(this->img, &min, &max);
    this->img = (this->img - min) / (max - min);

    /** Img in std vector **/
    this->img_data.resize(this->img.rows * this->img.cols);
    std::copy(this->img.begin<double>(), this->img.end<double>(), this->img_data.begin());

    /** Compute log image for the number of pyramids levels **/
    cv::Mat log_image;
    cv::log(this->img + KeyFrame::log_eps, log_image);
    this->log_img.push_back(log_image);

    /***********************/
    /** ONE PYRAMID LEVEL **/
    {
        /** Compute the gradient from the log normalized (0-1) image **/
        uint8_t i = 0;
        cv::Mat grad_x, grad_y, grad_xy;
        std::vector<cv::Mat> grad_channels;
        cv::Sobel(this->log_img[i], grad_x, CV_64FC1, 1, 0, 7); // derivative along x-axis
        cv::Sobel(this->log_img[i], grad_y, CV_64FC1, 0, 1, 7); // derivative along y-axis
        grad_channels.push_back(grad_x); grad_channels.push_back(grad_y);
        cv::merge(grad_channels, grad_xy);
        this->img_grad.push_back(grad_xy);

        /** Gradient frame in std vector [\Nabla_x, \Nabla_y]**/
        this->grad_frame.clear();
        for (int row=0; row<grad_xy.rows; ++row)
        {
            for (int col=0; col<grad_xy.cols; ++col)
            {
                this->grad_frame.push_back(grad_x.at<double>(row,col));
                this->grad_frame.push_back(grad_y.at<double>(row,col));
            }
        }

        /** Magnitude of the gradient (gradient norm). Magnitude is in the first element of cartToPolar **/
        cv::Mat mag_img, ang_img;
        cv::cartToPolar(grad_x, grad_y, mag_img, ang_img, true);
        this->mag.push_back(mag_img);

        /** Compute the candidate points coordinates **/
        float target_num_points = (this->img.rows*this->img.cols)*(this->percent_points/100.0);
        if (target_num_points > 0 && (target_num_points < (this->img.rows*this->img.cols)))
            this->candidatePoints(this->coord, cv::Size(this->adaptive_width_patch_factor * size.width, this->adaptive_height_patch_factor * size.height),
                                points_selection_method, target_num_points);
        else
            this->candidatePoints(this->coord, cv::Size(this->adaptive_width_patch_factor * size.width, this->adaptive_height_patch_factor * size.height),
                                eds::tracking::MEDIAN);

        /** Candidate points normalized coordinates **/
        double fx, fy, cx, cy;
        fx = this->K_ref.at<double>(0,0); fy = this->K_ref.at<double>(1,1);
        cx = this->K_ref.at<double>(0,2); cy = this->K_ref.at<double>(1,2);
        for (auto it=this->coord.begin(); it!=this->coord.end(); ++it)
        {
            cv::Point2d c;
            c.x = (it->x - cx)/fx; //x-coordinate
            c.y = (it->y - cy)/fy; //y-coordinate
            this->norm_coord.push_back(c);
        }

        /** Gradient of the candidate points **/
        this->grad.clear();
        for (auto it=this->coord.begin(); it!=this->coord.end(); ++it)
        {
            this->grad.push_back(cv::Point2d(grad_x.at<double>(*it), grad_y.at<double>(*it)));
        }

        /** Create the inverse depth map **/
        this->setDepthMap(depthmap, min_depth, max_depth, convergence_sigma2_thresh, {1.0, 1.0});

        /** Create the point patches. Adaptive patch size defines as 7 for 240x180 **/
        eds::utils::splitImageInPatches(this->img, this->coord, this->patches, this->adaptive_patch_factor * (size.width * size.height));

        /** Create the point bundle patches **/
        eds::utils::computeBundlePatches(this->patches, this->bundle_patches);

        /** Initial residuals **/
        this->residuals.resize(this->coord.size(), 0.0);

        /** Initial tracks **/
        this->tracks.resize(this->coord.size(), Eigen::Vector2d::Zero());

        /** Initial Optical flow **/
        this->flow.resize(this->coord.size(), Eigen::Vector2d::Zero());
    }

    /** Update the percentage of the actual tracking points **/
    this->percent_points = (this->coord.size() * 100.0)/(size.height*size.width);

    std::cout<<"[KEY_FRAME] norm_coord["<<std::addressof(this->norm_coord)<<"] size: "<<this->norm_coord.size()<<std::endl;
    std::cout<<"[KEY_FRAME] grad["<<std::addressof(this->grad)<<"] size: "<<this->grad.size() <<std::endl;
    std::cout<<"[KEY_FRAME] inv_depth["<<std::addressof(this->inv_depth)<<"] size: "<<this->inv_depth.size()<<std::endl;
    std::cout<<"[KEY_FRAME] img["<<std::addressof(this->img)<<"] size: "<<this->img.size()<<std::endl;
    std::cout<<"[KEY_FRAME] img_data["<<std::addressof(this->img_data)<<"] size: "<<this->img_data.size()<<std::endl;
    std::cout<<"[KEY_FRAME] grad_frame["<<std::addressof(this->grad_frame)<<"] size: "<<this->grad_frame.size()<<std::endl;
    std::cout<<"[KEY_FRAME] selected points: "<<this->percent_points<<"[%]"<<std::endl;
}

void KeyFrame::create(const uint64_t &idx, const ::base::Time &time, cv::Mat &img, ::eds::mapping::IDepthMap2d &depthmap,
                   const ::eds::mapping::Config &map_info, const float &percent_points,
                   const ::base::Affine3d &T, const cv::Size &out_size)
{
    CANDIDATE_POINT_METHOD points_selection_method = MEDIAN;
    if (percent_points > 0.0 && percent_points < 100.0) points_selection_method = MAX;

    return this->create(idx, time, img, depthmap, points_selection_method,
                    map_info.min_depth, map_info.max_depth,
                    map_info.convergence_sigma2_thresh, percent_points,
                    T, out_size);

}
void KeyFrame::create(const uint64_t &idx, const ::base::Time &time, cv::Mat &img,
            ::eds::mapping::IDepthMap2d &depthmap,
            const CANDIDATE_POINT_METHOD points_selection_method,
            const double &min_depth, const double &max_depth, const double &convergence_sigma2_thresh,
            const float &percent_points,
            const ::base::Affine3d &T,
            const cv::Size &out_size)
{
    /** Clean before creating a new Keyframe image**/
    this->clear();

    this->idx = idx; this->time = time; this->percent_points = percent_points; this->T_w_kf = T;
    std::cout<<"[KEY_FRAME] IDX: "<<idx<<" TIME: "<<time.toMicroseconds() <<std::endl;
    std::cout<<"[KEY_FRAME] img.type: "<<eds::utils::type2str(img.type()) <<std::endl;

    /** IMPORTANT: Image is already undistorted **/
    this->img = img.clone();
    cv::Size size = img.size();

    /** Check if the input image should be rescale **/
    if (this->out_scale[0] != 1 || this->out_scale[1] != 1)
    {
        cv::resize(this->img, this->img, out_size, cv::INTER_CUBIC);
        size = this->img.size();
    }

    /** Store the color image in uint8_t format **/
    this->uint8_img = this->img.clone();

    /** Convert the grayscale in case of color and normalize the image **/
    if (this->img.channels() > 1) cv::cvtColor(this->img, this->img, cv::COLOR_RGB2GRAY);
    this->img.convertTo(this->img, CV_64FC1);
    double min, max; cv::minMaxLoc(this->img, &min, &max);
    this->img = (this->img - min) / (max - min);

    /** Img in std vector **/
    this->img_data.resize(this->img.rows * this->img.cols);
    std::copy(this->img.begin<double>(), this->img.end<double>(), this->img_data.begin());

    /** Compute log image for the number of pyramids levels **/
    cv::Mat log_image;
    cv::log(this->img + KeyFrame::log_eps, log_image);
    this->log_img.push_back(log_image);

    /***********************/
    /** ONE PYRAMID LEVEL **/
    {
        /** Compute the gradient from the log normalized (0-1) image **/
        uint8_t i = 0;
        cv::Mat grad_x, grad_y, grad_xy;
        std::vector<cv::Mat> grad_channels;
        cv::Sobel(this->log_img[i], grad_x, CV_64FC1, 1, 0, 3); // derivative along x-axis
        cv::Sobel(this->log_img[i], grad_y, CV_64FC1, 0, 1, 3); // derivative along y-axis
        grad_channels.push_back(grad_x); grad_channels.push_back(grad_y);
        cv::merge(grad_channels, grad_xy);
        this->img_grad.push_back(grad_xy);

        /** Gradient frame in std vector [\Nabla_x, \Nabla_y]**/
        this->grad_frame.clear();
        for (int row=0; row<grad_xy.rows; ++row)
        {
            for (int col=0; col<grad_xy.cols; ++col)
            {
                this->grad_frame.push_back(grad_x.at<double>(row,col));
                this->grad_frame.push_back(grad_y.at<double>(row,col));
            }
        }

        /** Magnitude of the gradient (gradient norm). Magnitude is in the first element of cartToPolar **/
        cv::Mat mag_img, ang_img;
        cv::cartToPolar(grad_x, grad_y, mag_img, ang_img, true);
        this->mag.push_back(mag_img);

        /** Compute the candidate points coordinates **/
        float target_num_points = (this->img.rows*this->img.cols)*(this->percent_points/100.0);
        if (target_num_points > 0 && (target_num_points < (this->img.rows*this->img.cols)))
            this->candidatePoints(this->coord, cv::Size(20, 20), points_selection_method, target_num_points);
        else
            this->candidatePoints(this->coord, cv::Size(20, 20), eds::tracking::MEDIAN);

        /** Candidate points normalized coordinates **/
        double fx, fy, cx, cy;
        fx = this->K_ref.at<double>(0,0); fy = this->K_ref.at<double>(1,1);
        cx = this->K_ref.at<double>(0,2); cy = this->K_ref.at<double>(1,2);
        for (auto it=this->coord.begin(); it!=this->coord.end(); ++it)
        {
            cv::Point2d c;
            c.x = (it->x - cx)/fx; //x-coordinate
            c.y = (it->y - cy)/fy; //y-coordinate
            this->norm_coord.push_back(c);
        }

        /** Gradient of the candidate points **/
        this->grad.clear();
        for (auto it=this->coord.begin(); it!=this->coord.end(); ++it)
        {
            this->grad.push_back(cv::Point2d(grad_x.at<double>(*it), grad_y.at<double>(*it)));
        }

        /** Create the inverse depth map **/
        this->setDepthMap(depthmap, min_depth, max_depth, convergence_sigma2_thresh, {1.0, 1.0});

        /** Create the point patches. Adaptive patch size defines as 7 for 240x180 **/
        eds::utils::splitImageInPatches(this->img, this->coord, this->patches, this->adaptive_patch_factor * (size.width * size.height));

        /** Create the point bundle patches **/
        eds::utils::computeBundlePatches(this->patches, this->bundle_patches);

        /** Initial residuals **/
        this->residuals.resize(this->coord.size(), 0.0);

        /** Initial tracks **/
        this->tracks.resize(this->coord.size(), Eigen::Vector2d::Zero());

        /** Initial Optical flow **/
        this->flow.resize(this->coord.size(), Eigen::Vector2d::Zero());
    }

    this->cleanPoints(0.7);

    /** Update the percentage of the actual tracking points **/
    this->percent_points = (this->coord.size() * 100.0)/(size.height*size.width);

    std::cout<<"[KEY_FRAME] norm_coord["<<std::addressof(this->norm_coord)<<"] size: "<<this->norm_coord.size()<<std::endl;
    std::cout<<"[KEY_FRAME] grad["<<std::addressof(this->grad)<<"] size: "<<this->grad.size() <<std::endl;
    std::cout<<"[KEY_FRAME] inv_depth["<<std::addressof(this->inv_depth)<<"] size: "<<this->inv_depth.size()<<std::endl;
    std::cout<<"[KEY_FRAME] img["<<std::addressof(this->img)<<"] size: "<<this->img.size()<<std::endl;
    std::cout<<"[KEY_FRAME] img_data["<<std::addressof(this->img_data)<<"] size: "<<this->img_data.size()<<std::endl;
    std::cout<<"[KEY_FRAME] grad_frame["<<std::addressof(this->grad_frame)<<"] size: "<<this->grad_frame.size()<<std::endl;
    std::cout<<"[KEY_FRAME] selected points: "<<this->percent_points<<"[%]"<<std::endl;
}

void KeyFrame::create(const uint64_t &idx, const ::base::Time &time, cv::Mat &img,
            const std::vector<cv::Point2d> &coord, ::eds::mapping::IDepthMap2d &depthmap,
            const ::base::Affine3d &T, const cv::Size &out_size)
{
    /** Clean before creating a new Keyframe image**/
    this->clear();

    this->idx = idx; this->time = time; this->percent_points = percent_points; this->T_w_kf = T;
    std::cout<<"[KEY_FRAME] IDX: "<<idx<<" TIME: "<<time.toMicroseconds() <<std::endl;
    std::cout<<"[KEY_FRAME] img.type: "<<eds::utils::type2str(img.type()) <<std::endl;

    /** IMPORTANT: Image is already undistorted **/
    this->img = img.clone();
    cv::Size size = img.size();

    /** Check if the input image should be rescale **/
    if (this->out_scale[0] != 1 || this->out_scale[1] != 1)
    {
        cv::resize(this->img, this->img, out_size, cv::INTER_CUBIC);
        size = this->img.size();
    }

    /** Store the color image in uint8_t format **/
    this->uint8_img = this->img.clone();

    /** Convert the grayscale in case of color and normalize the image **/
    if (this->img.channels() > 1) cv::cvtColor(this->img, this->img, cv::COLOR_RGB2GRAY);
    this->img.convertTo(this->img, CV_64FC1);
    double min, max; cv::minMaxLoc(this->img, &min, &max);
    this->img = (this->img - min) / (max - min);

    /** Img in std vector **/
    this->img_data.resize(this->img.rows * this->img.cols);
    std::copy(this->img.begin<double>(), this->img.end<double>(), this->img_data.begin());

    /** Compute log image for the number of pyramids levels **/
    cv::Mat log_image;
    cv::log(this->img + KeyFrame::log_eps, log_image);
    this->log_img.push_back(log_image);

    /***********************/
    /** ONE PYRAMID LEVEL **/
    {
        /** Compute the gradient from the log normalized (0-1) image **/
        uint8_t i = 0;
        cv::Mat grad_x, grad_y, grad_xy;
        std::vector<cv::Mat> grad_channels;
        cv::Sobel(this->log_img[i], grad_x, CV_64FC1, 1, 0, 3); // derivative along x-axis
        cv::Sobel(this->log_img[i], grad_y, CV_64FC1, 0, 1, 3); // derivative along y-axis
        grad_channels.push_back(grad_x); grad_channels.push_back(grad_y);
        cv::merge(grad_channels, grad_xy);
        this->img_grad.push_back(grad_xy);

        /** Gradient frame in std vector [\Nabla_x, \Nabla_y]**/
        this->grad_frame.clear();
        for (int row=0; row<grad_xy.rows; ++row)
        {
            for (int col=0; col<grad_xy.cols; ++col)
            {
                this->grad_frame.push_back(grad_x.at<double>(row,col));
                this->grad_frame.push_back(grad_y.at<double>(row,col));
            }
        }

        /** Magnitude of the gradient (gradient norm). Magnitude is in the first element of cartToPolar **/
        cv::Mat mag_img, ang_img;
        cv::cartToPolar(grad_x, grad_y, mag_img, ang_img, true);
        this->mag.push_back(mag_img);

        /** Get the candidate points coordinates from the argument. No need of points selection **/
        this->coord = coord;

        /** Candidate points in normalized coordinates **/
        double fx, fy, cx, cy;
        fx = this->K_ref.at<double>(0,0); fy = this->K_ref.at<double>(1,1);
        cx = this->K_ref.at<double>(0,2); cy = this->K_ref.at<double>(1,2);
        for (auto it=this->coord.begin(); it!=this->coord.end(); ++it)
        {
            cv::Point2d c;
            c.x = (it->x - cx)/fx; //x-coordinate
            c.y = (it->y - cy)/fy; //y-coordinate
            this->norm_coord.push_back(c);
        }

        /** Gradient of the candidate points **/
        this->grad.clear();
        for (auto it=this->coord.begin(); it!=this->coord.end(); ++it)
        {
            this->grad.push_back(cv::Point2d(grad_x.at<double>(*it), grad_y.at<double>(*it)));
        }

        /** Create the inverse depth map **/
        double min_depth = * std::min_element(std::begin(depthmap.idepth), std::end(depthmap.idepth));
        double max_depth = * std::max_element(std::begin(depthmap.idepth), std::end(depthmap.idepth));
        this->setDepthMap(depthmap, min_depth, max_depth, 100, {1.0, 1.0});

        /** Create the point patches. Adaptive patch size defines as 7 for 240x180 **/
        eds::utils::splitImageInPatches(this->img, this->coord, this->patches, this->adaptive_patch_factor * (size.width * size.height));

        /** Create the point bundle patches **/
        eds::utils::computeBundlePatches(this->patches, this->bundle_patches);

        /** Initial residuals **/
        this->residuals.resize(this->coord.size(), 0.0);

        /** Initial tracks **/
        this->tracks.resize(this->coord.size(), Eigen::Vector2d::Zero());

        /** Initial Optical flow **/
        this->flow.resize(this->coord.size(), Eigen::Vector2d::Zero());
    }

    /** Update the percentage of the actual tracking points **/
    this->percent_points = (this->coord.size() * 100.0)/(size.height*size.width);

    std::cout<<"[KEY_FRAME] norm_coord["<<std::addressof(this->norm_coord)<<"] size: "<<this->norm_coord.size()<<std::endl;
    std::cout<<"[KEY_FRAME] grad["<<std::addressof(this->grad)<<"] size: "<<this->grad.size() <<std::endl;
    std::cout<<"[KEY_FRAME] inv_depth["<<std::addressof(this->inv_depth)<<"] size: "<<this->inv_depth.size()<<std::endl;
    std::cout<<"[KEY_FRAME] img["<<std::addressof(this->img)<<"] size: "<<this->img.size()<<std::endl;
    std::cout<<"[KEY_FRAME] img_data["<<std::addressof(this->img_data)<<"] size: "<<this->img_data.size()<<std::endl;
    std::cout<<"[KEY_FRAME] grad_frame["<<std::addressof(this->grad_frame)<<"] size: "<<this->grad_frame.size()<<std::endl;
    std::cout<<"[KEY_FRAME] selected points: "<<this->percent_points<<"[%]"<<std::endl;
}

void KeyFrame::create(const uint64_t &idx, const ::base::Time &time, cv::Mat &img,
            ::eds::mapping::IDepthMap2d &depthmap,
            const ::base::Affine3d &T, const cv::Size &out_size)
{
    /** Clean before creating a new Keyframe image**/
    this->clear();

    this->idx = idx; this->time = time; this->percent_points = percent_points; this->T_w_kf = T;
    std::cout<<"[KEY_FRAME] IDX: "<<idx<<" TIME: "<<time.toMicroseconds() <<std::endl;
    std::cout<<"[KEY_FRAME] given depthmap with: "<<depthmap.size() <<" points"<<std::endl;
    std::cout<<"[KEY_FRAME] img.type: "<<eds::utils::type2str(img.type()) <<std::endl;

    /** IMPORTANT: Image is already undistorted but at the original size **/
    this->img = img.clone();
    cv::Size size = img.size();

    /** Check if the input image should be rescale **/
    if (this->out_scale[0] != 1 || this->out_scale[1] != 1)
    {
        cv::resize(this->img, this->img, out_size, cv::INTER_CUBIC);
        size = this->img.size();
    }

    /** Store the color image in uint8_t format **/
    this->uint8_img = this->img.clone();

    /** Convert the grayscale in case of color and normalize the image **/
    if (this->img.channels() > 1) cv::cvtColor(this->img, this->img, cv::COLOR_RGB2GRAY);
    this->img.convertTo(this->img, CV_64FC1);
    double min, max; cv::minMaxLoc(this->img, &min, &max);
    this->img = (this->img - min) / (max - min);

    /** Img in std vector **/
    this->img_data.resize(this->img.rows * this->img.cols);
    std::copy(this->img.begin<double>(), this->img.end<double>(), this->img_data.begin());

    /** Compute log image for the number of pyramids levels **/
    cv::Mat log_image;
    cv::log(this->img + KeyFrame::log_eps, log_image);
    this->log_img.push_back(log_image);

    /***********************/
    /** ONE PYRAMID LEVEL **/
    {
        /** Compute the gradient from the log normalized (0-1) image **/
        uint8_t i = 0;
        cv::Mat grad_x, grad_y, grad_xy;
        std::vector<cv::Mat> grad_channels;
        cv::Sobel(this->log_img[i], grad_x, CV_64FC1, 1, 0, 3); // derivative along x-axis
        cv::Sobel(this->log_img[i], grad_y, CV_64FC1, 0, 1, 3); // derivative along y-axis
        grad_channels.push_back(grad_x); grad_channels.push_back(grad_y);
        cv::merge(grad_channels, grad_xy);
        this->img_grad.push_back(grad_xy);

        /** Gradient frame in std vector [\Nabla_x, \Nabla_y]**/
        this->grad_frame.clear();
        for (int row=0; row<grad_xy.rows; ++row)
        {
            for (int col=0; col<grad_xy.cols; ++col)
            {
                this->grad_frame.push_back(grad_x.at<double>(row,col));
                this->grad_frame.push_back(grad_y.at<double>(row,col));
            }
        }

        /** Magnitude of the gradient (gradient norm). Magnitude is in the first element of cartToPolar **/
        cv::Mat mag_img, ang_img;
        cv::cartToPolar(grad_x, grad_y, mag_img, ang_img, true);
        this->mag.push_back(mag_img);

        /** Get the candidate points coordinates from the depthmap argument. No need of points selection **/
        for (auto &it : depthmap.coord)
        {
            this->coord.push_back(it);
        }

        /** Candidate points in normalized coordinates **/
        double fx, fy, cx, cy;
        fx = this->K_ref.at<double>(0,0); fy = this->K_ref.at<double>(1,1);
        cx = this->K_ref.at<double>(0,2); cy = this->K_ref.at<double>(1,2);
        for (auto it=this->coord.begin(); it!=this->coord.end(); ++it)
        {
            cv::Point2d c;
            c.x = (it->x - cx)/fx; //x-coordinate
            c.y = (it->y - cy)/fy; //y-coordinate
            this->norm_coord.push_back(c);
        }

        /** Gradient of the candidate points **/
        this->grad.clear();
        for (auto it=this->coord.begin(); it!=this->coord.end(); ++it)
        {
            this->grad.push_back(cv::Point2d(grad_x.at<double>(*it), grad_y.at<double>(*it)));
        }

        /** Create the inverse depth map with equal weights **/
        double min_depth = * std::min_element(std::begin(depthmap.idepth), std::end(depthmap.idepth));
        double max_depth = * std::max_element(std::begin(depthmap.idepth), std::end(depthmap.idepth));
        this->inv_depth.init(this->K_ref, depthmap.idepth, min_depth, max_depth);
        this->weights.resize(this->coord.size());
        std::fill(this->weights.begin(), this->weights.end(), 1.0);


        /** Create the point patches. Adaptive patch size defines as 7 **/
        eds::utils::splitImageInPatches(this->img, this->coord, this->patches);

        /** Create the point bundle patches **/
        eds::utils::computeBundlePatches(this->patches, this->bundle_patches);

        /** Initial residuals **/
        this->residuals.resize(this->coord.size(), 0.0);

        /** Initial tracks **/
        this->tracks.resize(this->coord.size(), Eigen::Vector2d::Zero());

        /** Initial Optical flow **/
        this->flow.resize(this->coord.size(), Eigen::Vector2d::Zero());
    }

    /** Update the percentage of the actual tracking points **/
    this->percent_points = (this->coord.size() * 100.0)/(size.height*size.width);

    std::cout<<"[KEY_FRAME] norm_coord["<<std::addressof(this->norm_coord)<<"] size: "<<this->norm_coord.size()<<std::endl;
    std::cout<<"[KEY_FRAME] grad["<<std::addressof(this->grad)<<"] size: "<<this->grad.size() <<std::endl;
    std::cout<<"[KEY_FRAME] inv_depth["<<std::addressof(this->inv_depth)<<"] size: "<<this->inv_depth.size()<<std::endl;
    std::cout<<"[KEY_FRAME] img["<<std::addressof(this->img)<<"] size: "<<this->img.size()<<std::endl;
    std::cout<<"[KEY_FRAME] img_data["<<std::addressof(this->img_data)<<"] size: "<<this->img_data.size()<<std::endl;
    std::cout<<"[KEY_FRAME] grad_frame["<<std::addressof(this->grad_frame)<<"] size: "<<this->grad_frame.size()<<std::endl;
    std::cout<<"[KEY_FRAME] selected points: "<<this->percent_points<<"[%]"<<std::endl;
}

void KeyFrame::clear()
{
    this->coord.clear();
    this->norm_coord.clear();
    this->grad.clear();
    this->patches.clear();
    this->bundle_patches.clear();
    this->residuals.clear();
    this->weights.clear();
    this->tracks.clear();
    this->flow.clear();

    /** Images **/
    this->grad_frame.clear();
    this->log_img.clear();
    this->img_grad.clear();
    this->mag.clear();
    this->img_data.clear();
}

void KeyFrame::candidatePoints(std::vector<cv::Point2d> &coord, const cv::Size &patch_size,
                                CANDIDATE_POINT_METHOD method, const int &num_points, uint8_t level)
{
    cv::Mat mag_image = this->mag[level];
    cv::Size mag_size = mag_image.size();
    //float scale = std::pow(2.0, level);
    std::cout<<"[KEY_FRAME] mag image: "<<mag_image.size()<<std::endl;
    assert(mag_image.type() == CV_64FC1);

    /** Break the images in patches **/
    std::vector<cv::Mat> patches;
    std::vector<cv::Point> start_points;
    for (int y=0; y<=mag_size.height-patch_size.height; y+=patch_size.height)
    {
        for (int x=0; x<=mag_size.width-patch_size.width; x+=patch_size.width)
        {
            cv::Rect grid_rect(x, y, patch_size.width, patch_size.height);
            //std::cout<<grid_rect <<" at "<<cv::Point(x+patch_size.width,y+patch_size.height)<<std::endl;
            patches.push_back(mag_image(grid_rect));
            start_points.push_back(cv::Point(x, y));
            //coord.push_back(cv::Point2d(x, y));
        }
    }
    std::cout<<"[KEY_FRAME] number of patches: "<<patches.size()<<std::endl;

    switch(method)
    {
        case MAX:
        {
            int elem_per_patch = num_points/patches.size();
            std::cout<<"[KEY_FRAME] METHOD MAX: "<<elem_per_patch<<" point per patch"<<std::endl;

            /** Get max elem for each patch **/
            int k = 0;
            for(std::vector<cv::Mat>::iterator patch=patches.begin();
                patch!=patches.end(); ++patch)
            {
                //std::cout<<"patch size: "<<patch->size()<<eds::utils::type2str(patch->type())<<std::endl;
                cv::Point &s_point = start_points[k];
                for (int i=0; i<elem_per_patch; ++i)
                {
                    double min, max;
                    cv::Point min_loc, max_loc;
                    cv::minMaxLoc(*patch, &min, &max, &min_loc, &max_loc);
                    if (max == min) break;
                    //std::cout<<"max mag: "<<max<<std::endl;
                    coord.push_back(cv::Point2d(max_loc.x + s_point.x, max_loc.y + s_point.y));
                    (*patch).at<double>(max_loc) = 0.0;
                }
                k++;
            }
            break;
        }
        case MEDIAN:
        {
            std::cout<<"[KEY_FRAME] METHOD MEDIAN "<<std::endl;
            int k = 0;
            for(std::vector<cv::Mat>::iterator patch=patches.begin();
                patch!=patches.end(); ++patch)
            {
                cv::Point &s_point = start_points[k];
                double median = eds::utils::medianMat(*patch);
                //cv::Scalar mean = cv::mean(*patch);
                //std::cout<<"median: "<<median<<std::endl;
                //std::cout<<"mean: "<<mean.val[0]<<std::endl;
                for (uint16_t y=0; y<patch->rows; ++y)
                {
                    for(uint16_t x=0; x<patch->cols; ++x)
                    {
                        if (patch->at<double>(y, x) > median)
                            coord.push_back(cv::Point2d(x + s_point.x, y + s_point.y));
                    }
                }
                k++;
            }
            break;
        }
        default:
            throw std::runtime_error("error");
    }
    this->num_points = coord.size();
    std::cout<<"[KEY_FRAME] Desire num_points["<<num_points<<"] got ["<<this->num_points<<"]"<<std::endl;

}

bool KeyFrame::initialStructure(const cv::Mat &input, Eigen::Matrix3d &Rotation, Eigen::Vector3d &translation,
                                std::vector<cv::Vec3d> &lines, const float &ratio_thresh)
{
    /** Get the KF image in CV_8UC1 **/
    cv::Mat img_kf; this->img.convertTo(img_kf, CV_8UC1, 255, 0);

    /** Step 0: Resize and Undistort coming frame **/
    cv::Mat input_resize, img_undistort;
    cv::Size out_size = img_kf.size();
    cv::resize(input, input_resize, out_size, cv::INTER_CUBIC);
    if (this->distortion_model.compare("equidistant") != 0)
    {
        /** radtan model **/
        cv::undistort(input_resize, img_undistort, this->K, this->D, this->K_ref);
    }
    else
    {
        /** Kalibr equidistant model is opencv fisheye **/
        cv::fisheye::undistortImage(input_resize, img_undistort, this->K, this->D, this->K_ref);
    }

    /** Check the image is in grayscale **/
    if (img_undistort.channels() > 1)
    {
        cv::cvtColor(img_undistort, img_undistort, cv::COLOR_RGB2GRAY);
    }

    /**  Normalize the input since the kf image is also normalized **/
    img_undistort.convertTo(img_undistort, CV_64FC1);
    double min, max; cv::minMaxLoc(img_undistort, &min, &max);
    img_undistort = (img_undistort - min) / (max - min);
    cv::Mat img_in; img_undistort.convertTo(img_in, CV_8UC1, 255, 0);

    /** Step 1: Detect the keypoints using SURF Detector, compute the descriptors **/
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(10000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute(img_kf, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img_in, cv::noArray(), keypoints2, descriptors2);

    std::cout<<"[INIT] KP1 detected: "<<keypoints1.size()<<std::endl;
    std::cout<<"[INIT] KP2 detected: "<<keypoints2.size()<<std::endl;
    if (keypoints1.size() == 0 || keypoints2.size() == 0)
        return false;

    /** Step 2: Matching descriptor vectors with a FLANN based matcher **/
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    if (knn_matches.size() == 0)
        return 0;

    /**  Filter matches using the Lowe's ratio test **/
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2d> pts1, pts2;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            pts1.push_back(keypoints1[ knn_matches[i][0].queryIdx].pt);
            pts2.push_back(keypoints2[ knn_matches[i][0].trainIdx].pt);
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    /** Compute epilines **/
    lines.clear();
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_LMEDS);
    cv::computeCorrespondEpilines(pts1, 1, F, lines);


    /** Find the Essential matrix **/
    cv::Mat rot, trans, mask;
    std::vector<cv::Point2d> ref_pts1, ref_pts2;
    cv::correctMatches(F, pts1, pts2, ref_pts1, ref_pts2);
    cv::Mat E = cv::findEssentialMat(ref_pts1, ref_pts2, this->K_ref, cv::RANSAC, 0.99, 1.0, mask);
    int inlier_cnt = cv::recoverPose(E, ref_pts1, ref_pts2, this->K_ref, rot, trans, mask);
    std::cout << "[INIT] inlier_cnt " << inlier_cnt << std::endl;

   std::cout<<"[INIT] kp1.size:"<<keypoints1.size()<<" kp2.size:" <<keypoints2.size()<<" des1.size:"<<descriptors1.size()
   <<" des2.size:"<<descriptors2.size()<<" knn_matches.size:"<<knn_matches.size()<<" good_matches:"<<good_matches.size()
   <<" pts1.size:"<<pts1.size()<<" pts2.size: "<<pts2.size()<<" ref_pts1.size:"<<ref_pts1.size()<<" ref_pts2.size: "
   <<ref_pts2.size()<<std::endl;


   // /** Draw Epilines corresponding to points in KF and draw in the input img **/
   cv::Mat img_epi = ::eds::utils::epilinesViz(img_in, lines, 1);

   cv::Mat img_matches;
   cv::drawMatches(img_kf, keypoints1, img_in, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
   cv::imwrite("/tmp/img_matches_"+std::to_string(idx)+".png", img_matches);
   cv::imwrite("/tmp/img_epilines_"+std::to_string(idx)+".png", img_epi);

    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    for (int i = 0; i < 3; i++)
    {   
        T(i) = trans.at<double>(i, 0);
        for (int j = 0; j < 3; j++)
            R(i, j) = rot.at<double>(i, j);
    }

    Rotation = R.transpose();
    translation = -R.transpose() * T;
    if(inlier_cnt > 12)
        return true;
    else
        return false;
}

void KeyFrame::trackPoints(const cv::Mat &img, std::vector<cv::Point2d> &track_coord, const uint8_t &patch_size)
{
    /** Perform KLT tracker **/
    std::vector<uchar> status;
    std::vector<float> error;
    std::vector<cv::Point2f> coord, tr_coord;
    for (auto &it: this->coord)
        coord.push_back(cv::Point2f(it.x, it.y));

    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
    cv::calcOpticalFlowPyrLK(this->uint8_img, img, coord, tr_coord, status, error, cv::Size(patch_size, patch_size), 4, criteria);

    track_coord.clear();
    int idx = 0;
    auto it_c = this->coord.begin();
    auto it_track = this->tracks.begin();
    auto it_tr = tr_coord.begin();
    auto it_s = status.begin();
    auto it_e = error.begin();
    for (; it_c != this->coord.end();)
    {
        //std::cout<<"it_c:  "<<*it_c<<" it_tr: "<<*it_tr<<" error: "<<*it_e<<std::endl;
        if(*it_s == 0)
        {
            auto its = this->erasePoint(idx);
            it_c = its.coord; it_track = its.tracks;
            it_tr = tr_coord.erase(tr_coord.begin()+idx);
            it_s = status.erase(status.begin()+idx);
            it_e = error.erase(error.begin()+idx);
        }
        else
        {
            track_coord.push_back(cv::Point2d(it_tr->x, it_tr->y));
            Eigen::Vector2d f = Eigen::Vector2d((it_tr->x)-(it_c->x), (it_tr->y)-(it_c->y));
            (*it_track) += f;
            ++idx; ++it_c; ++it_track; ++it_tr; ++it_s; ++it_e;
        }
    }
    /** Store current image for the next iteration **/
    this->uint8_img = img;
    std::cout<<"[KEY_FRAME] KLT coord.size(): "<<this->coord.size()<<" track_coord.size(): "<<track_coord.size()<<std::endl;
}

void KeyFrame::trackPoints(const cv::Mat &img, const int &border_type, const uint8_t &border_value)
{
    std::vector<cv::Point2d> track_coord;
    int idx = 0;
    for (auto &it : this->tracks)
    {
        cv::Point2d c = this->coord[idx] + cv::Point2d(it[0], it[1]);
        track_coord.push_back(c);
        ++idx;
    }
    uint16_t patch_radius = this->patches[0].cols * 2;
    std::vector<cv::Mat> img_patches;
    eds::utils::splitImageInPatches(img, track_coord, img_patches, patch_radius, border_type, border_value);
    idx = 0;
    auto it_c =  this->coord.begin();
    auto it_p = this->patches.begin();
    auto it_tr = track_coord.begin();
    auto it_img = img_patches.begin();
    for (; it_p != this->patches.end(); ++idx, ++it_c, ++it_p, ++it_tr, ++it_img)
    {
        cv::Mat img; (*it_img).convertTo(img, CV_32FC1);
        cv::Point2d center ((img.cols - (*it_p).cols + 1)/2.0, (img.rows - (*it_p).rows + 1)/2.0);
        cv::Point2d p_ssd = *it_tr + (eds::utils::matchTemplate(eds::utils::viz(img-cv::mean(img)), eds::utils::viz(*it_p-cv::mean(*it_p)), cv::TM_SQDIFF_NORMED) - center);
        cv::Point2d p_ncc = *it_tr + (eds::utils::matchTemplate(eds::utils::viz(img-cv::mean(img)), eds::utils::viz(*it_p-cv::mean(*it_p)), cv::TM_CCORR_NORMED) - center);
        cv::Point2d flow = eds::utils::matchTemplate(eds::utils::viz(img-cv::mean(img)), eds::utils::viz(*it_p-cv::mean(*it_p)), cv::TM_SQDIFF_NORMED) - center;
        if (cv::norm(p_ssd-p_ncc) < 2)
        {
            std::cout<<"** [TEST] coord["<<idx<<"]: "<<*it_c<<" p_ssd: "<<p_ssd<<" p_ncc: "<<p_ncc
            <<" diff: "<<std::fabs(cv::norm(p_ssd)-cv::norm(p_ncc)) <<" track_coord: "<<*it_tr<<" flow: "<<flow<<std::endl;
            this->tracks[idx] += Eigen::Vector2d(flow.x, flow.y);
        }
    }

    //cv::Mat match_img;
    //cv::copyMakeBorder(img, match_img, 7, 7, 7, 7, border_type, border_value);
    //match_img.convertTo(match_img, CV_32FC1);
    //int idx = 1000;
    //auto it_c =  this->coord.begin()+idx;
    //auto it_p = this->patches.begin()+idx;
    ////auto it_tr = tracker_coord.begin();
    ////for (; it_p != this->patches.end(); ++idx, ++it_c, ++it_p)
    //{
    //    cv::Point2d p_ssd = eds::utils::matchTemplate(eds::utils::viz(match_img-cv::mean(match_img)), eds::utils::viz(*it_p-cv::mean(*it_p)), cv::TM_SQDIFF_NORMED);
    //    cv::Point2d p_ncc = eds::utils::matchTemplate(eds::utils::viz(match_img-cv::mean(match_img)), eds::utils::viz(*it_p-cv::mean(*it_p)), cv::TM_CCORR_NORMED);
    //    std::cout<<"** [TEST] coord["<<idx<<"]: "<<*it_c<<" p_ssd: "<<p_ssd<<" p_ncc: "<<p_ncc
    //    <<" diff: "<<std::fabs(cv::norm(p_ssd)-cv::norm(p_ncc)) <<std::endl;
    //    this->tracks[idx] = Eigen::Vector2d(p_ssd.x-(*it_c).x, p_ssd.y-(*it_c).y);
    //    cv::imwrite("/tmp/patch_"+std::to_string(idx)+".png", eds::utils::viz(*it_p));
    //}
}

void KeyFrame::pointsRefinement(const cv::Mat &event_frame, const double &event_diff, const uint16_t &patch_radius,
                        const int &border_type, const uint8_t &border_value)
{
    /** Split the image in patches **/
    std::vector<cv::Mat> event_patches;
    eds::utils::splitImageInPatches(event_frame, this->coord, event_patches, patch_radius, border_type, border_value);

    int idx = 0;
    int num_removed_points = 0;
    auto it_p = event_patches.begin();
    for (; it_p != event_patches.end();)
    {
        double min, max;
        cv::minMaxLoc(*it_p, &min, &max);
        if (std::fabs(max-min) < event_diff) //less than event_diff
        {
            this->erasePoint(idx);
            it_p = event_patches.erase(event_patches.begin()+idx);
            num_removed_points++;
        }
        else
        {
            ++it_p; ++idx;
        }
    }
    this->num_points = coord.size();
    std::cout<<"[KEY_FRAME] KF["<<this->idx<<"] Points Refinement. New Number points: "<<this->num_points<<" removed["<<num_removed_points<<"]"<<std::endl;
}

eds::tracking::KFPointIterators KeyFrame::erasePoint(const int &idx)
{
    /** It is very important to keep consistency size **/
    assert(this->coord.size() == this->norm_coord.size());
    assert(this->norm_coord.size() == this->grad.size());
    assert(this->grad.size() == this->patches.size());
    assert(this->patches.size() == this->bundle_patches.size());
    assert(this->bundle_patches.size() == this->residuals.size());
    assert(this->residuals.size() == this->weights.size());
    assert(this->weights.size() == this->tracks.size());
    assert(this->tracks.size() == this->flow.size());
    assert(this->flow.size() == this->inv_depth.size());

    eds::tracking::KFPointIterators its;

    /** delete pixel coord **/
    its.coord = this->coord.erase(this->coord.begin()+idx);

    /** delete normalized coord **/
    its.norm_coord = this->norm_coord.erase(this->norm_coord.begin()+idx);

    /** delere gradient **/
    its.grad = this->grad.erase(this->grad.begin()+idx);

    /** delete patches **/
    its.patches = this->patches.erase(this->patches.begin()+idx);

    /** delete bundle patches **/
    its.bundle_patches = this->bundle_patches.erase(this->bundle_patches.begin()+idx);

    /** delete the residual **/
    its.residuals = this->residuals.erase(this->residuals.begin()+idx);

    /** delete the weights **/
    its.weights = this->weights.erase(this->weights.begin()+idx);

    /** delete the tracks **/
    its.tracks = this->tracks.erase(this->tracks.begin()+idx);

    /** delete the flow **/
    its.flow = this->flow.erase(this->flow.begin()+idx);

    /** delete the inverse depth **/
    its.inv_depth = this->inv_depth.erase(idx);

    return its;
}

cv::Mat KeyFrame::viz(const cv::Mat &img, bool color)
{
    double min, max;
    cv::minMaxLoc(img, &min, &max);
    double bmax = 0.7 * max;
    cv::Mat norm_img = (img + bmax)/(2*bmax);//(img - min)/(max -min); //between 0 - 1

    cv::Mat img_viz;
    norm_img.convertTo(img_viz, CV_8UC1, 255, 0);
    if (color)
    {
        eds::utils::BlueWhiteRed bwr;
        bwr(img_viz, img_viz);
        //cv::applyColorMap(img_viz, img_viz, cv::COLORMAP_JET);
    }

    return img_viz;
}

void KeyFrame::setDepthMap(::eds::mapping::IDepthMap2d &depthmap, const ::eds::mapping::Config &map_info)
{
    this->setDepthMap(depthmap, map_info.min_depth, map_info.max_depth, map_info.convergence_sigma2_thresh, this->out_scale);
}

void KeyFrame::setDepthMap(::eds::mapping::IDepthMap2d &depthmap, const ::eds::mapping::Config &map_info, const std::array<double, 2> &scale)
{
    this->setDepthMap(depthmap, map_info.min_depth, map_info.max_depth, map_info.convergence_sigma2_thresh, scale);
}

void KeyFrame::setDepthMap(::eds::mapping::IDepthMap2d &depthmap, const double &min_depth, const double &max_depth,
                        const double &convergence_sigma2_thresh, const std::array<double, 2> &scale)
{
    /* Local inverse depth vector**/
    std::vector<double> idp;

    /** Local distance to closest point **/
    std::vector<double> distance;

    /** Get points depth from the projected map **/
    if (!depthmap.empty())
    {
        std::cout<<"[KEY_FRAME] Given depthmap"<<std::endl;

        /** Create the KDTree to search **/
        ::eds::mapping::KDTree<eds::mapping::Point2d> kdtree(depthmap.coord);
        for (auto point=this->coord.begin(); point!=this->coord.end(); ++point)
        {
            /** Create the query point **/
            const eds::mapping::Point2d query(point->x * scale[0], point->y * scale[1]);
            /** Index of the closest **/
            const int idx = kdtree.nnSearch(query);
            /** Push the inverse depth value **/
            idp.push_back(depthmap.idepth[idx]);
            /** Push distance norm to the closest point **/
            cv::Point2d dist(depthmap.coord[idx].x() - point->x, depthmap.coord[idx].y() - point->y);
            distance.push_back(cv::norm(dist));
        }
        std::cout<<"[KEY_FRAME] max idp: "<<*std::max_element(std::begin(idp), std::end(idp))<<std::endl;
        std::cout<<"[KEY_FRAME] min idp: "<<*std::min_element(std::begin(idp), std::end(idp))<<std::endl;

        /** Create the point weights **/
        this->weights.clear();
        double max = *std::max_element(std::begin(distance), std::end(distance));
        double min = *std::min_element(std::begin(distance), std::end(distance));
        if (min != max)
        {
            for (auto &it : distance)
                weights.push_back(1.0 - ((it-min)/(max-min)));
        }
        else
        {
            this->weights.resize(this->coord.size());
            std::fill(this->weights.begin(), this->weights.end(), 1.0);
        }

        std::cout<<"[KEY_FRAME] max weights: "<<*std::max_element(std::begin(this->weights), std::end(this->weights))<<std::endl;
        std::cout<<"[KEY_FRAME] min weights: "<<*std::min_element(std::begin(this->weights), std::end(this->weights))<<std::endl;
    }
    else
    {
        std::cout<<"[KEY_FRAME] empty depthmap with init_depth: "<<(max_depth-min_depth)/2.0<<std::endl;
        /** initialize the point with initial guess **/
        idp.resize(this->coord.size());
        std::fill(idp.begin(), idp.end(), 1.0/((max_depth-min_depth)/2.0));
        this->weights.resize(this->coord.size());
        std::fill(this->weights.begin(), this->weights.end(), 1.0);
    }

    /** Create the inverse depth **/
    this->inv_depth.init(this->K_ref, idp, min_depth, max_depth, convergence_sigma2_thresh);
}

void KeyFrame::setPose(const ::base::Transform3d& pose)
{
    this->T_w_kf = pose;
}

::base::Transform3d& KeyFrame::getPose()
{
    return this->T_w_kf;
}

::base::Matrix4d KeyFrame::getPoseMatrix()
{
    return this->T_w_kf.matrix();
}

std::pair<Eigen::Vector3d, Eigen::Quaterniond> KeyFrame::getTransQuater()
{
    return std::make_pair(this->T_w_kf.translation(), Eigen::Quaterniond(this->T_w_kf.rotation()));
}

std::vector<base::Point> KeyFrame::getDepthMap()
{
    std::vector<base::Point> pcl;
    assert(this->coord.size() == this->inv_depth.size());
    double fx, fy, cx, cy;
    fx = this->K_ref.at<double>(0,0); fy = this->K_ref.at<double>(1,1);
    cx = this->K_ref.at<double>(0,2); cy = this->K_ref.at<double>(1,2);

    auto it_p=this->coord.begin();
    auto it_i=this->inv_depth.begin();
    for(;it_p<this->coord.end(); ++it_p, ++it_i)
    {
        double d_i = 1.0/::eds::mapping::mu(*it_i);
        pcl.push_back(::base::Point(d_i*(((*it_p).x-cx)/fx), d_i*(((*it_p).y-cy)/fy), d_i));
    }

    return pcl;
}

base::samples::Pointcloud KeyFrame::getMap(const std::vector<double> &model, const MAP_COLOR_MODE &color_mode)
{
    return this->getMap(this->inv_depth.getIDepth(), model, color_mode);
}

base::samples::Pointcloud KeyFrame::getMap(const std::vector<double> &idp, const std::vector<double> &model, const MAP_COLOR_MODE &color_mode)
{
    base::samples::Pointcloud pcl;
    assert(this->norm_coord.size() == idp.size());

    /** Color in case there is a model and the color mode is EVENTS **/
    ::base::Vector4d color_positive(0.0, 0.0, 1.0, 1.0); //positive blue color
    ::base::Vector4d color_negative(1.0, 0.0, 0.0, 1.0); //negative red color

    pcl.time = this->time;
    int idx = 0;
    auto it_c=this->coord.begin();
    auto it_p=this->norm_coord.begin();
    auto it_i=idp.begin();
    for(;it_p<this->norm_coord.end(); ++it_c, ++it_p, ++it_i, ++idx)
    {
        double d_i = 1.0/(*it_i);
        /** Push points in the world frame coordinate **/
        pcl.points.push_back(this->T_w_kf * ::base::Point(d_i*(*it_p).x, d_i*(*it_p).y, d_i));
        switch (color_mode)
        {
        case MAP_COLOR_MODE::IMG:
        {
            double color = this->img.at<double>(*it_c);
            pcl.colors.push_back(::base::Vector4d(color, color, color, 1.0));//img intensity
            break;
        }
        case MAP_COLOR_MODE::EVENTS:
        {
            if (model.size() == idp.size())
            {
                if (model[idx] > 0)
                {
                    pcl.colors.push_back(color_positive);
                }
                else
                {
                    pcl.colors.push_back(color_negative);
                }
            }
            break;
        }
        case MAP_COLOR_MODE::RED:
        {
            pcl.colors.push_back(::base::Vector4d(1.0, 0.0, 0.0, 1.0));//red color
            break;
        }
        default:
        {
            pcl.colors.push_back(::base::Vector4d(1.0, 1.0, 1.0, 1.0));//black color
            break;
        }
        }
    }

    return pcl;
}

cv::Mat KeyFrame::getGradientMagnitude(const std::string &method, const float s)
{
    return this->getGradientMagnitude(this->coord, method, s);
}

cv::Mat KeyFrame::getGradientMagnitude(const std::vector<cv::Point2d> &coord, const std::string &method, const float s)
{
    std::vector<double> magnitude;
    for (auto it=this->grad.begin(); it<this->grad.end(); ++it)
    {
        magnitude.push_back(cv::norm(*it));
    }
    cv::Mat img = eds::utils::drawValuesPoints(coord, magnitude, this->img.rows, this->img.cols, method, s);

    return img;
}

cv::Mat KeyFrame::getGradient_x(const std::string &method, const float s)
{
    return this->getGradient_x(this->coord, method, s);
}

cv::Mat KeyFrame::getGradient_x(const std::vector<cv::Point2d> &coord, const std::string &method, const float s)
{
    std::vector<double> grad_x;
    for (auto it=this->grad.begin(); it<this->grad.end(); ++it)
    {
        grad_x.push_back(it->x);
    }
    cv::Mat img = eds::utils::drawValuesPoints(coord, grad_x, this->img.rows, this->img.cols, method, s);

    return img;
}

cv::Mat KeyFrame::getGradient_y(const std::string &method, const float s)
{
    return this->getGradient_y(this->coord, method, s);
}

cv::Mat KeyFrame::getGradient_y(const std::vector<cv::Point2d> &coord, const std::string &method, const float s)
{
    std::vector<double> grad_y;
    for (auto it=this->grad.begin(); it<this->grad.end(); ++it)
    {
        grad_y.push_back(it->y);
    }
    cv::Mat img = eds::utils::drawValuesPoints(coord, grad_y, this->img.rows, this->img.cols, method, s);

    return img;
}

std::vector<double> KeyFrame::getSparseModel(const Eigen::Vector3d &vx, const Eigen::Vector3d &wx)
{
    return this->getSparseModel(this->coord, vx, wx);
}

std::vector<double> KeyFrame::getSparseModel(const std::vector<cv::Point2d> &coord, const Eigen::Vector3d &vx, const Eigen::Vector3d &wx)
{
    assert(coord.size() == this->grad.size());
    assert(coord.size() == this->inv_depth.size());

    /** Camera intrinsics **/
    double fx, fy, cx, cy;
    fx = this->K_ref.at<double>(0,0); fy = this->K_ref.at<double>(1,1);
    cx = this->K_ref.at<double>(0,2); cy = this->K_ref.at<double>(1,2);

    /** Get flow **/
    std::vector<Eigen::Vector2d> flow;
    auto it_c = coord.begin();
    auto it_d = this->inv_depth.begin();
    for (; it_c != coord.end(); ++it_c, ++it_d)
    {
        /** Get normalized coord **/
        cv::Point2d c;
        c.x = (it_c->x - cx)/fx; //x-coordinate
        c.y = (it_c->y - cy)/fy; //y-coordinate
        /** Compute the flow using depth **/
        Eigen::Vector2d f;
        ::eds::utils::compute_flow(c.x, c.y, vx, wx, ::eds::mapping::mu(*it_d), f);
        flow.push_back(f);
    }

    /** Compute the model **/
    std::vector<double> model;
    auto it_flow = flow.begin();
    auto it_grad = this->grad.begin();
    double model_norm_sq = 1e-03;
    for (; it_grad!=this->grad.end(); ++it_grad, ++ it_flow)
    {
        cv::Point2d &g = *it_grad;
        Eigen::Vector2d &f = *it_flow;
        double value = -(g.x * f[0] + g.y * f[1]);
        model.push_back(value);
        model_norm_sq += value * value;
    }

    /** Normalized model **/
    double model_norm = std::sqrt(model_norm_sq);
    std::for_each(model.begin(), model.end(), [model_norm](double &v){ v /= model_norm; });

    return model;
}

cv::Mat KeyFrame::getModel(const Eigen::Vector3d &vx, const Eigen::Vector3d &wx, const std::string &method, const float &s)
{
    return this->getModel(this->coord, vx, wx, method, s);

}

cv::Mat KeyFrame::getModel(const std::vector<cv::Point2d> &coord, const Eigen::Vector3d &vx, const Eigen::Vector3d &wx,
                        const std::string &method, const float &s)
{
    assert(coord.size() == this->grad.size());
    assert(coord.size() == this->inv_depth.size());

    std::vector<double> model = this->getSparseModel(coord, vx, wx);

    /** The model image **/
    cv::Mat img = eds::utils::drawValuesPoints(coord, model, this->img.rows, this->img.cols, method, s);

    return img;
}

cv::Mat KeyFrame::idepthmapViz(const std::string &method, const float s)
{
   return this->idepthmapViz(this->coord, this->inv_depth.getIDepth(), method, s);
}

cv::Mat KeyFrame::idepthmapViz(const std::vector<cv::Point2d> &coord, const std::vector<double> &idp, const std::string &method, const float s)
{
    cv::Mat idepth_img = cv::Mat(this->img.rows, this->img.cols, CV_64FC1, cv::Scalar(0));
    auto it_p = coord.begin();
    auto it_i = idp.begin();
    for (; it_i != idp.end(); ++it_p, ++it_i)
    {
        idepth_img.at<double>(*it_p) = *it_i;
    }

    double min, max;
    cv::minMaxLoc(idepth_img, &min, &max);
    idepth_img = (idepth_img - min)/(max - min);
    cv::Mat idepth_color; idepth_img.convertTo(idepth_color, CV_8UC1, 255, 0);
    cv::applyColorMap(idepth_color, idepth_color, cv::COLORMAP_JET);

    cv::Mat img_color; this->img.convertTo(img_color, CV_8UC1, 255, 0);
    cv::cvtColor(img_color, img_color, cv::COLOR_GRAY2RGB);
    for (auto it=coord.begin(); it!=coord.end(); ++it)
    {
        cv::Vec3b & point = img_color.at<cv::Vec3b>(*it);
        point = idepth_color.at<cv::Vec3b>(*it);
    }

    return img_color;
}

cv::Mat KeyFrame::weightsViz(double &min, double &max)
{
   return this->weightsViz(this->coord, this->weights, min, max);
}

cv::Mat KeyFrame::weightsViz(const std::vector<cv::Point2d> &coord, const std::vector<double> &weights, double &min, double &max)
{
    cv::Mat weights_img = cv::Mat(this->img.rows, this->img.cols, CV_64FC1, cv::Scalar(0));
    auto it_p = coord.begin();
    auto it_w = weights.begin();
    for (; it_w != weights.end(); ++it_p, ++it_w)
    {
        weights_img.at<double>(*it_p) = *it_w;
    }

    cv::minMaxLoc(weights_img, &min, &max);
    weights_img = (weights_img - min)/(max - min);
    cv::Mat img_color; weights_img.convertTo(img_color, CV_8UC1, 255, 0);
    cv::applyColorMap(img_color, img_color, cv::COLORMAP_RAINBOW);

    return img_color;
}

cv::Mat KeyFrame::residualsViz()
{
    /** Draw the residuals **/
    cv::Mat img = cv::Mat(this->img.rows, this->img.cols, CV_64FC1, cv::Scalar(0));
    auto it_p = this->coord.begin();
    auto it_r = this->residuals.begin();
    for (; it_r != this->residuals.end(); ++it_p, ++it_r)
    {
        img.at<double>(*it_p) = *it_r;
    }

    double min, max;
    cv::minMaxLoc(img, &min, &max);
    cv::Mat norm_img = (img - min)/(max -min);

    cv::Mat residuals_viz;
    norm_img.convertTo(residuals_viz, CV_8UC1, 255, 0);
    cv::applyColorMap(residuals_viz, residuals_viz, cv::COLORMAP_JET);

    cv::Mat img_color; this->img.convertTo(img_color, CV_8UC1, 255, 0);
    cv::cvtColor(img_color, img_color, cv::COLOR_GRAY2RGB);
    for (auto it=this->coord.begin(); it!=this->coord.end(); ++it)
    {
        cv::Vec3b & point = img_color.at<cv::Vec3b>(*it);
        point = residuals_viz.at<cv::Vec3b>(*it);
    }

    return img_color;
}

cv::Mat KeyFrame::eventsOnKeyFrameViz(const cv::Mat &event_frame)
{
    /** Event colors **/
    cv::Vec3b color_positive, color_negative;
    color_positive = cv::Vec3b(255.0, 0.0, 0.0); //BGR (BLUE COLOR)
    color_negative = cv::Vec3b(0.0, 0.0, 255.0); //BGR (RED COLOR)

    /** Convert the frame KF to a color image to draw the events in color **/
    cv::Mat img_color; this->img.convertTo(img_color, CV_8UC1, 255, 0);
    cv::cvtColor(img_color, img_color, cv::COLOR_GRAY2RGB);

    /** Draw the events **/
    for (int x=0; x<img_color.cols; ++x)
    {
        for(int y=0; y<img_color.rows; ++y)
        {
            double p = event_frame.at<double>(y, x);
            if (p>0.0)
            {
                img_color.at<cv::Vec3b>(y, x) = color_positive;
            }
            else if (p<-0.0)
            {
                img_color.at<cv::Vec3b>(y, x) = color_negative;
            }
        }
    }

    return img_color;
}

void KeyFrame::meanResiduals(double &mean, double &st_dev)
{
    ::eds::utils::mean_std_vector(this->residuals, mean, st_dev);
}

void KeyFrame::medianResiduals(double &median, double &third_q)
{
    median = eds::utils::n_quantile_vector(this->residuals, this->residuals.size()/2);
    third_q = eds::utils::n_quantile_vector(this->residuals, this->residuals.size()/3);
}

bool KeyFrame::needNewKF(const double &percent_thr)
{
    /** initial num points - current num points > (percent_thr * initial points) **/
    std::cout<<"[KEY_FRAME] Need KF: "<<(this->num_points - this->coord.size())<<" > "<<(percent_thr * this->num_points)<<"?"<<std::endl;
    return (this->num_points - this->coord.size()) > (percent_thr * this->num_points);
}

bool KeyFrame::needNewKFImageCriteria(const double &percent)
{
    /** If the current number of points are below a percent of the image plane **/
    std::cout<<"[KEY_FRAME] PERCENT["<<percent*100.0<<"] Need KF IMG CRITERIA: "<<(this->coord.size())<<" < "<<(this->img.cols * this->img.rows) * percent<<"?"<<std::endl;
    return this->coord.size() < ((this->img.cols * this->img.rows) * percent);
}

void KeyFrame::cleanPoints(const double &w_norm_thr)
{
    int idx=0;
    int num_removed_points=0;
    auto it_w = this->weights.begin();

    /** Weights are between 0 - 1 meaning 1: good 0: bad **/
    for (; it_w != this->weights.end();)
    {
        if (*it_w < w_norm_thr)
        {
            auto its = this->erasePoint(idx);
            it_w = its.weights;
            num_removed_points++;
        }
        else
        {
            it_w++; idx++;
        }
    }
    std::cout<<"[KEY_FRAME] CLEANED "<<num_removed_points<<" POINTS BECAUSE OF WEIGHTS"<<std::endl;
}
