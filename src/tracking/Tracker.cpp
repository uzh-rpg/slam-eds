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

#include "Tracker.hpp"
#include <eds/utils/Utils.hpp>
#include <eds/utils/Transforms.hpp>

#include <eds/tracking/PhotometricError.hpp>
/*uncoment this and comment the other in case of testting */
//#include <eds/tracking/PhotometricErrorNC.hpp>

#include <iostream>

using namespace eds::tracking;

Tracker::Tracker(std::shared_ptr<eds::tracking::KeyFrame> kf, const eds::tracking::Config &config)
{
    (*this) = Tracker(config);
    this->kf = kf;
    std::cout<<"[TRACKER] KF address: "<<std::addressof(*(this->kf))<<std::endl;
}

Tracker::Tracker(const eds::tracking::Config &config)
{
    this->config = config;
    this->px = Eigen::Vector3d::Zero();
    this->qx = Eigen::Quaterniond::Identity();
    this->vx<<0.001, 0.001, 0.001, 0.001, 0.001, 0.001;
    this->vx.normalize();
}

void Tracker::reset(std::shared_ptr<eds::tracking::KeyFrame> kf, const Eigen::Vector3d &px, const Eigen::Quaterniond &qx, const bool &keep_velo)
{
    this->kf = kf;
    this->px = px;
    this->qx = qx;
    this->poses.clear();

    //By default we keep the velocity
    if (!keep_velo)
    {
        this->vx<<0.001, 0.001, 0.001, 0.001, 0.001, 0.001;
        this->vx.normalize();
    }
    std::cout<<"[TRACKER] New KF at address: "<<std::addressof(*(this->kf))<<std::endl;
}

void Tracker::reset(std::shared_ptr<eds::tracking::KeyFrame> kf, const Eigen::Vector3d &px, const Eigen::Quaterniond &qx, const base::Vector6d &velo)
{
    this->kf = kf;
    this->px = px;
    this->qx = qx;
    this->vx = velo;
    this->poses.clear();
}

void Tracker::set(const base::Transform3d &T_kf_ef)
{
    /** Remember the tracker works internally to search a corresponding point brightness in the event frame **/
    this->px = T_kf_ef.inverse().translation();
    this->qx = Eigen::Quaterniond(T_kf_ef.inverse().rotation());
}

void Tracker::optimize(const int &id, const std::vector<double> *event_frame, ::base::Transform3d &T_kf_ef,
                        const Eigen::Vector3d &px, const Eigen::Quaterniond &qx, 
                        const eds::tracking::LOSS_PARAM_METHOD loss_param_method)
{
    /** Init position and quaternion **/
    this->px = px;
    this->qx = qx;

    /** Optimize **/
    this->optimize(id, event_frame, T_kf_ef, loss_param_method);
}

void Tracker::optimize(const int &id, const std::vector<double> *event_frame, ::base::Transform3d &T_kf_ef, 
                        const Eigen::Matrix<double, 6, 1> &vx, 
                        const eds::tracking::LOSS_PARAM_METHOD loss_param_method)
{
    /** Init velocity parameters **/
    this->vx = vx;

    /** Optimize **/
    this->optimize(id, event_frame, T_kf_ef, loss_param_method);
}

bool Tracker::optimize(const int &id, const std::vector<double> *event_frame, ::base::Transform3d &T_kf_ef,
                        const eds::tracking::LOSS_PARAM_METHOD loss_param_method)
{
    /* Ceres problem **/
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::LossFunction *loss_function;
    ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;
    ceres::LocalParameterization* velocity_local_parameterization =
      new ceres::AutoDiffLocalParameterization<eds::tracking::UnitNormVectorAddition, 6, 6>;

    /** Ceres options **/
    switch(config.options.linear_solver_type)
    {
    case DENSE_QR:
        options.linear_solver_type = ceres::DENSE_QR;
        break; 
    case DENSE_SCHUR:
        options.linear_solver_type = ceres::DENSE_SCHUR;
        break; 
    case SPARSE_SCHUR:
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        break; 
    case SPARSE_NORMAL_CHOLESKY:
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; 
        break; 
    default:
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; 
        break; 
    }

    std::cout<<"[TRACKER] LEVEL "<<id<<std::endl;

    options.num_threads = config.options.num_threads;
    options.max_num_iterations = config.options.max_num_iterations[id];
    options.function_tolerance = config.options.function_tolerance;
    options.minimizer_progress_to_stdout = config.options.minimizer_progress_to_stdout;
    options.gradient_tolerance = 1e-08;
    options.parameter_tolerance = 1e-06;
    //options.use_nonmonotonic_steps = true; // allow the solver to jump over boulders and hills

    switch (config.loss_type)
    {
    case HUBER:
        std::cout<<"[TRACKER] with Huber loss "<<std::endl;
        loss_function = new ceres::HuberLoss(config.loss_params[0]);
        break;

    case CAUCHY:
        std::cout<<"[TRACKER] with Chauchy loss "<<std::endl;
        loss_function = new ceres::CauchyLoss(config.loss_params[0]);
        break;

    default:
        loss_function = NULL;
        break;
    }

    /** Variables for the cost function **/
    double fx, fy, cx, cy;
    fx = kf->K_ref.at<double>(0,0); fy = kf->K_ref.at<double>(1,1);
    cx = kf->K_ref.at<double>(0,2); cy = kf->K_ref.at<double>(1,2);
    std::vector<double> idp; kf->inv_depth.getIDepth(idp);
    std::vector< std::pair<ceres::ResidualBlockId, ceres::CostFunction*> > residual_blocks;

    std::cout<<"[TRACKER] fx: "<<fx<<" fy: "<<fy<<" cx: "<<cx<<" cy: "<<cy<<std::endl;
    std::cout<<"[TRACKER] grad ["<<std::addressof(kf->grad)<<"] size: "<<kf->grad.size()<<std::endl;
    std::cout<<"[TRACKER] norm_coord ["<<std::addressof(kf->norm_coord)<<"] size: "<<kf->norm_coord.size()<<std::endl;
    std::cout<<"[TRACKER] idp ["<<std::addressof(idp)<<"] size: "<<idp.size()<<std::endl;
    std::cout<<"[TRACKER] event_frame ["<<std::addressof(*event_frame)<<"] size "<<event_frame->size()<<std::endl;
    std::cout<<"[TRACKER] init px ["<<px[0]<<","<<px[1]<<","<<px[2]<<"] qx ["<<qx.x()<<","<<qx.y()<<","<<qx.z()<<","<<qx.w()<<"]\n";
    std::cout<<"[TRACKER] init vx ["<<vx[0]<<","<<vx[1]<<","<<vx[2]<<"] wx ["<<vx[3]<<","<<vx[4]<<","<<vx[5]<<"]\n";

    int num_elements = kf->norm_coord.size()/options.num_threads;
    for (int i=0; i<options.num_threads; ++i)
    {
        int extra_elements = 0;
        if (i+1 == options.num_threads)
        {
            extra_elements = kf->norm_coord.size() - (i+1) * num_elements;
        }

        int s_point = i * num_elements;
        std::cout<<"\tRESIDUAL["<<i<<"]: start point: "<<s_point<<" end point: "<<s_point+num_elements+extra_elements<<std::endl;
        ceres::CostFunction* cost_function =
                        PhotometricError::Create(&(kf->grad), &(kf->norm_coord), &(idp), &(kf->weights), event_frame,
                                            kf->img.rows, kf->img.cols, fx, fy, cx, cy, s_point, num_elements+extra_elements);
        ::ceres::ResidualBlockId b_id = problem.AddResidualBlock(cost_function, loss_function,
                    this->px.data(), this->qx.coeffs().data(), this->vx.data());
        residual_blocks.push_back(std::make_pair(b_id, cost_function));
    }

    problem.SetParameterization(this->qx.coeffs().data(), quaternion_local_parameterization);
    problem.SetParameterization(this->vx.data(), velocity_local_parameterization);

    ceres::Solver::Summary summary;
    auto start = std::chrono::high_resolution_clock::now();
    ceres::Solve(options, &problem, &summary);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << summary.FullReport() << "\n";
    std::cout << "[TRACKER] px ["<<px[0]<<","<<px[1]<<","<<px[2]<<"] qx ["<<qx.x()<<","<<qx.y()<<","<<qx.z()<<","<<qx.w()<<"]\n";
    std::cout << "[TRACKER] vx ["<<vx[0]<<","<<vx[1]<<","<<vx[2]<<"] wx ["<<vx[3]<<","<<vx[4]<<","<<vx[5]<<"]\n";

    /** Save status information **/
    this->info.meas_time_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    this->info.num_points = summary.num_residuals;
    this->info.num_iterations = summary.num_successful_steps + summary.num_unsuccessful_steps;
    this->info.time_seconds = summary.total_time_in_seconds; 
    this->info.success = summary.IsSolutionUsable(); 

    /** Valid numerical solution, converge or no-coverger
     * depending on the number of max iterations **/
    if (summary.IsSolutionUsable())
    {
        /** Return the inverse **/
        T_kf_ef = this->getTransform().inverse();

        /** Get the residuals into the Keyframe. There are as many residuals as active points **/
        this->kf->residuals.resize(kf->norm_coord.size());
        std::vector<double*> params;
        params.push_back(this->px.data()); params.push_back(this->qx.coeffs().data()); params.push_back(this->vx.data());
        for (int i=0; i<options.num_threads; ++i)
        {
            ceres::CostFunction *cost_function = residual_blocks[i].second; 
            cost_function->Evaluate(&(params[0]), &(this->kf->residuals[i*num_elements]), nullptr);
        }

        /** Compute the Loss parameter based on the points residuals **/
        this->config.loss_params = this->getLossParams(loss_param_method);

        return true;
    }
    else
    {
        return false;
    }
}

::base::Transform3d Tracker::getTransform()
{
    ::eds::SE3 se3(this->qx, this->px);
    base::Transform3d pose = base::Transform3d::Identity();
    pose.matrix() = se3.matrix();
    return pose;
}

::base::Transform3d Tracker::getTransform(bool &result)
{
    ::eds::SE3 se3(this->qx, this->px);
    this->poses.push_back(se3);
    result = this->getFilteredPose(se3);
    base::Transform3d pose = base::Transform3d::Identity();
    if (result)
        pose.matrix() = se3.matrix();
    return pose;
}

Eigen::Matrix<double, 6, 1> &Tracker::getVelocity()
{
    return this->vx;
}

const Eigen::Vector3d Tracker::linearVelocity()
{
    Eigen::Vector3d v;
    v<<this->vx[0], this->vx[1], this->vx[2];
    return v;
}

const Eigen::Vector3d Tracker::angularVelocity()
{
    Eigen::Vector3d w;
    w<<this->vx[3], this->vx[4], this->vx[5];
    return w;
}

std::vector<double> Tracker::getLossParams(eds::tracking::LOSS_PARAM_METHOD method)
{

    /** See Pages 19, 20 and 75 in Simon Klenk's Msc Thesis:
    * https://vision.in.tum.de/_media/members/demmeln/klenk2020ma.pdf**/
    switch(method)
    {
        case CONSTANT:
        {
            break;
        }
        case MAD:
        {
            std::vector<double> abs_med;
            double median = eds::utils::n_quantile_vector(this->kf->residuals, this->kf->residuals.size()/2);
            for (auto it=this->kf->residuals.begin(); it!=this->kf->residuals.end(); ++it)
            {
                abs_med.push_back(std::abs(*it- median));
            }
            double mad = 1.4826 * eds::utils::n_quantile_vector(abs_med, abs_med.size()/2);
            std::cout<<"[TRACKER] MAD median: "<<median<<" MAD: "<<mad<<" tau: "<<1.345*mad<<std::endl;
            std::vector<double> params; params.push_back(1.345 * mad);
            return params;
            break;
        }
        case STD:
        {
            double mean, st_dev;
            eds::utils::mean_std_vector(this->kf->residuals, mean, st_dev);
            std::cout<<"[TRACKER] STD mean: "<<mean<<" st_dev: "<<st_dev<<" tau: "<<1.345*st_dev<<std::endl;
            std::vector<double>params; params.push_back(1.345 * st_dev);
            return params;
            break;
        }
    }
    return this->config.loss_params;
}

std::vector<cv::Point2d> Tracker::getCoord(const bool &delete_out_point)
{
    std::vector<cv::Point2d> coord;

    double fx, fy, cx, cy;
    fx = this->kf->K_ref.at<double>(0,0); fy = this->kf->K_ref.at<double>(1,1);
    cx = this->kf->K_ref.at<double>(0,2); cy = this->kf->K_ref.at<double>(1,2);

    /** Get the rotation matrix R_ef_kf **/
    Eigen::Matrix3d R = this->qx.toRotationMatrix();

    /** Reset squared nom flow **/
    this->squared_norm_flow = 0;

    /** Iterators **/
    int idx = 0;
    int num_removed_points = 0;
    auto it_c = this->kf->norm_coord.begin();
    auto it_i = this->kf->inv_depth.begin();
    for (; it_c != this->kf->norm_coord.end() && it_i != this->kf->inv_depth.end();)
    {
        Eigen::Vector3d p;
        p[2] = 1.0/::eds::mapping::mu(*it_i);
        p[0] = (*it_c).x * p[2];
        p[1] = (*it_c).y * p[2];
        p = R * p + this->px; // point in the event frame

        /** Project the point into the event frame **/
        double xp = fx * (p[0]/p[2]) + cx;
        double yp = fy * (p[1]/p[2]) + cy;

        /** Check whether the point is out of the frame **/
        bool outlier = ((xp<0.0 || xp>this->kf->img.cols) || (yp<0.0 || yp>this->kf->img.rows));
        if (delete_out_point & outlier)
        {
            //std::cout<<"GOING TO DELETE POINT: "<<idx<<" xp: "<<xp<<" yp: "<<yp<<std::endl;
            auto its = this->kf->erasePoint(idx);
            it_c = its.norm_coord; it_i = its.inv_depth; // next elements
            num_removed_points++;
        }
        else
        {
            coord.push_back(cv::Point2d(xp, yp));
            cv::Point2d track =  cv::Point2d(xp, yp) - this->kf->coord[idx]; //new point - old point
            this->kf->tracks[idx] = Eigen::Vector2d(track.x, track.y);//flow in tracks
            this->squared_norm_flow += this->kf->tracks[idx].squaredNorm();
            ++it_c; ++it_i; //next_elements
            idx++;
        }
    }

    /** Mean of the squared norm flow **/
    this->squared_norm_flow /= idx;

    std::cout<<"[TRACKER] GET_COORD REMOVED "<<num_removed_points<<" POINTS"<<std::endl;

    return coord; /** points in event frame **/
}

void Tracker::trackPoints(const cv::Mat &event_frame, const uint16_t &patch_radius)
{
    /* Coordinates of warpped points **/
    std::vector<cv::Point2d> coord = this->getCoord(true);

    /** Warpped gradient images **/
    cv::Mat grad_x = this->kf->getGradient_x(coord, "bilinear");
    cv::Mat grad_y = this->kf->getGradient_y(coord, "bilinear");

    /** Gradient patches **/
    std::vector<cv::Mat> grad_patches_x, grad_patches_y;
    eds::utils::splitImageInPatches(grad_x, coord, grad_patches_x, patch_radius);
    eds::utils::splitImageInPatches(grad_y, coord, grad_patches_y, patch_radius);

    /** Event frame patches **/
    std::vector<cv::Mat> event_patches;
    eds::utils::splitImageInPatches(event_frame, coord, event_patches, patch_radius);

    /** Brightness model change **/
    //cv::Mat model = this->kf->getModel(coord, this->vx, this->wx, "bilinear");
    //std::vector<cv::Mat> model_patches;
    //eds::utils::splitImageInPatches(model, coord, model_patches, patch_radius);

    /** Compute the optical flow (pixel displacement)**/
    int idx = 0;
    int num_removed_points = 0;
    auto it_c = coord.begin();
    auto it_patch_x = grad_patches_x.begin();
    auto it_patch_y = grad_patches_y.begin();
    auto it_patch_e = event_patches.begin();
    for (; it_c != coord.end();)
    {
        /** KLT tracker **/
        Eigen::Vector2d f = ::eds::utils::kltTracker(*it_patch_x, *it_patch_y, *it_patch_e);
        this->kf->flow[idx] = f;//2D-flow in x-y axis in this order
        /** Update the active points tracks in keyframe **/
        //std::cout<<"[TRACKER] point["<<i<<"] size "<<grad_patches_y[i].size()<<" f["<<f[0]<<","<<f[1]<<"]"
        //         <<" flow["<<this->kf->flow[i][0]<<","<<this->kf->flow[i][1]<<"]"<<std::endl;
        this->kf->tracks[idx] += f;//2D-flow in x-y axis in this order
        bool oulier = false; //f.norm() > 1.0; 
        if (oulier)
        {
            this->kf->erasePoint(idx);
            it_c = coord.erase(coord.begin()+idx);
            it_patch_x = grad_patches_x.erase(grad_patches_x.begin()+idx);
            it_patch_y = grad_patches_y.erase(grad_patches_y.begin()+idx);
            it_patch_e = event_patches.erase(event_patches.begin()+idx);
            num_removed_points++;
        }
        else
        {
            ++it_c; ++it_patch_x; ++it_patch_y; ++it_patch_e;
            idx++;
        }
    }
    std::cout<<"[TRACKER] KLT TRACKER REMOVED "<<num_removed_points<<" POINTS"<<std::endl;
}

void Tracker::trackPointsPyr(const cv::Mat &event_frame, const size_t num_level)
{

    /** patch radius based on the num_level (min patch size 3x3 at the coarsest level) **/
    uint16_t patch_radius = (3*std::pow(2.0, static_cast<double>(num_level-1))) + num_level;
    patch_radius = patch_radius / 2;

    /* Coordinates of warpped points **/
    std::vector<cv::Point2d> coord = this->getCoord(true);

    /** Warpped gradient images **/
    cv::Mat grad_x = this->kf->getGradient_x(coord, "bilinear");
    cv::Mat grad_y = this->kf->getGradient_y(coord, "bilinear");

    /** Gradient patches **/
    std::vector<cv::Mat> grad_patches_x, grad_patches_y;
    eds::utils::splitImageInPatches(grad_x, coord, grad_patches_x, patch_radius);
    eds::utils::splitImageInPatches(grad_y, coord, grad_patches_y, patch_radius);

    /** Event frame patches **/
    std::vector<cv::Mat> event_patches;
    eds::utils::splitImageInPatches(event_frame, coord, event_patches, patch_radius);

    /** Compute the optical flow (pixel displacement)
     * per point with num pyramid level **/
    for (size_t i=0; i<coord.size(); ++i)
    {
        /** Create pyramid **/
        std::vector<cv::Mat> grad_pyr_patches_x;
        eds::utils::pyramidPatches(grad_patches_x[i], grad_pyr_patches_x, num_level);
        std::vector<cv::Mat> grad_pyr_patches_y;
        eds::utils::pyramidPatches(grad_patches_y[i], grad_pyr_patches_y, num_level);
        std::vector<cv::Mat> event_pyr_patches;
        eds::utils::pyramidPatches(event_patches[i], event_pyr_patches, num_level);

        /** Compute flow per pyramid level **/
        Eigen::Vector2d f(0, 0);
        for (int j=num_level-1; j>-1; --j)
        {
            double scale = std::pow(2.0, static_cast<double>(j));
            f += (1.0/scale) * ::eds::utils::kltTracker(grad_pyr_patches_x[j], grad_pyr_patches_y[j], event_pyr_patches[j])/scale;
            //std::cout<<"point["<<i<<"] level["<<j<<"] scale: "<<scale<<" patch size: "
            //        <<event_pyr_patches[j].size()<<" f["<<f[0]<<","<<f[1]<<"]"<<std::endl;
        }
        //f = f/num_level;
        //std::cout<<"point["<<i<<"]"<<" f["<<f[0]<<","<<f[1]<<"]"<<std::endl;
        /** Store the flow (displacement) **/
        this->kf->flow[i] += f;//2D-flow in x-y axis in this order
        /** Update the active points tracks in keyframe **/
        this->kf->tracks[i] += f;
        //std::cout<<"**********************"<<std::endl;
    }
}

std::vector<cv::Point2d> Tracker::trackPointsAlongEpiline(const cv::Mat &event_frame, const uint16_t &patch_radius,
                                        const int &border_type, const uint8_t &border_value)
{
    /** Brightness model change **/
    cv::Mat model = this->kf->getModel(this->linearVelocity(), this->angularVelocity(), "bilinear");
    std::vector<cv::Mat> model_patches;
    eds::utils::splitImageInPatches(model, this->kf->coord, model_patches, patch_radius, border_type, border_value);

    /** Get Epilines **/
    std::vector<cv::Vec3d> lines;
    cv::computeCorrespondEpilines(this->kf->coord, 1, this->getFMatrix(), lines);
    std::cout<<"** [TRACKER] EPILINE SEARCH WITH: "<<lines.size()<<" lines"<<std::endl;

    /** Create a bigger event frame image (padding) **/
    cv::Mat event_img;
    cv::copyMakeBorder(event_frame, event_img, patch_radius, patch_radius, patch_radius, patch_radius, border_type, border_value);
    event_img.convertTo(event_img, CV_32FC1);
    std::cout<<"** [TRACKER] EPILINE SEARCH EVENT IMG: "<<event_img.size()<<std::endl;
    cv::imwrite("/tmp/event_img.png", this->kf->viz(event_img));
    cv::imwrite("/tmp/model_img.png", this->kf->viz(model));

    std::vector<cv::Point2d> tracker_coord;// = this->getCoord(false);
    /** Search correspondence along the epiline **/
    int idx = 0;
    int num_removed_points = 0;
    auto it_c =  this->kf->coord.begin();
    auto it_nc =  this->kf->norm_coord.begin();
    auto it_id = this->kf->inv_depth.begin();
    //auto it_tr =  tracker_coord.begin();
    auto it_p =  model_patches.begin();
    auto it_l =  lines.begin();
    for (; it_p != model_patches.end();)
    {
        (*it_p).convertTo(*it_p, CV_32FC1);
        //cv::imwrite("/tmp/model_patch.png", this->kf->viz(model_patches[i]));
        cv::Mat result;
        cv::Point2d p_ssd = eds::utils::matchTemplate(event_img, *it_p, cv::TM_SQDIFF_NORMED);
        cv::Point2d p_ncc = eds::utils::matchTemplate(event_img, *it_p, cv::TM_CCORR_NORMED);
        //double idepth = eds::mapping::mu(*it_id);
        //double sigma = std::sqrt(eds::mapping::sigma2(*it_id));
        //cv::Point2d p = ::eds::utils::searchAlongEpiline(event_frame.size(), event_img, *it_p, *it_l,
                       //                                 this->getTransform(), *it_nc, idepth, sigma,
                       //                                 this->kf->K_ref, ::eds::utils::ZSAD);
        //std::cout<<"** [TRACKER] EPILINE SEARCH coord["<<idx<<"]: "<<*it_c<<" p_ssd: "<<p_ssd<<" p_ncc: "<<p_ncc
        //<<" p: " <<p<<" diff: "<<std::fabs(cv::norm(p_ssd)-cv::norm(p_ncc)) <<std::endl;//" tracker_coord: "<<*it_tr<<std::endl;

        if (std::fabs(cv::norm(p_ssd)-cv::norm(p_ncc)) > 5.0)
        {
            this->kf->erasePoint(idx);
            num_removed_points++;
            it_p = model_patches.erase(model_patches.begin()+idx);
            //it_tr = tracker_coord.erase(tracker_coord.begin()+idx);
        }
        else
        {
            ++it_c; ++it_nc; ++it_id; /*++it_tr;*/ ++it_p; ++it_l;
            ++idx;
            tracker_coord.push_back(p_ssd);
        }
    }
    std::cout<<"** [TRACKER] AVAILABLE POINTS: "<<this->kf->coord.size()<<" DELETED: "<<num_removed_points<<std::endl;

    return tracker_coord;
}

cv::Mat Tracker::getEMatrix()
{
    /** Current transformation **/
    base::Transform3d T_ef_kf = this->getTransform();

    /** Translation cross product matrix **/
    cv::Mat_<double> t_x = cv::Mat_<double>::zeros(3, 3);
    t_x(0,1) = -T_ef_kf.translation()[2]; t_x(0,2) = T_ef_kf.translation()[1];
    t_x(1,0) = T_ef_kf.translation()[2]; t_x(1,2) = -T_ef_kf.translation()[0];
    t_x(2,0) = -T_ef_kf.translation()[1]; t_x(2,1) = T_ef_kf.translation()[0];

    /** Rotation **/
    Eigen::Matrix3d R_ = T_ef_kf.rotation();
    cv::Mat_<double> R = cv::Mat_<double>::zeros(3,3);
    R(0,0) = R_(0,0); R(0,1) = R_(0,1); R(0,2) = R_(0,2);
    R(1,0) = R_(1,0); R(1,1) = R_(1,1); R(1,2) = R_(1,2);
    R(2,0) = R_(2,0); R(2,1) = R_(2,1); R(2,2) = R_(2,2);

    /** Fundamental matrix **/
    return t_x * R;
}

cv::Mat Tracker::getFMatrix()
{
    /** Intrinsics **/
    double fx, fy, cx, cy;
    fx = this->kf->K_ref.at<double>(0,0); fy = this->kf->K_ref.at<double>(1,1);
    cx = this->kf->K_ref.at<double>(0,2); cy = this->kf->K_ref.at<double>(1,2);
    cv::Mat_<double> K = cv::Mat_<double>::eye(3,3);
    K(0,0) = fx; K(1,1) = fy; K(0,2) = cx; K(1,2) = cy;

    return K.t().inv() * this->getEMatrix() * K.inv();
}

::eds::tracking::TrackerInfo Tracker::getInfo()
{
    return this->info;
}

bool Tracker::getFilteredPose(eds::SE3 &pose, const size_t &mean_filter_size)
{
    if (mean_filter_size < 2)
    {
        pose = this->poses.back();
        return true;
    }

    //if (this->poses.size() < mean_filter_size/2.0)
    if (this->poses.size() < mean_filter_size)
    {
        return false;
    }

    size_t num_elements = std::min(this->poses.size(), mean_filter_size);
    std::cout<<"[TRACKER] FILTER NUM ELEMENTS: "<<this->poses.size()<<std::endl;

    static base::Vector6d P = base::Vector6d::Zero();

    // Take the first rotation q0 as the reference
    // Then, for the remainders rotations qi, instead of
    // averaging directly the qi's, average the incremental rotations
    // q0^-1 * q_i (in the Lie algebra), and then get the original mean
    // rotation by multiplying the mean incremental rotation on the left
    // by q0.

    base::Quaterniond tf_q0 = this->poses[this->poses.size()-num_elements].unit_quaternion();
    const base::Quaterniond q0(tf_q0.w(), tf_q0.x(), tf_q0.y(), tf_q0.z());
    const base::Quaterniond q0_inv = q0.inverse();

    for (size_t i=this->poses.size()-num_elements; i !=this->poses.size(); ++i)
    {

        const base::Quaterniond& tf_q = this->poses[i].unit_quaternion();
        const base::Quaterniond q(tf_q.w(), tf_q.x(), tf_q.y(), tf_q.z());
        const base::Quaterniond q_inc = q0_inv * q;

        const base::Vector3d& t = this->poses[i].translation();

        eds::SE3 T(q_inc, t);

        P += T.log();
    }

    P /= num_elements;
    eds::SE3 T = SE3::exp(P);

    const Eigen::Vector3d& t_mean = T.translation();
    const base::Quaterniond q_mean = q0 * T.unit_quaternion();

    eds::SE3 filtered_pose(q_mean, t_mean);
    pose = filtered_pose;

    return true;
}

bool Tracker::needNewKeyframe(const double &weight_factor)
{
    double image_weight = (this->kf->img.cols + this->kf->img.rows) * weight_factor;
    return (image_weight * sqrtf(this->squared_norm_flow) / (this->kf->img.cols + this->kf->img.rows)) > 1;
}
