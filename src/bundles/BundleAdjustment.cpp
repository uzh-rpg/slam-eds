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

#include "BundleAdjustment.hpp"
#include <eds/bundles/PhotometricBAError.hpp>
#include <iostream>

using namespace eds::bundles;


BundleAdjustment::BundleAdjustment(const eds::bundles::Config &config, double &min_depth, double &max_depth)
{
    this->config = config;
    this->config.options = config.options;
    this->idp_lower_bound = 1.0/max_depth;
    this->idp_upper_bound = 1.0/min_depth;
    std::cout<<"[PBA] idp lower bound(1.0/max_depth): "<<this->idp_lower_bound<<std::endl;
    std::cout<<"[PBA] idp upper bound(1.0/min_depth): "<<this->idp_upper_bound<<std::endl;
}

bool BundleAdjustment::optimize(std::shared_ptr<eds::mapping::GlobalMap> &global_map,
                                uint64_t &kf_to_marginalize,
                                eds::bundles::LOSS_PARAM_METHOD loss_param_method, const bool &is_initialized)
{
    /** At least two keyframes to perform PBA **/
    if (global_map->camera_kfs.size() < 2)
    {
        std::cout<<"[PBA] No optimization: at least two KFs are required"<<std::endl;
        kf_to_marginalize = eds::mapping::NONE_KF;
        return false;
    }

    /**  Points visible and residual in the last keyframe **/
    this->visible_points.clear();
    for(auto key : global_map->camera_kfs)
    {
        this->visible_points.insert({key.first, 0.0});
    }

    /* Ceres problem **/
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::LossFunction *loss_function;
    ceres::LocalParameterization* quaternion_local_parameterization =
            new ceres::EigenQuaternionParameterization;


    /** Ceres options **/
    switch(this->config.options.linear_solver_type)
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

    options.num_threads = this->config.options.num_threads;
    options.max_num_iterations = this->config.options.max_num_iterations;
    options.function_tolerance = this->config.options.function_tolerance;
    options.minimizer_progress_to_stdout = this->config.options.minimizer_progress_to_stdout;

    switch (this->config.loss_type)
    {
    case HUBER:
        std::cout<<"[PBA] with Huber loss "<<std::endl;
        loss_function = new ceres::HuberLoss(config.loss_params[0]);
        break;

    case CAUCHY:
        std::cout<<"[PBA] with Chauchy loss "<<std::endl;
        loss_function = new ceres::CauchyLoss(config.loss_params[0]);
        break;

    default:
        loss_function = NULL;
        break;
    }

    std::cout<<"[PBA] Number of iterations: "<<this->config.options.max_num_iterations<<std::endl;
    /** Number of ceres residuals blocks **/
    int num_residuals = 0;

    /** Ids for the residuals blocks and pointer to the residual **/
    std::vector< std::pair<ceres::ResidualBlockId, ResidualInfo> > residual_blocks;

    /** For all the Keyframes (host frames) **/
    for (auto kf=global_map->camera_kfs.begin(); kf!=global_map->camera_kfs.end(); ++kf)
    {
        /** Host key frame pose **/
        Eigen::Vector3d &t_w_host = kf->second.t;
        Eigen::Quaterniond &q_w_host = kf->second.q;
        ::base::Transform3d T_w_host(q_w_host);
        T_w_host.translation() = t_w_host;

        /** Affine brightness parameters exp(a)(I - b) in host **/
        double &a_host = kf->second.a; double &b_host = kf->second.b;

        /**  For the N active points in this host keyframe [N = global_map->active_points_per_kf]**/
        int point_id = 0;
        for (auto point=global_map->point_kfs.at(kf->first).begin();
            point!=global_map->point_kfs.at(kf->first).end(); ++point)
        {
            /** Get coordinates and norm_coord **/
            ::eds::mapping::Point2d &host_coord = point->coord;
            ::eds::mapping::Point2d &host_norm_coord = point->norm_coord;
            /** Inverse depth **/
            double &idp = point->inv_depth;
            /** Compute the 3D point in host frame using the inverse depth **/
            double point_depth = 1.0/idp;
            ::base::Vector3d kp(host_norm_coord.x() * point_depth,
                                host_norm_coord.y() * point_depth,
                                point_depth);
            /** Get the patch: host keyframe template **/
            std::vector<double> &patch = point->patch;

            /** Get the residual **/
            double &residual = point->residual;

            /** Check in which keyframes (target frames) the host point is visible **/
            for (auto target_kf=global_map->camera_kfs.begin();
                target_kf!=global_map->camera_kfs.end(); ++target_kf)
            {
                /** The target keyframe is not the host keyframe **/
                if (target_kf->first != kf->first)
                {
                    bool target_is_in_problem = false;
                    /** Host key frame pose **/
                    Eigen::Vector3d &t_w_target = target_kf->second.t;
                    Eigen::Quaterniond &q_w_target = target_kf->second.q;
                    ::base::Transform3d T_w_target(q_w_target);
                    T_w_target.translation() = t_w_target;
                    /** T_target_host **/
                    ::base::Transform3d T_target_host = T_w_target.inverse() * T_w_host;
                    /** 3D point in the target frame **/
                    Eigen::Vector3d kp_target = T_target_host * kp;
                    /** Get the target keyframe frame **/
                    std::vector<double> &frame = target_kf->second.img;
                    /** Target KF Image size **/
                    double height = static_cast<double>(target_kf->second.height);
                    double width = static_cast<double>(target_kf->second.width);
                    /**  Back project the 3D point into the target keyframe **/
                    Eigen::Matrix3d &K = target_kf->second.K;
                    double &fx = K(0, 0); double &fy = K(1, 1);
                    double &cx = K(0, 2); double &cy = K(1, 2);
                    ::eds::mapping::Point2d coord(fx * (kp_target[0]/kp_target[2]) + cx,
                                                  fy * (kp_target[1]/kp_target[2]) + cy);

                    /**  Check if the coord is in the image **/
                    bool inlier = ((coord[0]>-1.0) and (coord[0]<width)) and ((coord[1]>-1.0) and (coord[1]<height));

                    /**  Point_id in kf_id is visible in target_kf **/
                    if (inlier)
                    {
                        /** Affine brightness parameters exp(a)(I - b) in target **/
                        double &a_target = target_kf->second.a; double &b_target = target_kf->second.b;
                        /** The point is visible in the target frame **/
                        if (target_is_in_problem == false) target_is_in_problem = true;
                        /**  Increase the counter (information of the number of residuals) **/
                        num_residuals += 1;
                        /** If the current host keyframe is the last keyframe
                        perform the computation to check number of visible points in the target frame **/
                        if (kf->first == global_map->last_kf_id)
                            this->visible_points.at(target_kf->first) += 1; //number of visible points ++

                        /** Create the residual: T_host, T_target, patch, target_image, intrinsics **/
                        ceres::CostFunction* cost_function =
                                PhotometricBAError::Create(&(patch), &(frame), host_coord.x(), host_coord.y(), height, width, fx, fy, cx, cy);

                        /** Add residual to the problem **/
                        ::ceres::ResidualBlockId b_id =  problem.AddResidualBlock(cost_function, loss_function,
                                t_w_host.data(), q_w_host.coeffs().data(),
                                t_w_target.data(), q_w_target.coeffs().data(),
                                &(a_host), &(b_host), &(a_target), &(b_target), &(idp));

                        /** Problem inverse depth constraints **/
                        problem.SetParameterLowerBound(&(idp), 0, this->idp_lower_bound/*1e-03*/);
                        problem.SetParameterUpperBound(&(idp), 0, this->idp_upper_bound);

                        /** Store the residual block ID **/
                        residual_blocks.push_back(
                            std::make_pair(b_id, ::eds::bundles::ResidualInfo(kf->first, &host_coord, &residual)));

                        /*************/
                        //std::cout<<"[PBA] host id: "<<kf->first<<" point_id: "<< point_id <<" target_id: "<< target_kf->first<<std::endl;
                        //std::cout<<"p_host["<<std::addressof(kf->second.t)<<"]: "<<t_w_host[0]<<" "<<t_w_host[1]<<" "<<t_w_host[2]
                        //        <<" q_host["<<std::addressof(kf->second.q)<<"]: "<< q_w_host.x()<<" "<<q_w_host.y()<<" "<<q_w_host.z()<<" "<<q_w_host.w()<<std::endl;
                        //std::cout<<"[PBA] p_target["<<std::addressof(t_w_target)<<"]: "<<t_w_target[0]<<" "<<t_w_target[1]<<" "<<t_w_target[2]
                        //        <<" q_target["<<std::addressof(q_w_target)<<"]: "<<q_w_target.x()<<" "<<q_w_target.y()<<" "<<q_w_target.z()<<" "<<q_w_target.w()<<std::endl;
                        //std::cout<<"[PBA] T_target_host:\n"<<T_target_host.matrix()<<std::endl;
                        //std::cout<<"[PBA] 3D point[host]: "<< kp[0]<<","<<kp[1]<<","<<kp[2]<<std::endl;
                        //std::cout<<"[PBA] 3D point[target]: "<< kp_target[0]<<","<<kp_target[1]<<","<<kp_target[2]<<std::endl;
                        //std::cout<<"[PBA] patch["<<std::addressof(patch)<<"]: "<<patch.size()<<std::endl;
                        //std::cout<<"[PBA] frame["<<std::addressof(frame)<<"]: ["<<height<<","<<width<<"]"<<std::endl;
                        //std::cout<<"[PBA] K:\n"<<K<<std::endl;
                        //std::cout<<"[PBA] host_coord["<<std::addressof(host_coord)<<"]: "<<cv::Point2d(host_coord)<<"->target_coord: "<<cv::Point2d(coord)<<std::endl;
                        //std::cout<<"[PBA] host_norm_coord: "<<cv::Point2d(host_norm_coord)<<std::endl;
                        //std::cout<<"[PBA] inverse_depth["<<std::addressof(idp)<<"]: "<<idp<<std::endl;
                        //std::cout<<"[PBA] inlier: "<<inlier<<std::endl;
                    }
                    //std::cout << "[PBA] Press Enter to Continue";
                    //std::cin.ignore();
                    //std::cout<<"*****"<<std::endl;

                    if (target_is_in_problem)
                    {
                        /** Problem target keyframe rotation constraints for target kf **/
                        problem.SetParameterization(q_w_target.coeffs().data(), quaternion_local_parameterization);
                        /** Problem host keyframe rotation constraints for host kf**/
                        problem.SetParameterization(q_w_host.coeffs().data(), quaternion_local_parameterization);
                    }
                }
            }
            /** Check whether we hace included the N active points **/
            if (point_id > global_map->active_points_per_kf && is_initialized)
            {
                std::cout<<"\t KF["<<kf->first<<"] BREAK "<<global_map->active_points_per_kf<<" points reached"<<std::endl;
                break;
            }
            point_id++;
        }
    }

    /** It is better to properly constrain the gauge freedom. This can be done by
     * setting the first KF pose as constant so the optimizer cannot change it.
     * This mainly improves the final trajectroy drift **/
    if (num_residuals > 0)
    {
        auto start_pose_it = global_map->camera_kfs.begin();
        problem.SetParameterBlockConstant(start_pose_it->second.t.data());
        problem.SetParameterBlockConstant(start_pose_it->second.q.coeffs().data());
    }

    std::cout<<"[PBA] BEFORE OPTIMIZATION: "<<std::endl;
    for (auto& it : global_map->camera_kfs)
    {
        Eigen::Vector3d &trans = it.second.t;
        Eigen::Quaterniond &quater = it.second.q;
        double &v = this->visible_points.at(it.first);
        std::cout<<"\tKF["<<it.first<<"] with "<<global_map->point_kfs.at(it.first).size()<<" points "
        <<" t["<<trans[0]<<","<<trans[1]<<","<<trans[2]<<"] q["<<quater.x()<<","<<quater.y()<<","
        <<quater.z()<<","<<quater.w()<<"] a["<<it.second.a<<"] b["<<it.second.b<<"] v["<<v<<" points]\n";
    }
    std::cout<<"Residuals blocks: "<<num_residuals<<std::endl;

    /** Summary and solve the optimization **/
    ceres::Solver::Summary summary;
    auto start = std::chrono::high_resolution_clock::now();
    ceres::Solve(options, &problem, &summary);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << summary.FullReport() <<std::endl;

    /** Select the keyframe to marginalize (if any) **/
    kf_to_marginalize = this->selecKFToMarginalize(global_map->camera_kfs, global_map->point_kfs,
                                            global_map->last_kf_id, global_map->prev_last_kf_id,
                                            global_map->active_points_per_kf, this->config.percent_marginalize_vis);

    bool success = summary.IsSolutionUsable();

    /** Save status information **/
    this->info.meas_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    this->info.num_iterations = summary.num_successful_steps + summary.num_unsuccessful_steps;
    this->info.time_seconds = summary.total_time_in_seconds; 
    this->info.success = success; 

    if (success != true)
    {
        return false;
    }

    /** Summary of KFs info **/
    std::cout<<"[PBA] Summary: number of residuals: "<<summary.num_residuals<<std::endl;
    std::cout<<"[PBA] AFTER OPTIMIZATION: "<<std::endl;
    for (auto& it : global_map->camera_kfs)
    {
        Eigen::Vector3d &trans = it.second.t;
        Eigen::Quaterniond &quater = it.second.q;
        double &v = this->visible_points.at(it.first);
        double s = (global_map->camera_kfs.size()==this->config.window_size)? this->score_dict.at(it.first) : ::base::NaN<double>();
        std::cout<<"\tKF["<<it.first<<"] with "<<global_map->point_kfs.at(it.first).size()<<" points "
        <<" t["<<trans[0]<<","<<trans[1]<<","<<trans[2]<<"] q["<<quater.x()<<","<<quater.y()<<","
        <<quater.z()<<","<<quater.w()<<"] a["<<it.second.a<<"] b["<<it.second.b<<"] v["<<v<<"%] s["<<s<<"]\n";
    }

    /** Use residuals and remove outliers **/
    {
        /** Get the residuals **/
        std::cout<<"[PBA] Get residual_blocks size(): "<<residual_blocks.size()<<std::endl;
        std::vector<double> tmp_norm_residual;
        this->residuals.clear();
        for (auto& it : residual_blocks)
        {
            //std::cout<<"KF["<<it.second.kf_id <<"] coord["<<it.second.coord->x()<<","<<it.second.coord->y() <<"] residual before: "<<*(it.second.residual);
            double cost; std::vector<double> residuals;
            const ceres::CostFunction *cost_function =  problem.GetCostFunctionForResidualBlock(it.first);
            residuals.resize(cost_function->num_residuals());
            problem.EvaluateResidualBlock(it.first, false, &(cost), &(residuals[0]), nullptr);
            this->residuals.push_back(residuals);
            double patch_residual = ::eds::utils::vectorNorm(residuals.begin(), residuals.end());
            tmp_norm_residual.push_back(patch_residual);
            *(it.second.residual) = patch_residual;
            //std::cout<<" after: "<<*(it.second.residual)<<std::endl;
        }
        /** Compute the Loss parameter based on residuals **/
        this->config.loss_params = this->getLossParams(tmp_norm_residual, loss_param_method);

        /** Outlier removal based on the residual (TO-DO: make it configurable/optional)**/
        double mean, st_dev;
        eds::utils::mean_std_vector(tmp_norm_residual, mean, st_dev);
        std::cout<<"[PBA] RESIDUALS MEAN: "<<mean<<" STD: "<<st_dev<<std::endl;
        if (loss_param_method != ::eds::bundles::LOSS_PARAM_METHOD::CONSTANT)
        {
            global_map->outliersRemoval(fabs(mean + (0.0 * st_dev)), is_initialized);
        }
    }

    /** Reorder points based on residual method and complete the point depth**/
    for (const auto& kf: global_map->camera_kfs)
    {
        std::cout<<"[PBA] DEPTH_COMPLETION KF: "<<kf.first<<std::endl;
        global_map->depthCompletion(global_map->point_kfs.at(kf.first));
        global_map->orderPoints(global_map->point_kfs.at(kf.first), ::eds::mapping::POINTS_SELECT_METHOD::RESIDUALS);
    }

    std::cout<<"[PBA] OPTIMIZATION FINISHED"<<std::endl;
    return success;
}

uint64_t BundleAdjustment::selecKFToMarginalize(const ::eds::mapping::CameraKFs &camera_kfs, const ::eds::mapping::PointKFs &point_kfs,
                        const uint64_t &last_kf_id, const uint64_t &prev_last_kf_id, const int &num_points_per_kf, const float &percent)
{

    /** Initial value to not marginatlize any keyframe **/
    uint64_t kf_to_marginalize = ::eds::mapping::NONE_KF;

    /* Check which keyframe has less than percent (5% default) of the points visible
    in the last keyframe **/
    int num_points_in_last_kf = num_points_per_kf;// points in last keyframe
    for (auto kf=this->visible_points.begin(); kf!=this->visible_points.end(); ++kf)
    {
        /** Always keep the last two keyframes **/
        if ((kf->first == last_kf_id) || (kf->first == prev_last_kf_id))
        {
            this->visible_points.at(kf->first) = 100.0;
        }
        else
        {
            /** Compute the percentage **/
            this->visible_points.at(kf->first) = (this->visible_points.at(kf->first) * 100.0)/num_points_in_last_kf;
        }
    }

    auto my_pair = ::eds::utils::mapToVectors(this->visible_points); // pair(keys, values)
    std::vector<uint64_t> &array_keys = my_pair.first;
    std::vector<double> &array_values = my_pair.second;
    int min_score_idx = std::distance(array_values.begin(), std::min_element(array_values.begin(), array_values.end()));
    if (array_values[min_score_idx] < percent)
    {
        kf_to_marginalize =  array_keys[min_score_idx];
    }

    /** Compute the euclidean score and store it in score_dict **/
    this->euclideanScore(camera_kfs, this->score_dict, last_kf_id, prev_last_kf_id);

    /** If the number of keyframes in global_map is equal to the desired sliding window size
     * and kf to maginalize is None we are forced to select a KF to marginalize **/
    if ((camera_kfs.size() >= this->config.window_size) and (kf_to_marginalize == ::eds::mapping::NONE_KF))
    {
        /** Get the maximum score to marginalize **/
        auto my_pair = ::eds::utils::mapToVectors(this->score_dict);
        std::vector<uint64_t> &array_keys = my_pair.first;
        std::vector<double> &array_values = my_pair.second;
        int max_score_idx = std::distance(array_values.begin(), std::max_element(array_values.begin(), array_values.end()));

        kf_to_marginalize = array_keys[max_score_idx];
    }
 
    return kf_to_marginalize;
}

void BundleAdjustment::euclideanScore(const ::eds::mapping::CameraKFs &kfs_dict, 
                        std::map<uint64_t, double> &score_dict,
                        const uint64_t &last_kf_id, const uint64_t &prev_last_kf_id,
                        const double epsilon)
{
    this->score_dict.clear();
    for(auto kv : kfs_dict)
    {
        this->score_dict.insert({kv.first, 0.0});
    }

    /** position of the last kf **/
    Eigen::Vector3d p_last_kf = kfs_dict.at(last_kf_id).t;
    //std::cout<<"[PBA] Euclidean score position last KF:\n"<<p_last_kf<<std::endl;

    /** Compute the score **/
    for(auto kv_s : score_dict)
    {
        Eigen::Vector3d p_kf_i = kfs_dict.at(kv_s.first).t;// position of the current kf
        double dist = (p_kf_i - p_last_kf).norm(); //euclian distance
        double sum = 0.0; // initial sum
        for(auto kv_kf : kfs_dict)
        {
            if ((kv_kf.first != last_kf_id) && (kv_kf.first != prev_last_kf_id) && (kv_kf.first != kv_s.first))
            {
                Eigen::Vector3d p_kf_j =  kfs_dict.at(kv_kf.first).t;
                sum += 1.0/((p_kf_i - p_kf_j).norm() + epsilon);
            }
        }
        this->score_dict.at(kv_s.first) = sqrt(dist) * sum;
    }
    return;
}

std::vector<double> BundleAdjustment::getLossParams(std::vector<double> &residuals, eds::bundles::LOSS_PARAM_METHOD method)
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
            double median = eds::utils::n_quantile_vector(residuals, residuals.size()/2);
            std::vector<double> abs_med;
            for (auto it=residuals.begin(); it!=residuals.end(); ++it)
            {
                abs_med.push_back(std::abs(*it- median));
            }
            double mad = 1.4826 * eds::utils::n_quantile_vector(abs_med, abs_med.size()/2);
            std::cout<<"[PBA] MAD: "<<mad<<" tau: "<<1.345*mad<<std::endl;
            std::vector<double> params; params.push_back(1.345 * mad);
            return params;
            break;
        }
        case STD:
        {
            double mean, st_dev;
            eds::utils::mean_std_vector(residuals, mean, st_dev);
            std::cout<<"[PBA] STD mean: "<<mean<<" st_dev: "<<st_dev<<" tau: "<<1.345*st_dev<<std::endl;
            std::vector<double>params; params.push_back(1.345 * st_dev);
            return params;
            break;
        }
    }
    return this->config.loss_params;
}
::eds::bundles::PBAInfo BundleAdjustment::getInfo()
{
    return this->info;
}