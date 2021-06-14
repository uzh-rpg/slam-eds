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

#ifndef _EDS_BUNDLES_PHOTOMETRIC_BA_ERROR_HPP_
#define _EDS_BUNDLES_PHOTOMETRIC_BA_ERROR_HPP_

#include <eds/mapping/Types.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/cubic_interpolation.h>
#include <memory>

namespace eds { namespace bundles {
 
struct PhotometricBAError
{
    /** Constructor
     * patch: D X D patch (template in host keyframe)
     * coord: coord [X, Y] in host keyframe
     * target_frame: H x W frame of target keyframe
     * fx, fy, cx, cy: target keyframe intrinsics
     * **/
    PhotometricBAError(const std::vector<double> *patch_,
                    const std::vector<double> *target_frame_,
                    const double &px_x_, const double &px_y_,
                    const int &height_, const int &width_,
                    const double &fx_, const double &fy_,
                    const double &cx_, const double &cy_,
                    const double &exp_time_h_ = 1.0,
                    const double &exp_time_t_ = 1.0)
    :px_x(px_x_), px_y(px_y_), height(height_), width(width_),
    fx(fx_), fy(fy_), cx(cx_), cy(cy_), exp_time_h(exp_time_h_), exp_time_t(exp_time_t_)
    {
        /** Get the parameters **/
        this->patch = patch_;

        /** Create the grid for the target frame interpolate **/
        target_grid.reset(new ceres::Grid2D<double, 1> (target_frame_->data(), 0, height, 0, width));
        target_interp.reset(new ceres::BiCubicInterpolator< ceres::Grid2D<double, 1> > (*target_grid));

        //std::cout<<"[PHOTO_BA_ERROR] patch["<<this->patch<<"]: "<<this->patch->size()<<std::endl;
        //std::cout<<"[PHOTO_BA_ERROR] target_frame_["<<target_frame_<<"]"<<std::endl;
    }

    /** X-axis pixel offset in pattern **/
    inline double px_x_offset(const size_t &i) const
    {
        /** i is between 0 and 7 **/
        switch(i)
        {
        case 0:
        case 4:
        case 7:
            return 0.0;
            break;
        case 1:
        case 3:
            return -1.0;
            break;
        case 2:
            return -2.0;
            break;
        case 5:
            return 1.0;
            break;
        case 6:
            return 2.0;
            break;
        default:
            return std::numeric_limits<double>::quiet_NaN();
            break;
        }
    }

    /** Y-axis pixel offset in pattern **/
    inline double px_y_offset(const size_t &i) const
    {
        /** i is between 0 and 7 **/
        switch(i)
        {
        case 0:
            return 2.0;
            break;
        case 1:
            return 1.0;
            break;
        case 2:
        case 6:
        case 7:
            return 0.0;
            break;
        case 3:
        case 5:
            return -1.0;
            break;
        case 4:
            return -2.0;
            break;
        default:
            return std::numeric_limits<double>::quiet_NaN();
            break;
        }
    }


    template <typename T>
    bool operator()(const T* p_host, const T* q_host,
                    const T* p_target, const T* q_target,
                    const T* a_h, const T* b_h,
                    const T* a_t, const T* b_t,
                    const T* idp, T* residual) const
    {
        //std::cout<<"\n\n************************************************"<<std::endl;
        //std::cout<<"p_host["<<&(p_host[0])<<"]: "<<p_host[0]<<","<<p_host[1]<<","<<p_host[2]<<std::endl;
        //std::cout<<"q_host["<<&(q_host[0])<<"]: "<<q_host[0]<<","<<q_host[1]<<","<<q_host[2]<<","<<q_host[3]<<std::endl;
        //std::cout<<"p_target["<<std::addressof(p_target)<<"]: "<<p_target[0]<<","<<p_target[1]<<","<<p_target[2]<<std::endl;
        //std::cout<<"q_target["<<std::addressof(q_target)<<"]: "<<q_target[0]<<","<<q_target[1]<<","<<q_target[2]<<","<<q_target[3]<<std::endl;

        //std::cout<<"idp["<<std::addressof(idp)<<"]: "<<idp[0]<<std::endl;

        /** Host Key frame pose **/
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w_h(p_host);
        Eigen::Map<const Eigen::Quaternion<T>> q_w_h(q_host);
        //std::cout<<"p_w_h: "<<p_w_h<<std::endl;
        //std::cout<<"q_w_h: "<<q_w_h.toRotationMatrix()<<std::endl;

        /** Target Key frame pose **/
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w_t(p_target);
        Eigen::Map<const Eigen::Quaternion<T>> q_w_t(q_target);
        //std::cout<<"p_w_t: "<<p_w_t<<std::endl;
        //std::cout<<"q_w_t: "<<q_w_t.toRotationMatrix()<<std::endl;

        /** Relative transformation between keyframes **/
        Eigen::Quaternion<T> q_t_w = q_w_t.conjugate();
        Eigen::Quaternion<T> q_t_h = (q_t_w * q_w_h).normalized();
        Eigen::Matrix<T, 3, 1> p_t_h = q_t_w * (p_w_h - p_w_t);
        //std::cout<<"q_t_h["<<q_t_h.w()<<","<<q_t_h.vec()<<"]"<<std::endl;

        /** Get the 3D point in host frame **/
        Eigen::Matrix<T, 3, 1> point_host;
        point_host[2] = T(1.0)/idp[0]; // z-element

        /** Compute photometric residual **/
        for (size_t i=0; i<this->patch->size(); ++i)
        {
            /* Get the value of this pixel in the patch **/
            T model = T((*this->patch)[i]);

            /** Get the pixel coordinate in the host frame **/
            double xp = px_x + this->px_x_offset(i);
            double yp = px_y + this->px_y_offset(i);
            //std::cout<<"xp: "<<xp <<" yp: "<<yp<<std::endl;

            /** Get the 3D point from the inverse depth parameter in host frame **/
            point_host[0] = T((xp-cx)/fx)*point_host[2]; point_host[1] = T((yp-cy)/fy)*point_host[2];
            //std::cout<<"point_host: "<<point_host[0]<<" "<<point_host[1]<<" "<<point_host[2]<<std::endl;

            /** Transform the point in the target frame **/
            Eigen::Matrix<T, 3, 1> point_target; //3D point in target frame
            point_target = (q_t_h * point_host) + p_t_h;
            //std::cout<<"point_target: "<<point_target[0]<<" "<<point_target[1]<<" "<<point_target[2]<<std::endl;

            /** Project point on the target frame (pixel coord) **/
            T xt = fx * T(point_target[0]/point_target[2]) + cx;
            T yt = fy * T(point_target[1]/point_target[2]) + cy;
            //std::cout<<"xt: "<<xt<<" yt: "<<yt<<std::endl;

            /** Compute the residual **/
            T value;
            target_interp->Evaluate(yt, xt, &value);
            //std::cout<<"value: "<<value<<std::endl;
            //std::cout<<"model: "<<model<<std::endl;
            //std::cout<<"diff: "<<value-model<<std::endl;
            T affine_correction = ((T(exp_time_t) * ceres::exp(a_t[0])) / (T(exp_time_h) * ceres::exp(a_h[0])));
            residual[i] = (value - b_t[0]) - ( affine_correction * (model-b_h[0]) );
            //std::cout<<"residual: "<<residual[0]<<std::endl;
        }

       return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const std::vector<double> *patch,
                                       const std::vector<double> *target_frame,
                                       const double &px_x, const double &px_y,
                                       const int &height, const int &width,
                                       const double &fx, const double &fy,
                                       const double &cx, const double &cy,
                                       const double &exp_time_h = 1.0,
                                       const double &exp_time_t = 1.0)
    {   
        int size = patch->size();
        PhotometricBAError* functor = new PhotometricBAError(patch, target_frame, px_x, px_y, height, width,
                                                            fx, fy, cx, cy, exp_time_h, exp_time_t);
        return new ceres::AutoDiffCostFunction<PhotometricBAError, ceres::DYNAMIC, 3, 4, 3, 4, 1, 1, 1, 1, 1>(functor, size);
    }

    static constexpr double eps = 1e-05;

    const std::vector<double> *patch; // bundle patch (template in host keyframe)
    double px_x, px_y; // Pixel coordinates of point in host frame
    int height, width; // height and width of the target image
    double fx, fy, cx, cy; // Intrinsics
    double exp_time_h, exp_time_t; // Image exposure time h=host, t=target (1.0 by default)
    std::unique_ptr< ceres::Grid2D<double, 1> > target_grid; // 2D grid for the frame
    std::unique_ptr< ceres::BiCubicInterpolator< ceres::Grid2D<double, 1> > > target_interp; // Grid interpolation

};

struct PhotometricBAErrorFixedDepth
{
    /** Constructor: Test class with a fixed depth
     * patch: D X D patch (template in host keyframe)
     * coord: coord [X, Y] in host keyframe
     * target_frame: H x W frame of target keyframe
     * idepth: inverse depth for the point in residual
     * fx, fy, cx, cy: target keyframe intrinsics
     * **/
    PhotometricBAErrorFixedDepth(const std::vector<double> *patch_,
                    const std::vector<double> *target_frame_,
                    const double &px_x_, const double &px_y_,
                    const int &height_, const int &width_,
                    const double &idepth_,
                    const double &fx_, const double &fy_,
                    const double &cx_, const double &cy_,
                    const double &exp_time_h_ = 1.0,
                    const double &exp_time_t_ = 1.0)

    :px_x(px_x_), px_y(px_y_), height(height_), width(width_), idepth(idepth_),
    fx(fx_), fy(fy_), cx(cx_), cy(cy_), exp_time_h(exp_time_h_), exp_time_t(exp_time_t_)
    {
        /** Get the parameters **/
        this->patch = patch_;

        /** Create the grid for the target frame interpolate **/
        target_grid.reset(new ceres::Grid2D<double, 1> (target_frame_->data(), 0, height, 0, width));
        target_interp.reset(new ceres::BiCubicInterpolator< ceres::Grid2D<double, 1> > (*target_grid));

        //std::cout<<"[PHOTO_BA_ERROR] patch["<<this->patch<<"]: "<<this->patch->size()<<std::endl;
        //std::cout<<"[PHOTO_BA_ERROR] target_frame_["<<target_frame_<<"]"<<std::endl;
    }

    /** X-axis pixel offset in pattern **/
    inline double px_x_offset(const size_t &i) const
    {
        /** i is between 0 and 7 **/
        switch(i)
        {
        case 0:
        case 4:
        case 7:
            return 0.0;
            break;
        case 1:
        case 3:
            return -1.0;
            break;
        case 2:
            return -2.0;
            break;
        case 5:
            return 1.0;
            break;
        case 6:
            return 2.0;
            break;
        default:
            return std::numeric_limits<double>::quiet_NaN();
            break;
        }
    }

    /** Y-axis pixel offset in pattern **/
    inline double px_y_offset(const size_t &i) const
    {
        /** i is between 0 and 7 **/
        switch(i)
        {
        case 0:
            return 2.0;
            break;
        case 1:
            return 1.0;
            break;
        case 2:
        case 6:
        case 7:
            return 0.0;
            break;
        case 3:
        case 5:
            return -1.0;
            break;
        case 4:
            return -2.0;
            break;
        default:
            return std::numeric_limits<double>::quiet_NaN();
            break;
        }
    }

    template <typename T>
    bool operator()(const T* p_host, const T* q_host,
                    const T* p_target, const T* q_target,
                    const T* a_h, const T* b_h,
                    const T* a_t, const T* b_t,
                    T* residual) const
    {
        //std::cout<<"\n\n************************************************"<<std::endl;
        //std::cout<<"p_host["<<&(p_host[0])<<"]: "<<p_host[0]<<","<<p_host[1]<<","<<p_host[2]<<std::endl;
        //std::cout<<"q_host["<<&(q_host[0])<<"]: "<<q_host[0]<<","<<q_host[1]<<","<<q_host[2]<<","<<q_host[3]<<std::endl;
        //std::cout<<"p_target["<<std::addressof(p_target)<<"]: "<<p_target[0]<<","<<p_target[1]<<","<<p_target[2]<<std::endl;
        //std::cout<<"q_target["<<std::addressof(q_target)<<"]: "<<q_target[0]<<","<<q_target[1]<<","<<q_target[2]<<","<<q_target[3]<<std::endl;

        //std::cout<<"idp["<<std::addressof(idp)<<"]: "<<idp[0]<<std::endl;

        /** Host Key frame pose **/
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w_h(p_host);
        Eigen::Map<const Eigen::Quaternion<T>> q_w_h(q_host);
        //std::cout<<"p_w_h: "<<p_w_h<<std::endl;
        //std::cout<<"q_w_h: "<<q_w_h.toRotationMatrix()<<std::endl;

        /** Target Key frame pose **/
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w_t(p_target);
        Eigen::Map<const Eigen::Quaternion<T>> q_w_t(q_target);
        //std::cout<<"p_w_t: "<<p_w_t<<std::endl;
        //std::cout<<"q_w_t: "<<q_w_t.toRotationMatrix()<<std::endl;

        /** Relative transformation between keyframes **/
        Eigen::Quaternion<T> q_t_w = q_w_t.conjugate();
        Eigen::Quaternion<T> q_t_h = (q_t_w * q_w_h).normalized();
        Eigen::Matrix<T, 3, 1> p_t_h = q_t_w * (p_w_h - p_w_t);
        //std::cout<<"q_t_h["<<q_t_h.w()<<","<<q_t_h.vec()<<"]"<<std::endl;

        /** Get the 3D point in host frame **/
        Eigen::Matrix<T, 3, 1> point_host;
        point_host[2] = T(1.0/this->idepth); // z-element

        /** Compute photometric residual **/
        for (size_t i=0; i<this->patch->size(); ++i)
        {
            /* Get the value of this pixel in the patch **/
            T model = T((*this->patch)[i]);

            /** Get the pixel coordinate in the host frame **/
            double xp = px_x + this->px_x_offset(i);
            double yp = px_y + this->px_y_offset(i);
            //std::cout<<"xp: "<<xp <<" yp: "<<yp<<std::endl;

            /** Get the 3D point from the inverse depth parameter in host frame **/
            point_host[0] = T((xp-cx)/fx)*point_host[2]; point_host[1] = T((yp-cy)/fy)*point_host[2];
            //std::cout<<"point_host: "<<point_host[0]<<" "<<point_host[1]<<" "<<point_host[2]<<std::endl;

            /** Transform the point in the target frame **/
            Eigen::Matrix<T, 3, 1> point_target; //3D point in target frame
            point_target = (q_t_h * point_host) + p_t_h;
            //std::cout<<"point_target: "<<point_target[0]<<" "<<point_target[1]<<" "<<point_target[2]<<std::endl;

            /** Project point on the target frame (pixel coord) **/
            T xt = fx * T(point_target[0]/point_target[2]) + cx;
            T yt = fy * T(point_target[1]/point_target[2]) + cy;
            //std::cout<<"xt: "<<xt<<" yt: "<<yt<<std::endl;

            /** Compute the residual **/
            T value;
            target_interp->Evaluate(yt, xt, &value);
            //std::cout<<"value: "<<value<<std::endl;
            //std::cout<<"model: "<<model<<std::endl;
            //std::cout<<"diff: "<<value-model<<std::endl;
            T affine_correction = ((T(exp_time_t) * ceres::exp(a_t[0])) / (T(exp_time_h) * ceres::exp(a_h[0])));
            residual[i] = (value - b_t[0]) - ( affine_correction * (model-b_h[0]) );
            //std::cout<<"residual: "<<residual[0]<<std::endl;
        }

       return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const std::vector<double> *patch,
                                       const std::vector<double> *target_frame,
                                       const double &px_x, const double &px_y,
                                       const int &height, const int &width,
                                       const double &idepth,
                                       const double &fx, const double &fy,
                                       const double &cx, const double &cy,
                                       const double &exp_time_h = 1.0,
                                       const double &exp_time_t = 1.0)
    {   
        int size = patch->size();
        PhotometricBAErrorFixedDepth* functor = new PhotometricBAErrorFixedDepth(patch, target_frame, px_x, px_y, height, width, idepth,
                                                                                fx, fy, cx, cy, exp_time_h, exp_time_t);
        return new ceres::AutoDiffCostFunction<PhotometricBAErrorFixedDepth, ceres::DYNAMIC, 3, 4, 3, 4, 1, 1, 1, 1>(functor, size);
    }

    const std::vector<double> *patch; // bundle patch (template in host keyframe)
    double px_x, px_y; // Pixel coordinates of point in host frame
    int height, width; // height and width of the target image
    double idepth; //fixed epth for point in residual
    double fx, fy, cx, cy; // Intrinsics
    double exp_time_h, exp_time_t; // Image exposure time h=host, t=target (1.0 by default)
    std::unique_ptr< ceres::Grid2D<double, 1> > target_grid; // 2D grid for the frame
    std::unique_ptr< ceres::BiCubicInterpolator< ceres::Grid2D<double, 1> > > target_interp; // Grid interpolation

};

} //bundles namespace
} // end namespace

#endif // _EDS_BUNDLES_PHOTOMETRIC_BA_ERROR_HPP_