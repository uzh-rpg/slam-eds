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

#ifndef _EDS_PHOTOMETRIC_ERROR_HPP_
#define _EDS_PHOTOMETRIC_ERROR_HPP_

#include <eds/mapping/Types.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/cubic_interpolation.h>
#include <memory>

namespace eds { namespace tracking {

struct UnitNormVectorAddition
{

    template <typename Scalar>
    bool operator()(const Scalar* x, const Scalar* delta, Scalar* x_plus_delta) const {

        Scalar sum = Scalar(0);
        for (size_t i = 0; i < 6; i++) {
            Scalar s = x[i] + delta[i];
            sum += s * s;
            x_plus_delta[i] = s;
        }


        sum = Scalar(1)/ceres::sqrt(sum);

        for (size_t i = 0; i < 6; i++) {
            x_plus_delta[i] *= sum;
        }

        return true;
    }
};
 
struct PhotometricErrorNC
{
    PhotometricErrorNC(const std::vector<cv::Point2d> *grad,
                     const std::vector<cv::Point2d> *norm_coord,
                     const std::vector<double> *idp,
                     const std::vector<double> *weights,
                     const std::vector<double> *event_frame,
                     const int &height, const int &width,
                     const double &fx, const double &fy,
                     const double &cx, const double &cy,
                     const int &start_element, const int &num_elements)
    :height(height), width(width), fx(fx), fy(fy), cx(cx), cy(cy)
    {
        /** Sanity checks **/
        assert(grad->size() == norm_coord->size());
        assert(norm_coord->size() == idp->size());
        assert(idp->size() == weights->size());
        assert(event_frame->size() == (size_t)(height * width));

        this->start = start_element;
        this->n_points = num_elements;

        /** Get the parameters **/
        this->grad = grad;
        this->norm_coord = norm_coord;
        this->idp = idp;
        this->weights = weights;
        this->event_frame = event_frame;

        //std::cout<<"[PHOTO_ERROR] height: "<<height<<" width: "<<width<<std::endl;
        //std::cout<<"[PHOTO_ERROR] fx: "<<fx<<" fy: "<<fy<<" cx: "<<cx<<" cy: "<<cy<<std::endl;
        //std::cout<<"[PHOTO_ERROR] num points: "<<this->n_points<<std::endl;
        //std::cout<<"[PHOTO_ERROR] grad ["<<std::addressof(*(this->grad))<<"] size:"<<this->grad->size()<<std::endl;
        //std::cout<<"[PHOTO_ERROR] norm_coord ["<<std::addressof(*(this->norm_coord))<<"] size:"<<this->norm_coord->size()<<std::endl;
        //std::cout<<"[PHOTO_ERROR] idp ["<<std::addressof(*(this->idp))<<"] size: "<<this->idp->size()<<std::endl;
        //std::cout<<"[PHOTO_ERROR] argument event_frame ["<<std::addressof(*(event_frame))<<"] size: "<<event_frame->size()<<std::endl;
        //std::cout<<"[PHOTO_ERROR] event_frame ["<<std::addressof(*(this->event_frame))<<"] size: "<<this->event_frame->size()<<std::endl;

        /** Generate the 3D points **/
        kp.clear();
        int idx = this->start;
        for (int i=0; i<n_points; ++i)
        {
            ::eds::mapping::Point3d p;
            p[2] = 1.0/((*idp)[idx]+this->eps);
            p[0] = (*norm_coord)[idx].x*p[2];
            p[1] = (*norm_coord)[idx].y*p[2];

            kp.push_back(p);
            idx++;
        }
        assert(kp.size() == n_points);

        /** Create the grid for the event frame interpolate **/
        event_grid.reset(new ceres::Grid2D<double, 1> (this->event_frame->data(), 0, height, 0, width));
        event_grid_interp.reset(new ceres::BiCubicInterpolator< ceres::Grid2D<double, 1> > (*event_grid));
    }

    template <typename T> inline
    void compute_flow(const T &xp, const T &yp, const T* vx, const T &idp, T *result) const
    {
        result[0] = (-idp*vx[0]) + (xp*idp*vx[2])
                        + (xp*yp*vx[3]) - (T(1.0)+ceres::pow(xp, 2))*vx[4] + (yp*vx[5]);

        result[1] = (-idp*vx[1]) + (yp*idp*vx[2])
                        + (T(1.0)+ceres::pow(yp, 2))*vx[3] - (xp*yp*vx[4]) - (xp*vx[5]);
    }

    template <typename T>
    bool operator()(const T* px, const T* qx, const T* vx, T* residual) const
    {
        /** Get current pose and quaternion **/
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_x(px);
        Eigen::Map<const Eigen::Quaternion<T>> q_x(qx);

        /** Compute the model for the active points **/
        T model_norm_sq(1e-03);
        int idx = this->start;
        for (int i=0; i<this->n_points; i++)
        {
            /** Flow in the key frame **/
            Eigen::Matrix<T, 2, 1> flow; compute_flow<T>(T((*norm_coord)[idx].x), T((*norm_coord)[idx].y), vx, T((*idp)[idx]), flow.data()); // 2 x 1 flow
            //flow[2] = T(0.0);

            /** Contra rotate the flow: flow in keyframe **/
            //flow = q_x.toRotationMatrix().transpose() * flow;

            /** Model **/
            residual[i] = -((*grad)[idx].x * flow[0] + (*grad)[idx].y * flow[1]);

            /** Get the normalization for later **/
            model_norm_sq += ceres::pow(residual[i], 2);
            idx++;
        }

        T meas_norm_sq(1e-03);
        std::vector<T> event_points_brightness;
        for (int i=0; i<this->n_points; i++)
        {
            /** Get the point in the keyframe **/
            Eigen::Matrix<T, 3, 1> point;
            point[0] = T(kp[i][0]); point[1] = T(kp[i][1]); point[2] = T(kp[i][2]);
            //std::cout<<"["<<i<<"] point: "<<point[0]<<" "<<point[1]<<" "<<point[2]<<std::endl;

            /** Rotate and translate the point **/
            Eigen::Matrix<T, 3, 1> p;
            p = q_x.toRotationMatrix() * point + p_x;
            //std::cout<<"p: "<<p[0]<<" "<<p[1]<<" "<<p[2]<<std::endl;

            /** Project the point into the event frame **/
            T xp = T(fx) * (p[0]/p[2]) + T(cx);
            T yp = T(fy) * (p[1]/p[2]) + T(cy);

            /** Brightness change from the events **/
            T event_brightness;
            event_grid_interp->Evaluate(yp, xp, &event_brightness);
            event_points_brightness.push_back(event_brightness);

            /** Get the normalization for later **/
            meas_norm_sq += ceres::pow(event_brightness, 2);
        }

        /** Compute the residuals: model - event frame (sparse points) **/
        T model_norm = ceres::sqrt(model_norm_sq);
        T meas_norm = ceres::sqrt(meas_norm_sq);
        idx = this->start;
        for (int i=0; i<this->n_points; i++)
        {
            residual[i] = T((*weights)[idx]) * ((residual[i]/model_norm) - (event_points_brightness[i]/meas_norm));
            idx++;
        }

        //std::cout << "[CERES] Press Enter to Continue";
        //std::cin.ignore();
        //std::cout<<"*****"<<std::endl;
        return true;
    }


    /** NOTE: this class requires the event frame NOT normalized **/
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const std::vector<cv::Point2d> *grad,
                                       const std::vector<cv::Point2d> *norm_coord,
                                       const std::vector<double> *idp,
                                       const std::vector<double> *weights,
                                       const std::vector<double> *event_frame,
                                       const int &height, const int &width,
                                       const double &fx, const double &fy,
                                       const double &cx, const double &cy,
                                       const int &start_element, const int &num_elements)
    {
        PhotometricErrorNC* functor = new PhotometricErrorNC(grad, norm_coord, idp, weights, event_frame, height, width, fx, fy, cx, cy, start_element, num_elements); 
        return new ceres::AutoDiffCostFunction<PhotometricErrorNC, ceres::DYNAMIC, 3, 4, 6>(functor, num_elements);
    }

    static constexpr double eps = 1e-05;

    int start; // start of points
    int n_points; // number of active points
    int height, width; // height and width of the image
    double fx, fy, cx, cy; // intrinsics
    std::vector< ::eds::mapping::Point3d > kp; // 3D points [x, y , z]
    const std::vector<cv::Point2d> *grad; // N x 2 img gradient [\Nabla x, \Nabla y]
    const std::vector<cv::Point2d> *norm_coord; // N x 2 normalized keyframe coord [X, Y]
    const std::vector<double> *idp; // N x 1 inverse depth
    const std::vector<double> *weights; // N x 1 weight points
    const std::vector<double> *event_frame; // H x W event frame with the brightness change
    std::unique_ptr< ceres::Grid2D<double, 1> > event_grid;
    std::unique_ptr< ceres::BiCubicInterpolator< ceres::Grid2D<double, 1> > > event_grid_interp;
};

} //tracking namespace
} // end namespace

#endif // _EDS_PHOTOMETRIC_ERROR_HPP_
