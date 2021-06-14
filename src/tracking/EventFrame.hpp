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

#ifndef _EDS_EVENT_FRAME_HPP_
#define _EDS_EVENT_FRAME_HPP_

#include <eds/utils/Utils.hpp>
#include <eds/utils/Colormap.hpp>

#include <eds/tracking/Config.hpp>

namespace eds {
namespace tracking {
    class EventFrame
    {
        public:
            static constexpr float log_eps = 1e-06;
        public:
            /* unique id **/
            uint64_t idx;
            /** Desired Image dimension **/
            uint16_t height, width;
            /** Time stamps **/
            base::Time first_time, last_time, time, delta_time;
            /** Undistortion Maps (inverse and forward mapping) **/
            cv::Mat mapx, fwd_mapx, mapy, fwd_mapy;
            /** Rescale out img order: (W x H) like in cv::Size**/
            std::array<double, 2> out_scale;
            /** Events Coordinates and normalize coord **/
            std::vector<cv::Point2d> coord, undist_coord;
            /** Events polarities **/
            std::vector<int8_t> pol;
            /** Distortion model information **/
            std::string distortion_model;
            /** Intrisic and rectification matrices **/
            cv::Mat K, D, K_ref, R_rect;
            /** Event Frame pose **/
            ::base::Affine3d T_w_ef;
            /** Event frame (integration of events) no normalized **/
            std::vector<cv::Mat> frame;
            /** Normalized event frame in std vector for optimization **/
            std::vector< std::vector<double> > event_frame; // event_frame = frame / norm
            /** Norm of the event frame **/
            std::vector<double> norm;

        public:
            /** @brief Default constructor **/
            EventFrame(const ::eds::calib::Camera &cam, const ::eds::calib::Camera &newcam,  const std::string &distortion_model="radtan");

            EventFrame(const uint64_t &idx, const std::vector<base::samples::Event> &events,
                    const ::eds::calib::CameraInfo &cam_info, const int &num_levels = 1,
                    const ::base::Affine3d &T=::base::Affine3d::Identity(),
                    const cv::Size &out_size = cv::Size(0, 0));

            /** @brief Default constructor **/
            EventFrame(const uint64_t &idx, const std::vector<base::samples::Event> &events, const uint16_t height, const uint16_t width,
                        cv::Mat &K, cv::Mat &D, cv::Mat &R_rect, cv::Mat &P, const std::string distortion_model="radtan", const int &num_levels = 1,
                        const ::base::Affine3d &T=::base::Affine3d::Identity(), const cv::Size &out_size = cv::Size(0, 0));

            /** @brief Insert new Eventframe **/
            void create(const uint64_t &idx, const std::vector<base::samples::Event> &events,
                    const ::eds::calib::CameraInfo &cam_info, const int &num_levels = 1,
                    const ::base::Affine3d &T=::base::Affine3d::Identity(),
                    const cv::Size &out_size = cv::Size(0, 0));

            void create(const uint64_t &idx, const std::vector<base::samples::Event> &events,
                        const uint16_t height, const uint16_t width, const int &num_levels = 1,
                        const ::base::Affine3d &T=::base::Affine3d::Identity(), const cv::Size &out_size = cv::Size(0, 0));

            void clear();

            cv::Mat viz(size_t id=0, bool color = false);

            cv::Mat getEventFrame(const size_t &id=0);

            cv::Mat getEventFrameViz(const size_t &id=0, bool color = false);

            void setPose(const ::base::Transform3d& pose);

            ::base::Transform3d& getPose();

            ::base::Matrix4d getPoseMatrix();

            std::pair<Eigen::Vector3d, Eigen::Quaterniond> getTransQuater();

            cv::Mat epilinesViz(const std::vector<cv::Point2d> &coord, const cv::Mat &F, const size_t &skip_amount = 10.0);

            cv::Mat pyramidViz(const bool &color = false);

    };

} //tracking namespace
} // end namespace

#endif // _EDS_EVENT_FRAME_HPP_