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

#ifndef _EDS_TRACKER_HPP_
#define _EDS_TRACKER_HPP_

#include <eds/tracking/Types.hpp>
#include <eds/tracking/Config.hpp>
#include <eds/tracking/KeyFrame.hpp>
#include <eds/tracking/EventFrame.hpp>
#include <memory>
#include <vector>
#include <chrono>

namespace eds { namespace tracking{

enum LOSS_PARAM_METHOD{CONSTANT, MAD, STD};

class Tracker
{
    public:
        /** Configuration **/
        ::eds::tracking::Config config;

    private:
        /** Pointer to KeyFrame **/
        std::shared_ptr<eds::tracking::KeyFrame> kf;

        /** Optimization parameters **/
        Eigen::Vector3d px;
        Eigen::Quaterniond qx;
        Eigen::Matrix<double, 6, 1> vx;

        /** Status Information **/
        eds::tracking::TrackerInfo info;

        /** Vector of the last N poses **/
        std::vector<eds::SE3> poses;

        /** Squared Norm Mean Flow **/
        double squared_norm_flow;

    public:
        /** @brief Default constructor */
        Tracker(std::shared_ptr<eds::tracking::KeyFrame> kf, const eds::tracking::Config &config);
        
        /** @brief Default constructor */
        Tracker(const eds::tracking::Config &config);

        void reset(std::shared_ptr<eds::tracking::KeyFrame> kf, const Eigen::Vector3d &px, const Eigen::Quaterniond &qx, const bool &keep_velo = true);

        void reset(std::shared_ptr<eds::tracking::KeyFrame> kf, const Eigen::Vector3d &px, const Eigen::Quaterniond &qx, const base::Vector6d &velo);

        void set(const base::Transform3d &T_kf_ef);

        void optimize(const int &id, const std::vector<double> *event_frame, ::base::Transform3d &T_kf_ef,
                    const Eigen::Vector3d &px, const Eigen::Quaterniond &qx, 
                    const eds::tracking::LOSS_PARAM_METHOD loss_param_method);

        void optimize(const int &id, const std::vector<double> *event_frame, ::base::Transform3d &T_kf_ef,
                    const Eigen::Matrix<double, 6, 1> &vx, const eds::tracking::LOSS_PARAM_METHOD loss_param_method);

        bool optimize(const int &id, const std::vector<double> *event_frame, ::base::Transform3d &T_kf_ef,
                    const eds::tracking::LOSS_PARAM_METHOD loss_param_method = eds::tracking::LOSS_PARAM_METHOD::MAD);

        ::base::Transform3d getTransform();

        ::base::Transform3d getTransform(bool &result);

        Eigen::Matrix<double, 6, 1>& getVelocity();

        const Eigen::Vector3d linearVelocity();

        const Eigen::Vector3d angularVelocity();

        std::vector<double> getLossParams(eds::tracking::LOSS_PARAM_METHOD method=CONSTANT);

        /** Get warpped active points coordinates (point in event frame) **/
        std::vector<cv::Point2d> getCoord(const bool &delete_out_point = false);

        void trackPoints(const cv::Mat &event_frame, const uint16_t &patch_radius = 7);

        void trackPointsPyr(const cv::Mat &event_frame, const size_t num_level = 3);

        std::vector<cv::Point2d> trackPointsAlongEpiline(const cv::Mat &event_frame, const uint16_t &patch_radius = 7,
                            const int &border_type = cv::BORDER_DEFAULT, const uint8_t &border_value = 255);

        cv::Mat getEMatrix();

        cv::Mat getFMatrix();

        ::eds::tracking::TrackerInfo getInfo();

        bool getFilteredPose(eds::SE3 &pose, const size_t &mean_filter_size = 3);

        bool needNewKeyframe(const double &weight_factor = 0.03);
};

} //tracking namespace
} // end namespace

#endif // _EDS_TRACKER_HPP_
