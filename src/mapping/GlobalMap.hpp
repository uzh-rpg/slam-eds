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

#ifndef _EDS_MAPPING_GLOBAL_MAP_HPP_
#define _EDS_MAPPING_GLOBAL_MAP_HPP_

#include <eds/utils/Utils.hpp>
#include <eds/mapping/Config.hpp>
#include <eds/bundles/Config.hpp>
#include <eds/mapping/Types.hpp>
#include <eds/tracking/KeyFrame.hpp>

#include <base/samples/RigidBodyState.hpp>

#include <memory>

namespace eds { namespace mapping {

typedef std::map< uint64_t, eds::mapping::KeyFrameInfo> CameraKFs;
typedef std::map< uint64_t, std::vector<eds::mapping::PointInfo> > PointKFs;

constexpr uint64_t NONE_KF = std::numeric_limits<std::uint64_t>::max();

enum POINTS_SELECT_METHOD{NONE, RANDOM, RESIDUALS, GRADIENT, SPARSE_GRADIENT};

class GlobalMap
{
    private:
        /** Number of point from tracking **/
        double num_points_in_tracking;
 
        /** Bundles config: window size and
        the percent of tracking points to use
        for bundles (config values) **/
        ::eds::bundles::Config bundles_config;

    public:
        /** Configuration **/
        ::eds::mapping::Config config;

        /** Active points for bundles adjustment **/
        int total_active_points; //config_bundles.percent_points * num_points_in_traking

        /** Depends on the current
         * window size: total_active_points * window_size * */
        int active_points_per_kf;

        /** dictionary of keyframes **/
        CameraKFs camera_kfs;

        /** points at keyframe idx 
         * the first active_points_per_kf
         * are randomly selected **/
        PointKFs point_kfs;

        /** Last two inserted KeyFrames **/
        uint64_t last_kf_id, prev_last_kf_id;

    public:

        /** @brief Default constructor **/
        GlobalMap(const ::eds::mapping::Config &config,
                  const ::eds::calib::CameraInfo &cam_info,
                  const ::eds::bundles::Config &bundles_config,
                  const double &percent_tracking_points);

        /** @brief insert a new kf in the global map **/
        void insert(std::shared_ptr<eds::tracking::KeyFrame> kf,
                    const ::eds::mapping::POINTS_SELECT_METHOD &points_select_method = POINTS_SELECT_METHOD::NONE);

        /** @brief get homogeneous transformation of KF id **/
        ::base::Transform3d getKFTransform(const uint64_t &kf_idx);

        /** @brief check if KF is in map **/
        bool isKFinMap(const uint64_t &kf_idx);

        /** @brief remove KF with ID **/
        bool removeKeyFrame(const uint64_t &kf_id);

        /** @brief Return the current window size **/
        size_t size(){return this->camera_kfs.size();};

        cv::Mat vizKeyFrame(const uint64_t &kf_id);

        cv::Mat residualsViz(const uint64_t &kf_id);

        cv::Mat vizMosaic(const int &bundles_window_size);

        std::vector<::eds::mapping::Point3d> getMap(const bool &only_active_points=true, const bool &remove_outliers = false);

        void getMap(base::samples::Pointcloud &points_map, const bool &only_active_points=true, const bool &remove_outliers = false);

        void getIDepthMap(const uint64_t &kf_id, ::eds::mapping::IDepthMap2d &depthmap, const bool &only_active_points=true, const bool &remove_outliers=false);

        void getIDepthMap(const ::base::Transform3d &T_kf_w, const std::vector<double> &intrinsics, const cv::Size &img_size, ::eds::mapping::IDepthMap2d &depthmap, const bool &only_active_points=true, const bool &remove_outliers=false);

        void getKFPoses(std::vector<::base::samples::RigidBodyState> &poses);

        std::vector<cv::Point3d> projectMapOnKF(const std::vector<::eds::mapping::Point3d> &map,
                                                const uint64_t &kf_id);

        void setNumPointsInTracking(const uint16_t &height, const uint16_t &width, const double &percent_tracking_points, const size_t &window_size=7);

        void updateNumberActivePoints(const size_t &window_size);

        void orderPoints(std::vector<PointInfo> &points, const ::eds::mapping::POINTS_SELECT_METHOD &method=POINTS_SELECT_METHOD::NONE,
                        const int &num_points = 0, const int &block_size = 10);

        void depthCompletion(std::vector<PointInfo> &points);

        void outliersRemoval(const double &threshold, const bool &only_active_points=true);

        void eraseNaNResiduals(std::vector<eds::mapping::PointInfo> &points, const bool &only_active_points);

        void cleanMap(const uint8_t &num_points_pixel, const bool &only_active_points=true);

};

} // mapping namespace
} // end namespace
#endif // _EDS_MAPPING_GLOBAL_MAP_HPP_
