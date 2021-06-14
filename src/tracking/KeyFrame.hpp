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

#ifndef _EDS_KEY_FRAME_HPP_
#define _EDS_KEY_FRAME_HPP_

#include <eds/utils/Utils.hpp>
#include <eds/utils/KDTree.hpp>
#include <eds/mapping/Types.hpp>
#include <eds/mapping/DepthPoints.hpp>

#include <eds/tracking/Config.hpp>

namespace eds {
namespace tracking{

    struct KFPointIterators
    {
        std::vector<cv::Point2d>::iterator coord;
        std::vector<cv::Point2d>::iterator norm_coord;
        std::vector<cv::Point2d>::iterator grad;
        std::vector<cv::Mat>::iterator patches;
        std::vector< std::vector<double> >::iterator bundle_patches;
        std::vector<double>::iterator residuals;
        std::vector<double>::iterator weights;
        std::vector<Eigen::Vector2d>::iterator tracks;
        std::vector<Eigen::Vector2d>::iterator flow;
        ::eds::mapping::DepthPoints::iterator inv_depth;
    };
    
    enum CANDIDATE_POINT_METHOD{MAX, MEDIAN};
    enum MAP_COLOR_MODE{BLACK, RED, IMG, EVENTS};

    class KeyFrame
    {
        public:
            static constexpr float log_eps = 0.2;//1e-06;
            static constexpr float adaptive_patch_factor = 0.01631/100;
            static constexpr float adaptive_width_patch_factor = 0.08334;
            static constexpr float adaptive_height_patch_factor = 0.11112;
        public:
            /* unique id **/
            uint64_t idx;
            /** Time stamp **/
            base::Time time;
            /** Undistortion Maps **/
            cv::Mat mapx, mapy;
            /** Distortion model **/
            std::string distortion_model;
            /** Intrisic and rectification matrices **/
            cv::Mat K, D, K_ref, R_rect;
            /** Rescale out img order: (W x H) like in cv::Size**/
            std::array<double, 2> out_scale;
            /** Percent of points to select in the image **/
            double percent_points;
            /** Init number of points selected according to the percent **/
            unsigned int num_points;
            /** Image of intensities, cv::Mat and vector format **/
            cv::Mat img; std::vector<double> img_data; cv::Mat uint8_img;
            /** Image of log intensities, img gradient and magnitude **/
            std::vector<cv::Mat> log_img, img_grad, mag;
            /** Events Coordinates, normalize coord and grad (x,y) of the points **/
            std::vector<cv::Point2d> coord, norm_coord, grad;
            /** Gradient frame in std vector format **/
            std::vector<double> grad_frame;
            /** Point patches **/
            std::vector<cv::Mat> patches;
            /** Point patches (for bundles) using DSO pattern **/
            std::vector< std::vector<double> > bundle_patches;
            /** Tracking residuals **/
            std::vector<double> residuals;
            /** Point weights for tracking **/
            std::vector<double> weights;
            /** Point tracks and optical flow **/
            std::vector<Eigen::Vector2d> tracks, flow;
            /** Key frame pose **/
            ::base::Affine3d T_w_kf;
            /** Inverse depth points **/
            eds::mapping::DepthPoints inv_depth;


        public:
            /** @brief Default constructor **/
            KeyFrame(const ::eds::calib::Camera &cam, const ::eds::calib::Camera &newcam,  const std::string &distortion_model="radtan");

            KeyFrame(const uint64_t &idx, const ::base::Time &time, cv::Mat &img, ::eds::mapping::IDepthMap2d &depthmap,
                    const ::eds::calib::CameraInfo &cam_info, const ::eds::mapping::Config &map_info, const float &percent_points,
                    const ::base::Affine3d &T=::base::Affine3d::Identity(), const cv::Size &out_size = cv::Size(0, 0));

            KeyFrame(const uint64_t &idx, const ::base::Time &time, cv::Mat &img, ::eds::mapping::IDepthMap2d &depthmap,
                    cv::Mat &K, cv::Mat &D, cv::Mat &R_rect, cv::Mat &P,
                    const std::string &distortion_model="radtan",
                    const CANDIDATE_POINT_METHOD points_selection_method = MEDIAN,
                    const double &min_depth=1.0, const double &max_depth=2.0, const double &convergence_sigma2_thresh=10,
                    const float &percent_points = 0.0, const ::base::Affine3d &T=::base::Affine3d::Identity(),
                    const cv::Size &out_size = cv::Size(0, 0));

            /** @brief Create new Keyframe image **/
            void create(const uint64_t &idx, const ::base::Time &time, cv::Mat &img, ::eds::mapping::IDepthMap2d &depthmap,
                    const ::eds::mapping::Config &map_info, const float &percent_points,
                    const ::base::Affine3d &T=::base::Affine3d::Identity(), const cv::Size &out_size = cv::Size(0, 0));

            void create(const uint64_t &idx, const ::base::Time &time, cv::Mat &img, ::eds::mapping::IDepthMap2d &depthmap,
                    const CANDIDATE_POINT_METHOD points_selection_method = MEDIAN,
                    const double &min_depth=1.0, const double &max_depth=2.0, const double &convergence_sigma2_thresh=10,
                    const float &percent_points = 0.0, const ::base::Affine3d &T=::base::Affine3d::Identity(),
                    const cv::Size &out_size = cv::Size(0, 0));

            /** @brief Create new Keyframe image when using external coordinates selector and inverse depth **/
            void create(const uint64_t &idx, const ::base::Time &time, cv::Mat &img, const std::vector<cv::Point2d> &coord,
                        ::eds::mapping::IDepthMap2d &depthmap, const ::base::Affine3d &T=::base::Affine3d::Identity(), const cv::Size &out_size = cv::Size(0, 0));

            /** @brief Create new Keyframe image when using external depth map in the exact coordinates points **/
            void create(const uint64_t &idx, const ::base::Time &time, cv::Mat &img, ::eds::mapping::IDepthMap2d &depthmap,
                    const ::base::Affine3d &T=::base::Affine3d::Identity(), const cv::Size &out_size = cv::Size(0, 0));

            void clear();

            void insert(const uint64_t &idx, const ::base::Time &time, cv::Mat &img, ::eds::mapping::IDepthMap2d &depthmap,
                    const ::eds::mapping::Config &map_info, const float &percent_points = 0.0, const ::base::Affine3d &T=::base::Affine3d::Identity());

            void candidatePoints(std::vector<cv::Point2d> &coord, const cv::Size &patch_size = cv::Size(20, 20),
                    CANDIDATE_POINT_METHOD method = MAX, const int &num_points = 5000, uint8_t level=0);

            void pointsRefinement(const cv::Mat &event_frame, const double &event_diff = 1.0, const uint16_t &patch_radius=11,
                            const int &border_type = cv::BORDER_DEFAULT, const uint8_t &border_value = 255);

            bool initialStructure(const cv::Mat &input, Eigen::Matrix3d &Rotation, Eigen::Vector3d &translation,
                                std::vector<cv::Vec3d> &lines, const float &ratio_thresh = 0.9f);

            void trackPoints(const cv::Mat &img, std::vector<cv::Point2d> &track_coord, const uint8_t &patch_size=15);

            void trackPoints(const cv::Mat &img, const int &border_type = cv::BORDER_DEFAULT, const uint8_t &border_value = 255);

            /** Delete point information by index
             * return: iterators to the next element **/
            KFPointIterators erasePoint (const int &idx);

            cv::Mat viz(const cv::Mat &img, bool color=false);

            void setDepthMap(::eds::mapping::IDepthMap2d &depthmap, const ::eds::mapping::Config &map_info);

            void setDepthMap(::eds::mapping::IDepthMap2d &depthmap, const ::eds::mapping::Config &map_info, const std::array<double, 2> &scale={1.0, 1.0});

            void setDepthMap(::eds::mapping::IDepthMap2d &depthmap, const double &min_depth, const double &max_depth,
                            const double &convergence_sigma2_thresh, const std::array<double, 2> &scale={1.0, 1.0});

            void setPose(const ::base::Transform3d& pose);

            ::base::Transform3d& getPose();

            ::base::Matrix4d getPoseMatrix();
            
            std::pair<Eigen::Vector3d, Eigen::Quaterniond> getTransQuater();

            std::vector<base::Point> getDepthMap();

            cv::Mat getGradientMagnitude(const std::string &method="nn", const float s=0.5);

            cv::Mat getGradientMagnitude(const std::vector<cv::Point2d> &coord, const std::string &method="nn", const float s=0.5);

            cv::Mat getGradient_x(const std::string &method="nn", const float s=0.5);

            cv::Mat getGradient_x(const std::vector<cv::Point2d> &coord, const std::string &method="nn", const float s=0.5);

            cv::Mat getGradient_y(const std::string &method="nn", const float s=0.5);

            cv::Mat getGradient_y(const std::vector<cv::Point2d> &coord, const std::string &method="nn", const float s=0.5);

            std::vector<double> getSparseModel(const Eigen::Vector3d &vx, const Eigen::Vector3d &wx);

            std::vector<double> getSparseModel(const std::vector<cv::Point2d> &coord, const Eigen::Vector3d &vx, const Eigen::Vector3d &wx);

            cv::Mat getModel(const Eigen::Vector3d &vx, const Eigen::Vector3d &wx, const std::string &method="nn", const float &s=0.5);

            cv::Mat getModel(const std::vector<cv::Point2d> &coord, const Eigen::Vector3d &vx, const Eigen::Vector3d &wx, const std::string &method="nn", const float &s=0.5);

            base::samples::Pointcloud getMap(const std::vector<double> &model, const MAP_COLOR_MODE &color_mode=MAP_COLOR_MODE::RED);

            base::samples::Pointcloud getMap(const std::vector<double> &idp, const std::vector<double> &model, const MAP_COLOR_MODE &color_mode=MAP_COLOR_MODE::RED);

            cv::Mat idepthmapViz(const std::string &method="nn", const float s=0.5);

            cv::Mat idepthmapViz(const std::vector<cv::Point2d> &coord, const std::vector<double> &idp, const std::string &method="nn", const float s=0.5);

            cv::Mat residualsViz();

            cv::Mat eventsOnKeyFrameViz(const cv::Mat &event_frame);

            cv::Mat weightsViz(double &min, double &max);

            cv::Mat weightsViz(const std::vector<cv::Point2d> &coord, const std::vector<double> &weights, double &min, double &max);

            void meanResiduals(double &mean, double &st_dev);

            void medianResiduals(double &median, double &third_q);

            bool needNewKF(const double &percent_thr = 0.1);

            bool needNewKFImageCriteria(const double &percent);

            void cleanPoints(const double &w_norm_thr = 0.2);
    };

} //tracking namespace
} // end namespace

#endif // _EDS_KEY_FRAME_HPP_