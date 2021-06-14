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

#ifndef _EDS_CALIB_HPP_
#define _EDS_CALIB_HPP_

/** Yaml **/
#include <yaml-cpp/yaml.h>

/** Opencv  **/
#include <opencv2/opencv.hpp>

/** Base types **/
#include <base/Time.hpp>
#include <base/Eigen.hpp>

#include <vector>

namespace eds { namespace calib {

    struct CameraInfo
    {
        uint16_t height;// input image height
        uint16_t width;// input image width
        uint16_t out_height;// output image height
        uint16_t out_width;// output image width
        std::string distortion_model; // distortion model name
        std::vector<double> D; // distortion coefficients
        std::vector<double> intrinsics; // fx, fy, cx, cy intrinsics
        std::vector<double> R; //3x3 row-major rectification matrix
        /* this matrix specifies the intrinsic (camera) matrix
        of the processed (rectified) image. That is, the left 3x3 portion
        is the normal camera intrinsic matrix for the rectified image. **/
        std::vector<double> P; //3x4 row major 
        std::vector<double> T_cam_imu; //4x4 row-major p_cam = T_cam_imu * p_imu
        bool flip; //vertical flip when the image is comming from a beamsplitter

        void toDSOFormat(const std::string &filename="/tmp/dso_camera.txt");
    };

    struct Baseline
    {
        base::Vector3d translation;
        base::Quaterniond rotation; 
    };

    struct DualCamera
    {
        ::eds::calib::CameraInfo cam0; //rgb camera
        ::eds::calib::CameraInfo cam1; //event camera
        ::eds::calib::Baseline extrinsics; // Camera extrinsics
    };

    /** Camera calibration **/
    struct Camera
    {
        cv::Size size;
        cv::Size out_size;
        cv::Mat K, D, R;
        cv::Mat mapx, mapy;
        std::string distortion_model; // distortion model name

        Camera(){};
        Camera(const ::eds::calib::CameraInfo &cam_info, const base::Quaterniond &rotation = base::Quaterniond::Identity());
        void toDSOFormat(const std::string &filename="/tmp/dso_camera.txt");
        void undistort(const cv::Mat &input, cv::Mat &output)
        {
            remap(input, output, mapx, mapy, cv::INTER_CUBIC);
        }
        double fx(){return K.at<double>(0,0);}
        double fy(){return K.at<double>(1,1);}
        double cx(){return K.at<double>(0,2);}
        double cy(){return K.at<double>(1,2);}
        std::vector<double> intrinsics(){return {fx(), fy(), cx(), cy()};}
    };

    ::eds::calib::CameraInfo readCameraCalib(YAML::Node cam_calib);
    ::eds::calib::DualCamera readDualCalibration(YAML::Node cam_calib);
    ::eds::calib::Camera setNewCamera(const ::eds::calib::Camera &cam0, const ::eds::calib::Camera &cam1);
    ::eds::calib::Camera setNewCamera(const ::eds::calib::Camera &cam0, const ::eds::calib::Camera &cam1, const cv::Size out_size);
    void getMapping(::eds::calib::Camera &cam0, ::eds::calib::Camera &cam1, ::eds::calib::Camera &newcam);
    void getEventUndistCoord (const ::eds::calib::Camera &event_camera, const ::eds::calib::Camera &new_camera, std::vector<cv::Point2d> &undist_coord);

} //calib namespace
} // end namespace

#endif // _EDS_CALIB_HPP_