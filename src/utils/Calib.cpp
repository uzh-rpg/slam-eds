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

#include "Calib.hpp"

#include <opencv2/core/eigen.hpp>

namespace eds { namespace calib {

void CameraInfo::toDSOFormat(const std::string &filename)
{
    double coeff_sum = 0.00; for (uint i=0; i<D.size(); i++) coeff_sum+=D[i];
    std::ofstream myfile; myfile.open (filename.c_str());

    /** Write intrinsics **/
    if ((distortion_model.compare("equidistant") != 0) && (coeff_sum != 0.0))
        myfile<<std::setprecision(17)<<"RadTan "<<intrinsics[0]<<" "<<intrinsics[1]<<" "<<intrinsics[2]<<" "<<intrinsics[3];
    else if ((distortion_model.compare("equidistant") == 0) && (coeff_sum != 0.0))
        myfile<<std::setprecision(17)<<"EquiDistant "<<intrinsics[0]<<" "<<intrinsics[1]<<" "<<intrinsics[2]<<" "<<intrinsics[3];
    else
        myfile<<std::setprecision(17)<<"Pinhole "<<intrinsics[0]<<" "<<intrinsics[1]<<" "<<intrinsics[2]<<" "<<intrinsics[3]<<" 0\n";

    /** Write distortion **/
    if (coeff_sum != 0.00)
    {
        for (uint i=0; i<D.size(); ++i)
            myfile<<std::setprecision(17)<<" "<<D[i];
        myfile<<"\n";
    }

    /** Write in resolution **/
    myfile<<width<<" "<<height<<"\n";

    /** Write intrinsics rectified **/
    if (P.size() == 12)
        myfile<<std::setprecision(17)<<P[0]/out_width<<" "<<P[5]/out_height<<" "<<P[2]/out_width<<" "<<P[6]/out_height<<" 0\n";
    else if (coeff_sum != 0.0) 
        myfile<<std::setprecision(17)<<intrinsics[0]/out_width<<" "<<intrinsics[1]/out_height<<" "<<intrinsics[2]/out_width<<" "<<intrinsics[3]/out_height<<" 0\n";
    else
        myfile<<"crop\n";

    /** Write out resolution **/
    myfile<<out_width<<" "<<out_height<<"\n";

    myfile.close();
}

::eds::calib::CameraInfo readCameraCalib(YAML::Node cam_calib)
{
    ::eds::calib::CameraInfo cam_info;

    cam_info.width = cam_calib["resolution"][0].as<uint16_t>();
    cam_info.height = cam_calib["resolution"][1].as<uint16_t>();
    cam_info.distortion_model = cam_calib["distortion_model"].as<std::string>();
    cam_info.D = cam_calib["distortion_coeffs"].as<std::vector<double>>();
    cam_info.intrinsics = cam_calib["intrinsics"].as<std::vector<double>>();
    if (cam_calib["flip"])
    {
        cam_info.flip = cam_calib["flip"].as<bool>();
    }
    else
        cam_info.flip  = false;

    /** T cam imu extrinsincs **/
    for (int row=0; row<4; ++row)
    {
        for (int col=0; col<4; ++col)
        {
            if (cam_calib["T_cam_imu"])
                cam_info.T_cam_imu.push_back(cam_calib["T_cam_imu"][row][col].as<double>());
        }
    }

    /** Projection matrix **/
    for (int row=0; row<3; ++row)
    {
        for (int col=0; col<4; ++col)
        {
            if (cam_calib["P"])
                cam_info.P.push_back(cam_calib["P"][row][col].as<double>());
        }
    }

    /** Rectification matrix **/
    for (int row=0; row<3; ++row)
    {
        for (int col=0; col<3; ++col)
        {
            if (cam_calib["R"])
                cam_info.R.push_back(cam_calib["R"][row][col].as<double>());
        }
    }

    /** Out resolution size **/
    if (cam_calib["resolution_out"])
    {
        cam_info.out_width = cam_calib["resolution_out"][0].as<uint16_t>();
        cam_info.out_height = cam_calib["resolution_out"][1].as<uint16_t>();
    }
    else
    {
        cam_info.out_width = cam_info.width;
        cam_info.out_height = cam_info.height;
    }

    return cam_info;
}

::eds::calib::DualCamera readDualCalibration(YAML::Node cam_calib)
{
    ::eds::calib::DualCamera calib;

    auto read_extrinsics = [](YAML::Node &node)
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Zero();
        for (int row=0; row<3; ++row)
        {
            for (int col=0; col<4; ++col)
            {
                T(row,col) = node["T_cn_cnm1"][row][col].as<double>();
            }
        }
        T(3,3) = 1.0;
        base::Transform3d trans; trans.matrix() = T; 
        //std::cout<<trans.matrix()<<std::endl;
        ::eds::calib::Baseline baseline;
        baseline.rotation = base::Quaterniond(trans.rotation());
        baseline.translation = trans.translation();
        return baseline;
    };

    YAML::Node cam0 = cam_calib["cam0"];
    calib.cam0 = ::eds::calib::readCameraCalib(cam0);//rgb camera
    if (cam_calib["cam1"])
    {
        YAML::Node cam1 = cam_calib["cam1"];
        calib.cam1 = ::eds::calib::readCameraCalib(cam1);// event camera

        if (cam1["T_cn_cnm1"])
        {
            calib.extrinsics = read_extrinsics(cam1);
        }
    }
    else
    {
        calib.cam1 = ::eds::calib::readCameraCalib(cam0);//event camera is the same as rgb camera
        calib.extrinsics.rotation = base::Quaterniond::Identity();
        calib.extrinsics.translation = base::Vector3d::Zero();
    }

    return calib;
}

::eds::calib::Camera setNewCamera(const ::eds::calib::Camera &cam0, const ::eds::calib::Camera &cam1)
{
    return ::eds::calib::setNewCamera(cam0, cam1, cam0.size);
}

::eds::calib::Camera setNewCamera(const ::eds::calib::Camera &cam0, const ::eds::calib::Camera &cam1, const cv::Size out_size)
{
    ::eds::calib::Camera newcam;


    newcam.size = cv::Size(cam0.size);
    newcam.out_size = cv::Size(out_size);
    newcam.K = cam0.K.clone();
    newcam.R = cam1.R.clone();
    return newcam;
}

void getMapping(::eds::calib::Camera &cam0, ::eds::calib::Camera &cam1, ::eds::calib::Camera &newcam)
{
    if (cam0.distortion_model.compare("equidistant") != 0 && cam1.distortion_model.compare("equidistant") != 0)
    {

        cv::initUndistortRectifyMap(cam0.K, cam0.D,
                                cv::Mat(), newcam.K * newcam.R * cam0.R.t(),
                                newcam.size, CV_32FC1,
                                cam0.mapx, cam0.mapy);

        cv::initUndistortRectifyMap(cam1.K, cam1.D,
                                cv::Mat(), newcam.K * newcam.R * cam1.R.t()/*cv::Mat()*/,
                                newcam.size, CV_32FC1,
                                cam1.mapx, cam1.mapy);
    }
    else
    {
        cv::fisheye::initUndistortRectifyMap(cam0.K, cam0.D,
                                cv::Mat(), newcam.K * newcam.R * cam0.R.t(),
                                newcam.size, CV_32FC1,
                                cam0.mapx, cam0.mapy);

        cv::fisheye::initUndistortRectifyMap(cam1.K, cam1.D,
                                cv::Mat(), newcam.K * newcam.R * cam1.R.t()/*cv::Mat()*/,
                                newcam.size, CV_32FC1,
                                cam1.mapx, cam1.mapy);
    }
    std::cout<<"** newcam.K * newcam.R * cam0.R.t:\n"<<newcam.K * newcam.R * cam0.R.t()<<std::endl;
    std::cout<<"** newcam.K * newcam.R * cam1.R.t:\n"<< newcam.K * newcam.R * cam1.R.t()<<std::endl;
}

void getEventUndistCoord (const ::eds::calib::Camera &event_camera, const ::eds::calib::Camera &new_camera, std::vector<cv::Point2d> &undist_coord)
{
    std::vector<cv::Point2d> coord;
    for (int x=0; x<event_camera.size.width; ++x)
    {
        for (int y=0; y<event_camera.size.height; ++y)
        {
            /** Grid coordinates **/
            coord.push_back(cv::Point2d(x, y));
        }
    }

    cv::undistortPoints(coord, undist_coord, event_camera.K, event_camera.D, new_camera.R * event_camera.R.t(), new_camera.K);
}


Camera::Camera(const ::eds::calib::CameraInfo &cam_info, const base::Quaterniond &rotation)
{
    size.height = cam_info.height;
    size.width = cam_info.width;

    K = cv::Mat_<double>::eye(3, 3);
    K.at<double>(0,0) = cam_info.intrinsics[0];
    K.at<double>(1,1) = cam_info.intrinsics[1];
    K.at<double>(0,2) = cam_info.intrinsics[2];
    K.at<double>(1,2) = cam_info.intrinsics[3];
    distortion_model = cam_info.distortion_model;

    D = cv::Mat_<double>::zeros(4, 1);
    for (size_t i=0; i<cam_info.D.size(); ++i)
    {
        D.at<double>(i, 0) = cam_info.D[i];
    }

    R = cv::Mat_<double>::eye(3, 3);//Identity
    base::Matrix3d R_eigen= rotation.toRotationMatrix();
    cv::eigen2cv(R_eigen, R);
}

void Camera::toDSOFormat(const std::string &filename)
{
    double coeff_sum = 0.00; for (int i=0; i<D.cols*D.rows; i++) coeff_sum+=D.at<double>(i);
    std::ofstream myfile; myfile.open (filename.c_str());

    /** Write intrinsics **/
    if ((distortion_model.compare("equidistant") != 0) && (coeff_sum != 0.0))
        myfile<<std::setprecision(17)<<"RadTan "<<K.at<double>(0,0)<<" "<<K.at<double>(1,1)<<" "<<K.at<double>(0,2)<<" "<<K.at<double>(1,2)<<"\n";
    else if ((distortion_model.compare("equidistant") == 0) && (coeff_sum != 0.0))
        myfile<<std::setprecision(17)<<"Equidistant "<<K.at<double>(0,0)<<" "<<K.at<double>(1,1)<<" "<<K.at<double>(0,2)<<" "<<K.at<double>(1,2)<<"\n";
    else
        myfile<<std::setprecision(17)<<"Pinhole "<<K.at<double>(0,0)<<" "<<K.at<double>(1,1)<<" "<<K.at<double>(0,2)<<" "<<K.at<double>(1,2)<<" 0\n";

    /** Write distortion **/
    if (coeff_sum != 0.00)
    {
        for (int i=0; i<D.cols*D.rows; ++i)
            myfile<<std::setprecision(17)<<" "<<D.at<double>(i);
        myfile<<"\n";
    }

    /** Write in resolution **/
    myfile<<size.width<<" "<<size.height<<"\n";

    /** Write intrinsics rectified **/
    myfile<<"crop\n";

    /** Write out resolution **/
    myfile<<out_size.width<<" "<<out_size.height<<"\n";

    myfile.close();
}

}} //end namespace eds::utils