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

#include <boost/test/unit_test.hpp>
#include <eds/utils/KDTree.hpp>
#include <eds/EDS.h>

#include<iostream>
using namespace eds;

bool readCalibration(cv::Mat &K, cv::Mat &D, const std::string &filename)
{
    std::ifstream file(filename);
    std::string str;
    if (file.is_open())
    { 
        std::getline(file, str);
        std::istringstream iss(str);
        std::vector<std::string> tokens; tokens.resize(8);
        for (size_t i=0; i<tokens.size(); ++i)
        {
            std::string s;
            getline( iss, s, ' ' );
            tokens[i] = s;
            //std::cout<<s<<std::endl;
        }
        K.at<double>(0,0) = std::stod(tokens[0]);
        K.at<double>(1,1) = std::stod(tokens[1]);
        K.at<double>(0,2) = std::stod(tokens[2]);
        K.at<double>(1,2) = std::stod(tokens[3]);

        D.at<double>(0,0) = std::stod(tokens[4]);
        D.at<double>(1,0) = std::stod(tokens[5]);
        D.at<double>(2,0) = std::stod(tokens[6]);
        D.at<double>(3,0) = std::stod(tokens[7]);
        tokens.clear();
        file.close();
        return true;
    }
    return false;
}

BOOST_AUTO_TEST_CASE(test_init_3planes)
{
    BOOST_TEST_MESSAGE("###### TEST INIT 3PLANES ######");
    uint64_t idx = 100;
    ::base::Time ts;
    cv::Mat img = cv::imread("test/data/frame_00000000.png", cv::IMREAD_GRAYSCALE);
    eds::mapping::IDepthMap2d depthmap;
    base::Affine3d tf = base::Affine3d::Identity();
    cv::Mat K, D, R_rect, P;
    K = cv::Mat_<double>::eye(3, 3);
    D = cv::Mat_<double>::zeros(4, 1);
    readCalibration(K, D, "test/data/calib.txt");
    std::string test_yaml_fn = "test/data/eds.yaml";
    YAML::Node config = YAML::LoadFile(test_yaml_fn);
    ::eds::tracking::Config tracker_config;

    std::cout<<"Image size: "<<img.size()<<std::endl;
    tracking::KeyFrame kf(idx, ts, img, depthmap, K, D, R_rect, P, "radtan", 
                ::eds::tracking::MAX, 1.0 /*min_depth*/, 30.0/*max_depth*/,
                100.0 /*threshold*/, 10.0 /*percent_points */, tf);

    cv::Mat img_next = cv::imread("test/data/frame_00000036.png", cv::IMREAD_GRAYSCALE);
    std::cout<<"Image Next size: "<<img_next.size()<<std::endl;
    Eigen::Matrix3d R; Eigen::Vector3d t;
    std::vector<cv::Vec3d> lines;
    bool result = kf.initialStructure(img_next, R, t, lines);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Result: "<<result);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] R:\n"<<R);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] t:\n"<<t);
    BOOST_CHECK_EQUAL(result, true);
}

BOOST_AUTO_TEST_CASE(test_init_atrium)
{
    BOOST_TEST_MESSAGE("###### TEST INIT ATRIUM ######");
    uint64_t idx = 100;
    ::base::Time ts;
    cv::Mat img = cv::imread("test/data/atrium_00.png", cv::IMREAD_GRAYSCALE);
    eds::mapping::IDepthMap2d depthmap;
    base::Affine3d tf = base::Affine3d::Identity();
    cv::Mat K, D, R_rect, P;
    K = cv::Mat_<double>::eye(3, 3);
    D = cv::Mat_<double>::zeros(4, 1);
    readCalibration(K, D, "test/data/calib_atrium.txt");
    std::string test_yaml_fn = "test/data/eds.yaml";
    YAML::Node config = YAML::LoadFile(test_yaml_fn);
    ::eds::tracking::Config tracker_config;

    std::cout<<"Image size: "<<img.size()<<std::endl;
    tracking::KeyFrame kf(idx, ts, img, depthmap, K, D, R_rect, P, "radtan", 
                ::eds::tracking::MAX, 1.0 /*min_depth*/, 30.0/*max_depth*/,
                100.0 /*threshold*/, 10.0 /*percent_points */, tf);

    cv::Mat img_next = cv::imread("test/data/atrium_55.png", cv::IMREAD_GRAYSCALE);
    std::cout<<"Image Next size: "<<img_next.size()<<std::endl;
    Eigen::Matrix3d R; Eigen::Vector3d t;
    std::vector<cv::Vec3d> lines;
    bool result = kf.initialStructure(img_next, R, t, lines);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Result: "<<result);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] R:\n"<<R);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] t:\n"<<t);
    BOOST_CHECK_EQUAL(result, true);
}