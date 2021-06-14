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
#include <eds/EDS.h>

#include <yaml-cpp/yaml.h>
#include <iostream>

BOOST_AUTO_TEST_CASE(test_read_yaml)
{
    std::string test_yaml_fn = "test/data/eds.yaml";
    YAML::Node config = YAML::LoadFile(test_yaml_fn);
    YAML::Node options = config["tracker"]["options"];

    //std::cout<<config["tracker"]<<std::endl;

    ::ceres::Solver::Options ceres_options;
    ceres_options.num_threads = options["num_threads"].as<int>();
    BOOST_CHECK_EQUAL(ceres_options.num_threads, 4);

}


BOOST_AUTO_TEST_CASE(test_std_move)
{
    std::vector<int> a = {1, 2, 3, 4, 5}; // a has size 5
    auto a_copy = a;                      // copy a. now we have two vectors of size 5
    auto a_move = std::move(a);           // *move* a into a_move a becomes invalid


    BOOST_TEST_MESSAGE("a_move[2]: "<<a_move[2]);
    BOOST_CHECK_EQUAL(a_copy[0], a_move[0]);
    BOOST_CHECK_EQUAL(a_move.size(), 5);
    BOOST_CHECK_EQUAL(a.size(), 0);

    std::shared_ptr< std::vector<int> > p_a_move = std::make_shared< std::vector<int> >(std::move(a_move));
    BOOST_TEST_MESSAGE("p_a_move[2]: "<<(*p_a_move)[2]);
    BOOST_CHECK_EQUAL(a_copy[0], (*p_a_move)[0]);
    BOOST_CHECK_EQUAL(a_move.size(), 0);
}

BOOST_AUTO_TEST_CASE(test_camera_calib)
{
    std::string test_yaml_fn = "test/data/camera_calib.yaml";
    YAML::Node calib = YAML::LoadFile(test_yaml_fn);
    YAML::Node cam_calib = calib["cam0"];
    ::eds::calib::CameraInfo cam_info;

    cam_info.width = cam_calib["resolution"][0].as<uint16_t>();
    cam_info.height = cam_calib["resolution"][1].as<uint16_t>();
    cam_info.distortion_model = cam_calib["distortion_model"].as<std::string>();
    cam_info.D = cam_calib["distortion_coeffs"].as<std::vector<double>>();
    cam_info.intrinsics = cam_calib["intrinsics"].as<std::vector<double>>();

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

    /** Rectfication matrix **/
    for (int row=0; row<3; ++row)
    {
        for (int col=0; col<3; ++col)
        {
            if (cam_calib["R"])
                cam_info.R.push_back(cam_calib["R"][row][col].as<double>());
        }
    }


    BOOST_CHECK_EQUAL(cam_info.width, 346);
    BOOST_CHECK_EQUAL(cam_info.height, 260);
    BOOST_CHECK_EQUAL(cam_info.D.size(), 4);
    BOOST_CHECK_EQUAL(cam_info.intrinsics.size(), 4);
    BOOST_CHECK_EQUAL(cam_info.T_cam_imu.size(), 16);
    BOOST_CHECK_EQUAL(cam_info.R.size(), 9);
    BOOST_CHECK_EQUAL(cam_info.P.size(), 12);
}