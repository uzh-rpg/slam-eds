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

#include <memory>
#include <iostream>

BOOST_AUTO_TEST_CASE(test_read_dual_setup)
{
    BOOST_TEST_MESSAGE("###### TEST CALIBRATION ######");
    std::string calib_file = "/home/javi/rock/dev/bundles/eds/config/data/dual_setup/camera_info.yaml";
    BOOST_TEST_MESSAGE("Calibration file to read: "<<calib_file);

    YAML::Node node_config = YAML::LoadFile(calib_file);
    ::eds::calib::DualCamera cam_calib = eds::calib::readDualCalibration(node_config);

    /** Set cameras calibration **/
    std::shared_ptr<::eds::calib::Camera> cam0 = std::make_shared<eds::calib::Camera>(cam_calib.cam0);
    std::shared_ptr<::eds::calib::Camera> cam1 = std::make_shared<eds::calib::Camera>(cam_calib.cam1, cam_calib.extrinsics.rotation);

    std::cout<<"** Configuration CAM0: **"<<std::endl;
    std::cout<<"Model: "<<cam_calib.cam0.distortion_model<<std::endl;
    std::cout<<"Size: "<<cam0->size<<std::endl;
    std::cout<<"K:\n"<<cam0->K<<std::endl;
    std::cout<<"D:\n"<<cam0->D<<std::endl;
    std::cout<<"R:\n"<<cam0->R<<std::endl;

    std::cout<<"** Configuration CAM1: **"<<std::endl;
    std::cout<<"Model: "<<cam_calib.cam1.distortion_model<<std::endl;
    std::cout<<"Size: "<<cam1->size<<std::endl;
    std::cout<<"K:\n"<<cam1->K<<std::endl;
    std::cout<<"D:\n"<<cam1->D<<std::endl;
    std::cout<<"R:\n"<<cam1->R<<std::endl;

    std::shared_ptr<eds::calib::Camera> newcam = std::make_shared<::eds::calib::Camera>(eds::calib::setNewCamera(*(cam0), *(cam1)));

    std::cout<<"** Configuration NEWCAM: **"<<std::endl;
    std::cout<<"Size: "<<newcam->size<<std::endl;
    std::cout<<"K:\n"<<newcam->K<<std::endl;
    std::cout<<"D:\n"<<newcam->D<<std::endl;
    std::cout<<"R:\n"<<newcam->R<<std::endl;

    eds::calib::getMapping(*cam0, *cam1, *newcam);
    std::cout<<"cam0.mapx: "<<cam0->mapx.rows<<" x "<<cam0->mapx.cols<<std::endl;
    std::cout<<"cam0.mapy: "<<cam0->mapy.rows<<" x "<<cam0->mapy.cols<<std::endl;
    std::cout<<"cam1.mapx: "<<cam1->mapx.rows<<" x "<<cam1->mapx.cols<<std::endl;
    std::cout<<"cam1.mapy: "<<cam1->mapy.rows<<" x "<<cam1->mapy.cols<<std::endl;

    std::vector<cv::Point2d> undist_coord;
    eds::calib::getEventUndistCoord(*(cam1), *(newcam), undist_coord);
    std::cout<<"undist_coord.size: "<<undist_coord.size()<<std::endl;
    BOOST_CHECK_EQUAL(undist_coord.size(), cam1->size.height * cam1->size.width);
}


