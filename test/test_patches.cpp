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
#include <thread>
#include <iostream>
using namespace eds;


BOOST_AUTO_TEST_CASE(test_split)
{
    BOOST_TEST_MESSAGE("###### TEST PATCHES ######");
    uint16_t height = 100; uint16_t width = 100;
    uint16_t patch_radius = 7;
    cv::Mat img = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));

    for (int row=20; row<81; ++row)
    {
        for (int col=20; col<81; ++col)
        {
            img.at<uchar>(row,col) = 255.0;
        }
    }
    BOOST_TEST_MESSAGE("Image size: "<<img.size());
    cv::imshow("Display window", img);
    cv::waitKey(0); // Wait for a keystroke in the window

    cv::Mat img_border;
    cv::copyMakeBorder(img, img_border, patch_radius, patch_radius, patch_radius, patch_radius, cv::BORDER_CONSTANT, cv::Scalar(255));
    BOOST_TEST_MESSAGE("Image with border size: "<<img_border.size());
    cv::imshow("Display window", img_border);
    cv::waitKey(0); // Wait for a keystroke in the window

    std::vector<cv::Point2d> coord;
    coord.push_back(cv::Point2d(0, 0));
    coord.push_back(cv::Point2d(0, 99));
    coord.push_back(cv::Point2d(99, 0));
    coord.push_back(cv::Point2d(99, 99));
    coord.push_back(cv::Point2d(20, 20));
    coord.push_back(cv::Point2d(20, 80));
    coord.push_back(cv::Point2d(80, 20));
    coord.push_back(cv::Point2d(80, 80));
    BOOST_TEST_MESSAGE("coord size: "<<coord.size());

    /** Create the patches **/
    std::vector<cv::Mat> patches;
    ::eds::utils::splitImageInPatches(img, coord, patches, patch_radius, cv::BORDER_CONSTANT, 255.0);

    for (auto it=patches.begin(); it!=patches.end(); ++it)
    {
        cv::imshow("Pacth window", *it);
        cv::waitKey(0); // Wait for a keystroke in the window
    }

    /** Verify patches **/
    for (auto it : patches)
    {
        BOOST_CHECK(cv::countNonZero(it - it) == 0);
    }
}

BOOST_AUTO_TEST_CASE(test_bundles_patches_uchar)
{
    BOOST_TEST_MESSAGE("###### TEST BUNDLES PATCHES UCHAR ######");
    uint16_t height = 100; uint16_t width = 100;
    uint16_t patch_radius = 2;
    cv::Mat img = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));

    for (int row=20; row<81; ++row)
    {
        for (int col=20; col<81; ++col)
        {
            img.at<uchar>(row,col) = 255.0;
        }
    }

    std::vector<cv::Point2d> coord;
    coord.push_back(cv::Point2d(0, 0));
    coord.push_back(cv::Point2d(0, 99));
    coord.push_back(cv::Point2d(99, 0));
    coord.push_back(cv::Point2d(99, 99));
    coord.push_back(cv::Point2d(20, 20));
    coord.push_back(cv::Point2d(20, 80));
    coord.push_back(cv::Point2d(80, 20));
    coord.push_back(cv::Point2d(80, 80));
 
    /** Create the patches **/
    std::vector<cv::Mat> patches;
    ::eds::utils::splitImageInPatches(img, coord, patches, patch_radius, cv::BORDER_CONSTANT, 255.0);

    /** Create the bundles patches **/
    std::vector< std::vector<uchar> > bundles_patches;
    ::eds::utils::computeBundlePatches(patches, bundles_patches);

    cv::imshow("Patch N.0", patches[0]);
    cv::waitKey(0); // Wait for a keystroke in the window

    cv::Mat bundle_patch_0 (1, bundles_patches[0].size(), CV_8UC1, cv::Scalar(0));
    BOOST_CHECK_EQUAL(bundle_patch_0.cols, 8);
    for (size_t i=0; i<bundles_patches[0].size(); ++i)
    {
        BOOST_TEST_MESSAGE("patch 0["<<i<<"]: "<<static_cast<int>(bundles_patches[0][i]));
        bundle_patch_0.at<uchar>(0, i) = bundles_patches[0][i];
    }
    cv::imshow("Patch DSO N.0", bundle_patch_0);
    cv::waitKey(0); // Wait for a keystroke in the window

    cv::imshow("Patch N.4", patches[4]);
    cv::waitKey(0); // Wait for a keystroke in the window

    cv::Mat bundle_patch_4 (1, bundles_patches[4].size(), CV_8UC1, cv::Scalar(0));
    BOOST_CHECK_EQUAL(bundle_patch_4.cols, 8);
    for (size_t i=0; i<bundles_patches[4].size(); ++i)
    {
        BOOST_TEST_MESSAGE("patch 4["<<i<<"]: "<<static_cast<int>(bundles_patches[4][i]));
        bundle_patch_4.at<uchar>(0, i) = bundles_patches[4][i];
    }
    cv::imshow("Patch DSO N.4", bundle_patch_4);
    cv::waitKey(0); // Wait for a keystroke in the window
}

BOOST_AUTO_TEST_CASE(test_bundles_patches_double)
{
    BOOST_TEST_MESSAGE("###### TEST BUNDLES PATCHES DOUBLE ######");
    uint16_t height = 100; uint16_t width = 100;
    uint16_t patch_radius = 2;
    cv::Mat img = cv::Mat(height, width, CV_64FC1, cv::Scalar(0));

    for (int row=20; row<81; ++row)
    {
        for (int col=20; col<81; ++col)
        {
            img.at<double>(row,col) = 1.0;
        }
    }

    std::vector<cv::Point2d> coord;
    coord.push_back(cv::Point2d(0, 0));
    coord.push_back(cv::Point2d(0, 99));
    coord.push_back(cv::Point2d(99, 0));
    coord.push_back(cv::Point2d(99, 99));
    coord.push_back(cv::Point2d(20, 20));
    coord.push_back(cv::Point2d(20, 80));
    coord.push_back(cv::Point2d(80, 20));
    coord.push_back(cv::Point2d(80, 80));
 
    /** Create the patches **/
    std::vector<cv::Mat> patches;
    ::eds::utils::splitImageInPatches(img, coord, patches, patch_radius, cv::BORDER_CONSTANT, 1.0);

    /** Create the bundles patches **/
    std::vector< std::vector<double> > bundles_patches;
    ::eds::utils::computeBundlePatches(patches, bundles_patches);

    cv::imshow("Patch N.0", patches[0]);
    cv::waitKey(0); // Wait for a keystroke in the window

    cv::Mat bundle_patch_0 (1, bundles_patches[0].size(), CV_64FC1, cv::Scalar(0));
    BOOST_CHECK_EQUAL(bundle_patch_0.cols, 8);
    for (size_t i=0; i<bundles_patches[0].size(); ++i)
    {
        BOOST_TEST_MESSAGE("patch 0["<<i<<"]: "<<bundles_patches[0][i]);
        bundle_patch_0.at<double>(0, i) = bundles_patches[0][i];
    }
    cv::imshow("Patch DSO N.0", bundle_patch_0);
    cv::waitKey(0); // Wait for a keystroke in the window

    cv::imshow("Patch N.4", patches[4]);
    cv::waitKey(0); // Wait for a keystroke in the window

    cv::Mat bundle_patch_4 (1, bundles_patches[4].size(), CV_64FC1, cv::Scalar(0));
    BOOST_CHECK_EQUAL(bundle_patch_4.cols, 8);
    for (size_t i=0; i<bundles_patches[4].size(); ++i)
    {
        BOOST_TEST_MESSAGE("patch 4["<<i<<"]: "<<bundles_patches[4][i]);
        bundle_patch_4.at<double>(0, i) = bundles_patches[4][i];
    }
    cv::imshow("Patch DSO N.4", bundle_patch_4);
    cv::waitKey(0); // Wait for a keystroke in the window
}

BOOST_AUTO_TEST_CASE(test_pyramid_patches)
{
    BOOST_TEST_MESSAGE("###### TEST PYRAMID PATCHES ######");
    uint16_t height = 100; uint16_t width = 100;
    uint16_t patch_radius = 7;
    cv::Mat img = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));

    for (int row=20; row<81; ++row)
    {
        for (int col=20; col<81; ++col)
        {
            img.at<uchar>(row,col) = 255.0;
        }
    }

    std::vector<cv::Point2d> coord;
    coord.push_back(cv::Point2d(0, 0));
    coord.push_back(cv::Point2d(0, 99));
    coord.push_back(cv::Point2d(99, 0));
    coord.push_back(cv::Point2d(99, 99));
    coord.push_back(cv::Point2d(20, 20));
    coord.push_back(cv::Point2d(20, 80));
    coord.push_back(cv::Point2d(80, 20));
    coord.push_back(cv::Point2d(80, 80));
 
    /** Create the patches **/
    std::vector<cv::Mat> patches;
    ::eds::utils::splitImageInPatches(img, coord, patches, patch_radius, cv::BORDER_CONSTANT, 255.0);

    /** Create the pyramids **/
    for (auto it: patches)
    {
        cv::Size p_size = it.size();
        std::vector<cv::Mat> pyr_patches; eds::utils::pyramidPatches(it, pyr_patches);
        for (size_t i=0; i<pyr_patches.size(); ++i)
        {
            double scale = std::pow(2.0, static_cast<double>(i));
            cv::Mat &p = pyr_patches[i];
            std::cout<<"level["<<i<<"] scale: "<<scale<<" patch size: "<<p.size()<<std::endl;
            BOOST_CHECK_EQUAL(p.cols, p_size.width/(int)scale);
            BOOST_CHECK_EQUAL(p.rows, p_size.height/(int)scale);
        }
    }

    /** Visualize for patch 4 **/
    cv::Mat &patch_4 = patches[4];
    std::vector<cv::Mat> pyr_patches_4; eds::utils::pyramidPatches(patch_4, pyr_patches_4);

    for (size_t i=0; i<pyr_patches_4.size(); ++i)
    {
        int rows = pyr_patches_4[i].rows;
        int cols = pyr_patches_4[i].cols;
        cv::imshow("Patch N.4 level: " + std::to_string(i)+
                    " "+std::to_string(rows)+"x"+std::to_string(cols),pyr_patches_4[i]);
        cv::waitKey(0); // Wait for a keystroke in the window
    }
}