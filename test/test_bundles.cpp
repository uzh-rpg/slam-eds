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

::eds::bundles::Config readBundlesConfig(YAML::Node config)
{
    ::eds::bundles::Config bundles_config;

    /** Number of points to optimize within the current window **/
    bundles_config.percent_points = config["percent_points"].as<double>();
    /** Percent of visual point to seletc the kf to marginalize **/
    bundles_config.percent_marginalize_vis = config["percent_marginalize_vis"].as<double>();
    /** BA Windows size **/
    bundles_config.window_size = config["window_size"].as<uint16_t>();
    /** Config for tracker type (only ceres) **/
    bundles_config.type = config["type"].as<std::string>();
 
    /** Config the loss **/
    YAML::Node bundles_loss = config["loss_function"];
    std::string loss_name = bundles_loss["type"].as<std::string>();
    bundles_config.loss_type = ::eds::bundles::selectLoss(loss_name);
    bundles_config.loss_params = bundles_loss["param"].as<std::vector<double>>();
   
    /** Config for ceres options **/
    YAML::Node tracker_options = config["options"];
    bundles_config.options.linear_solver_type = ::eds::bundles::selectSolver(tracker_options["solver_type"].as<std::string>());
    bundles_config.options.num_threads = tracker_options["num_threads"].as<int>();
    bundles_config.options.max_num_iterations = tracker_options["max_num_iterations"].as<int>();
    bundles_config.options.function_tolerance = tracker_options["function_tolerance"].as<double>();
    bundles_config.options.minimizer_progress_to_stdout = tracker_options["minimizer_progress_to_stdout"].as<bool>();

    return bundles_config;
}

::eds::mapping::Config readMappingConfig(YAML::Node config)
{
    ::eds::mapping::Config mapping_config;
    
    mapping_config.min_depth = config["min_depth"].as<double>();
    if (mapping_config.min_depth < 0) mapping_config.min_depth= 1e0-6;
    mapping_config.max_depth = config["max_depth"].as<double>();
    if (mapping_config.max_depth < 0) mapping_config.max_depth= 1e0-6;

    return mapping_config;
}

BOOST_AUTO_TEST_CASE(test_bundles)
{
    BOOST_TEST_MESSAGE("###### TEST BUNDLES ######");
    uint64_t idx = 100;
    ::base::Time ts;
    cv::Mat img = cv::imread("test/data/frame_00000000.png", cv::IMREAD_GRAYSCALE);
    eds::mapping::IDepthMap2d depthmap;

    base::Affine3d tf = base::Affine3d::Identity();
    cv::Mat K, D, R_rect, P;
    K = cv::Mat_<double>::eye(3, 3);
    D = cv::Mat_<double>::zeros(4, 1);
    readCalibration(K, D, "test/data/calib.txt");
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Img size "<<img.size());
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Img type: "<<eds::utils::type2str(img.type()));

    /** Read bundles configuration **/
    std::string test_yaml_fn = "test/data/eds.yaml";
    YAML::Node config = YAML::LoadFile(test_yaml_fn);
    eds::bundles::Config bundles_config = readBundlesConfig(config["bundles"]);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Percent marginalize kf based on visual point: "<<bundles_config.percent_marginalize_vis);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Percent of points to optimize: "<<bundles_config.percent_points);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] BA max_num_iterations: "<<bundles_config.options.max_num_iterations);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Function tolerance: "<<bundles_config.options.function_tolerance);

    /** Read mapping configuration **/
    eds::mapping::Config mapping_config = readMappingConfig(config["mapping"]);

    /** Read tracking configuration **/
    eds::tracking::Config tracking_info = ::eds::tracking::readTrackingConfig(config["tracker"]);

    /** Read mapping configuration **/
    eds::mapping::Config mapping_info = ::eds::mapping::readMappingConfig(config["mapping"]);

    /** Camera info **/
    ::eds::calib::CameraInfo cam_info;
    cam_info.height = img.rows; cam_info.width = img.cols;
    cam_info.D = D;
    cam_info.intrinsics = std::vector<double> {K.at<double>(0,0), K.at<double>(1,1),
                                K.at<double>(0,2), K.at<double>(1,2)};

    /** Global Map **/
    std::shared_ptr<::mapping::GlobalMap> global_map =
            std::make_shared<eds::mapping::GlobalMap> (mapping_config, cam_info, bundles_config, tracking_info.percent_points);

    /* Create the Bundle Adjustment **/
    eds::bundles::BundleAdjustment bundles(bundles_config, mapping_config.min_depth, mapping_config.max_depth);

    uint64_t kf_to_marginalize;
    bool success = bundles.optimize(global_map, kf_to_marginalize);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] PBA Optimize duration: "<<bundles.getInfo().meas_time_ms<<"[ms]");
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] PBA Optimize duration: "<<bundles.getInfo().time_seconds<<"[s]");
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] PBA Optimize iterations: "<<bundles.getInfo().num_iterations);

    BOOST_CHECK_EQUAL(kf_to_marginalize, ::eds::mapping::NONE_KF);
    BOOST_CHECK_EQUAL(success, false);
 
    /** Create the First KeyFrame **/
    std::shared_ptr<eds::tracking::KeyFrame> kf =
            std::make_shared<eds::tracking::KeyFrame>(idx, ts, img, depthmap, K, D, R_rect, P,
                                                    "radtan", eds::tracking::MAX,
                                                    mapping_info.min_depth, mapping_info.max_depth,
                                                    mapping_info.convergence_sigma2_thresh, 0.5, tf);

    /** Insert First Keyframe in Global Map **/
    global_map->insert(kf);

    /** Create the Second KeyFrame **/
    kf.reset();
    idx++; ts = ts + ::base::Time::fromSeconds(1.0);
    img = cv::imread("test/data/frame_00000000.png", cv::IMREAD_GRAYSCALE);
    kf = std::make_shared<eds::tracking::KeyFrame>(idx, ts, img, depthmap, K, D, R_rect, P,
                                                    "radtan", eds::tracking::MAX,
                                                    mapping_info.min_depth, mapping_info.max_depth,
                                                    mapping_info.convergence_sigma2_thresh, 0.5, tf);

    /** Insert Second Keyframe in Global Map **/
    global_map->insert(kf);

    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Global map with: "<<global_map->camera_kfs.size()<<" frames");
    BOOST_CHECK_EQUAL(global_map->camera_kfs.size(), 2);

    /** Optimize **/
    success = bundles.optimize(global_map, kf_to_marginalize);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Frame to marginalize "<<kf_to_marginalize);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] PBA Optimize duration: "<<bundles.getInfo().meas_time_ms<<"[ms]");
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] PBA Optimize duration: "<<bundles.getInfo().time_seconds<<"[s]");
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] PBA Optimize iterations: "<<bundles.getInfo().num_iterations);
    BOOST_CHECK_EQUAL(kf_to_marginalize, ::eds::mapping::NONE_KF);
    BOOST_CHECK_EQUAL(success, true);

    ::base::Transform3d T = global_map->getKFTransform(idx);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Transform["<<idx<<"]\n"<<T.matrix());

    BOOST_CHECK_EQUAL(global_map->size(), 2);
    success = global_map->removeKeyFrame(idx); if(success)idx--;
    BOOST_CHECK_EQUAL(global_map->size(), 1);
    BOOST_CHECK_EQUAL(success, true);
    success = global_map->removeKeyFrame(idx); if(success)idx--;
    BOOST_CHECK_EQUAL(global_map->size(), 0);
    BOOST_CHECK_EQUAL(success, true);
    success = global_map->removeKeyFrame(idx); if(success)idx--;
    BOOST_CHECK_EQUAL(global_map->size(), 0);
    BOOST_CHECK_EQUAL(success, false);

    T = global_map->getKFTransform(idx);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Transform["<<idx<<"]\n"<<T.matrix());
}
