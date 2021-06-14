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

std::vector<base::samples::Event> readEvents(const std::string &filename, const int array_size)
{
    int i = 0;
    std::string str; 
    std::vector<::base::samples::Event> events;

    std::ifstream file(filename);
    if (!file.is_open())
        return events;
    
    while (std::getline(file, str) || i < array_size)
    {
        /** Split the line **/
        std::istringstream iss(str);
        std::vector<std::string> tokens;
        std::copy(std::istream_iterator<std::string>(iss),
                    std::istream_iterator<std::string>(),
                   std:: back_inserter(tokens));

        ::base::samples::Event ev(
            std::stoi(tokens[1]), std::stoi(tokens[2]),
            ::base::Time::fromSeconds(std::stod(tokens[0])),
            (uint8_t)std::stoi(tokens[3]));

        events.push_back(ev);
        ++i;
    }
    
    file.close();

    std::cout<<"[READ_EVENTS] events.size: "<<events.size()<<std::endl;
    return events;
}

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

BOOST_AUTO_TEST_CASE(test_read_tracking_config)
{
    BOOST_TEST_MESSAGE("###### TEST TRACKING CONFIG ######");

    std::string test_yaml_fn = "test/data/eds.yaml";
    YAML::Node config = YAML::LoadFile(test_yaml_fn);
    ::eds::tracking::Config tracker_config = ::eds::tracking::readTrackingConfig(config["tracker"]);
    BOOST_CHECK(tracker_config.type.compare("ceres")==0);
    BOOST_CHECK_EQUAL(tracker_config.loss_type, ::eds::tracking::LOSS_FUNCTION::HUBER);
    BOOST_CHECK_EQUAL(tracker_config.options.num_threads, 4);
}

BOOST_AUTO_TEST_CASE(test_event_frame)
{
    BOOST_TEST_MESSAGE("###### TEST EVENTFRAME ######");
    uint64_t idx = 0;
    uint16_t height = 180;
    uint16_t width = 240;
    base::Affine3d tf = base::Affine3d::Identity();
    std::vector<::base::samples::Event> events = readEvents("test/data/events.txt", 10000);
    cv::Mat K, D, R_rect, P;
    R_rect = cv::Mat_<double>::eye(3, 3);
    K = cv::Mat_<double>::eye(3, 3);
    D = cv::Mat_<double>::zeros(4, 1);
    readCalibration(K, D, "test/data/calib.txt");

    tracking::EventFrame ef(idx, events, height, width, K, D, R_rect, P, "radtan", 1.0, tf);

    BOOST_CHECK_EQUAL(ef.height, height);
    BOOST_CHECK_EQUAL(ef.width, width);
    BOOST_TEST_MESSAGE("K:\n"<<ef.K);
    BOOST_TEST_MESSAGE("D:\n"<<ef.D);
    BOOST_TEST_MESSAGE("K_ref:\n"<<ef.K_ref);
    BOOST_TEST_MESSAGE("R_rect:\n"<<ef.R_rect);
    BOOST_CHECK((ef.frame[0].data != nullptr) && (ef.frame[0].total()>0));
    BOOST_TEST_MESSAGE("ef.frame[0].size: "<<ef.frame[0].size());
    BOOST_TEST_MESSAGE("ef.frame[0].type: "<<eds::utils::type2str(ef.frame[0].type()));
    BOOST_TEST_MESSAGE("ef.norm[0]: "<<ef.norm[0]);

    double min, max;
    cv::minMaxLoc(ef.frame[0], &min, &max);
    BOOST_TEST_MESSAGE("ef.frame[0] min: "<<min<< " max: "<<max);

    cv::Mat events_viz = ef.viz(0, true);
    BOOST_TEST_MESSAGE("events_viz.size: "<<events_viz.size());
    BOOST_TEST_MESSAGE("events_viz.type: "<<eds::utils::type2str(events_viz.type()));
    BOOST_CHECK((events_viz.data != nullptr) && (events_viz.total()>0));
    cv::imwrite("test/data/out_event_frame.png", events_viz);

    cv::Mat event_mat = ef.getEventFrame(0);
    BOOST_TEST_MESSAGE("event_mat.size: "<<event_mat.size());
    BOOST_TEST_MESSAGE("event_mat.type: "<<eds::utils::type2str(event_mat.type()));
    cv::minMaxLoc(event_mat, &min, &max);
    BOOST_TEST_MESSAGE("event_mat min: "<<min<< " max: "<<max);
    BOOST_CHECK_EQUAL(event_mat.rows, ef.frame[0].rows);
    BOOST_CHECK_EQUAL(event_mat.cols, ef.frame[0].cols);
    BOOST_CHECK_CLOSE(cv::norm(event_mat), 1.0, 1e-09);

    cv::Mat tmp = ef.frame[0] * (1.0/ef.norm[0]);
    cv::Mat diff;
    cv::subtract(tmp, event_mat, diff);
    BOOST_TEST_MESSAGE("Count Non Zeros: "<<cv::countNonZero(diff));

    for (int i=0; i<ef.frame[0].rows; ++i)
    {
        for (int j=0; j<ef.frame[0].cols; ++j)
        {
            BOOST_CHECK_EQUAL((ef.frame[0].at<double>(i, j)/ef.norm[0]), event_mat.at<double>(i, j));
            BOOST_CHECK_EQUAL((ef.frame[0].at<double>(i, j)/ef.norm[0]), ef.event_frame[0][(ef.frame[0].cols*i)+j]);
            BOOST_CHECK_CLOSE(ef.frame[0].at<double>(i, j), (ef.norm[0] * event_mat.at<double>(i, j)), 1e-09);
            BOOST_CHECK_CLOSE(ef.frame[0].at<double>(i, j), (ef.norm[0] * event_mat.at<double>(i, j)), 1e-09);
        }
    }


    cv::Mat event_frame_viz = ef.getEventFrameViz(true);
    BOOST_TEST_MESSAGE("event_frame_viz.size: "<<event_frame_viz.size());
    BOOST_TEST_MESSAGE("event_frame_viz.type: "<<eds::utils::type2str(event_frame_viz.type()));
    BOOST_CHECK((event_frame_viz.data != nullptr) && (event_frame_viz.total()>0));
    cv::imwrite("test/data/out_norm_event_frame.png", event_frame_viz);

    /** Check pose methods **/
    ::base::Transform3d T_w_ef = ef.getPose();
    BOOST_CHECK(T_w_ef.matrix().isApprox(ef.T_w_ef.matrix()));
    ::base::Matrix4d m = ef.getPoseMatrix();
    BOOST_CHECK(m.isApprox(ef.T_w_ef.matrix()));
    auto trans_quater = ef.getTransQuater();
    BOOST_CHECK(trans_quater.first.isApprox(ef.T_w_ef.translation()));
    BOOST_CHECK(trans_quater.second.isApprox(Eigen::Quaterniond(ef.T_w_ef.rotation())));
}

BOOST_AUTO_TEST_CASE(test_key_frame)
{
    BOOST_TEST_MESSAGE("###### TEST KEYFRAME ######");
    uint64_t idx = 0;
    ::base::Time ts;
    cv::Mat img = cv::imread("test/data/frame_00000000.png", cv::IMREAD_GRAYSCALE);
    eds::mapping::IDepthMap2d depthmap;
    cv::Size s = img.size();
    base::Affine3d tf = base::Affine3d::Identity();
    cv::Mat K, D, R_rect, P;
    K = cv::Mat_<double>::eye(3, 3);
    D = cv::Mat_<double>::zeros(4, 1);
    readCalibration(K, D, "test/data/calib.txt");
    std::cout<<"Image size: "<<img.size()<<std::endl;

    tracking::KeyFrame kf(idx, ts, img, depthmap, K, D, R_rect, P, "radtan", 
                ::eds::tracking::MAX, 1.0 /*min_depth*/, 2.0/*max_depth*/,
                100.0 /*threshold*/, 10.0 /*percent_points */, tf);

    BOOST_TEST_MESSAGE("K:\n"<<kf.K);
    BOOST_TEST_MESSAGE("D:\n"<<kf.D);
    BOOST_TEST_MESSAGE("K_ref:\n"<<kf.K_ref);
    BOOST_TEST_MESSAGE("R_rect:\n"<<kf.R_rect);
    BOOST_CHECK((kf.img.data != nullptr) && (kf.img.total()>0));
    BOOST_TEST_MESSAGE("kf.img type: "<<eds::utils::type2str(kf.img.type()));
    double min, max;
    cv::minMaxLoc(kf.img, &min, &max);
    BOOST_TEST_MESSAGE("kf.img min: "<<min<< " max: "<<max);
    BOOST_CHECK_EQUAL(kf.img.at<double>(0, 0), kf.img_data[0]);
    BOOST_CHECK_EQUAL(kf.img.at<double>(1, 0), kf.img_data[kf.img.cols]);
    BOOST_CHECK((kf.log_img[0].data != nullptr) && (kf.log_img[0].total()>0));
    cv::Size s_kf = kf.log_img[0].size();
    BOOST_CHECK_EQUAL(s_kf.height, s.height);
    BOOST_CHECK_EQUAL(s_kf.width, s.width);
    cv::minMaxLoc(kf.log_img[0], &min, &max);
    BOOST_TEST_MESSAGE("kf.log_img[0] min: "<<min<< " max: "<<max);
    BOOST_CHECK_EQUAL(kf.log_img[0].type(), CV_64FC1);
    cv::minMaxLoc(kf.img_grad[0], &min, &max);
    BOOST_TEST_MESSAGE("kf.img_grad min: "<<min<< " max: "<<max);
    BOOST_CHECK_EQUAL(kf.img_grad[0].type(), CV_64FC2);
    BOOST_TEST_MESSAGE("kf.grad type: "<<eds::utils::type2str(kf.img_grad[0].type()));
    cv::minMaxLoc(kf.mag[0], &min, &max);
    BOOST_TEST_MESSAGE("kf.mag[0] min: "<<min<< " max: "<<max);
    BOOST_CHECK_EQUAL(kf.mag[0].type(), CV_64FC1);
    BOOST_TEST_MESSAGE("kf.mag type: "<<eds::utils::type2str(kf.mag[0].type()));

    cv::Mat mag_viz = kf.viz(kf.mag[0], true);
    cv::imwrite("test/data/out_magnitude.png", mag_viz);

    BOOST_TEST_MESSAGE("Number of points: "<<kf.coord.size());
    BOOST_TEST_MESSAGE("Number of Gradient points: "<<kf.grad.size());
    auto it_norm = kf.norm_coord.begin();
    auto it_grad = kf.grad.begin();
    double fx, fy, cx, cy;
    fx = kf.K_ref.at<double>(0,0); fy = kf.K_ref.at<double>(1,1);
    cx = kf.K_ref.at<double>(0,2); cy = kf.K_ref.at<double>(1,2);
    std::cout<<"fx:"<<fx<<" fy:"<<fy<<" cx:"<<cx<<" cy:"<<cy<<std::endl;
    for (auto it=kf.coord.begin(); it!=kf.coord.end(); ++it, ++it_norm, ++it_grad)
    {
        //std::cout<<"p: "<<*it<<" "<<*it_norm<<" grad: "<<*it_grad<<std::endl;
        //std::cout<<"coord: "<<(it_norm->x * fx) + cx<<" "<<(it_norm->y * fy) + cy<<std::endl;
        BOOST_CHECK_CLOSE((it_norm->x * fx)+cx, it->x, 1e-06);
        BOOST_CHECK_CLOSE((it_norm->y * fy)+cy, it->y, 1e-06);
    }

    cv::Mat points_img = kf.img.clone();
    eds::utils::drawPointsOnImage(kf.coord, points_img);
    BOOST_TEST_MESSAGE("points_img type: "<<eds::utils::type2str(points_img.type()));
    BOOST_TEST_MESSAGE("points_img: "<<points_img.size());
    cv::imwrite("test/data/out_img_points.png", points_img);

    cv::Mat mag_img = kf.getGradientMagnitude();
    BOOST_TEST_MESSAGE("mag_img type: "<<eds::utils::type2str(mag_img.type()));
    cv::Mat mag_img_viz = kf.viz(mag_img, true);
    BOOST_TEST_MESSAGE("mag_img_viz type: "<<eds::utils::type2str(mag_img_viz.type()));
    cv::imwrite("test/data/out_mag_img.png", mag_img_viz);

    cv::Mat grad_x_img = kf.getGradient_x();
    BOOST_TEST_MESSAGE("grad_x_img type: "<<eds::utils::type2str(grad_x_img.type()));
    cv::Mat grad_x_img_viz = kf.viz(grad_x_img, false);
    BOOST_TEST_MESSAGE("grad_x_img_viz type: "<<eds::utils::type2str(grad_x_img_viz.type()));
    cv::imwrite("test/data/out_grad_x_img.png", grad_x_img_viz);

    cv::Mat grad_y_img = kf.getGradient_y();
    BOOST_TEST_MESSAGE("grad_y_img type: "<<eds::utils::type2str(grad_y_img.type()));
    cv::Mat grad_y_img_viz = kf.viz(grad_y_img, false);
    BOOST_TEST_MESSAGE("grad_y_img_viz type: "<<eds::utils::type2str(grad_y_img_viz.type()));
    cv::imwrite("test/data/out_grad_y_img.png", grad_y_img_viz);

    Eigen::Vector3d vx (1, 1, 1); 
    Eigen::Vector3d wx (1, 1, 1);
    cv::Mat model_img = kf.getModel(vx/vx.norm(), wx/wx.norm());
    BOOST_TEST_MESSAGE("model_img type: "<<eds::utils::type2str(model_img.type()));
    cv::Mat model_img_viz = kf.viz(model_img, false);
    BOOST_TEST_MESSAGE("model_img_viz type: "<<eds::utils::type2str(model_img_viz.type()));
    cv::imwrite("test/data/out_model_img.png", model_img_viz);

    cv::Point2d px = kf.coord[10];
    std::vector<cv::Mat> grds;
    cv::split(kf.img_grad[0], grds);
    double gradx_px = grds[0].at<double>(px);
    double grady_px = grds[1].at<double>(px);
    BOOST_TEST_MESSAGE("px x: "<<px.x<<" px y: "<<px.y);
    BOOST_TEST_MESSAGE("I["<<px<<"]: "<<kf.img.at<double>(px));
    BOOST_TEST_MESSAGE("px"<<px<<" grad_x "<<gradx_px);
    BOOST_TEST_MESSAGE("px"<<px<<" grad_y "<<grady_px);
    BOOST_TEST_MESSAGE("grad["<<kf.grad[10].x<<","<<kf.grad[10].y<<"]");
    BOOST_CHECK_EQUAL(kf.grad[10].x, gradx_px);
    BOOST_CHECK_EQUAL(kf.grad[10].y, grady_px);

    /** Check pose methods **/
    ::base::Transform3d T_w_kf = kf.getPose();
    BOOST_CHECK(T_w_kf.matrix().isApprox(kf.T_w_kf.matrix()));
    ::base::Matrix4d m = kf.getPoseMatrix();
    BOOST_CHECK(m.isApprox(kf.T_w_kf.matrix()));
    auto trans_quater = kf.getTransQuater();
    BOOST_CHECK(trans_quater.first.isApprox(kf.T_w_kf.translation()));
    BOOST_CHECK(trans_quater.second.isApprox(Eigen::Quaterniond(kf.T_w_kf.rotation())));

    /** Check deleting points **/
    size_t n_points = kf.coord.size();
    auto its = kf.erasePoint(n_points-1);
    BOOST_CHECK_EQUAL(kf.coord.size() ,n_points-1);
    BOOST_CHECK_EQUAL(kf.norm_coord.size(), n_points-1);
    BOOST_CHECK_EQUAL(kf.grad.size(), n_points-1);
    BOOST_CHECK_EQUAL(kf.inv_depth.size(), n_points-1);
    BOOST_CHECK_EQUAL(kf.patches.size(), n_points-1);

    /**By deletin the last elemet we get the iterator
    * pointing to the end **/
    BOOST_CHECK(its.coord == kf.coord.end());
    BOOST_CHECK(its.norm_coord  == kf.norm_coord.end());
    BOOST_CHECK(its.grad == kf.grad.end());
    BOOST_CHECK(its.inv_depth == kf.inv_depth.end());
    BOOST_CHECK(its.patches == kf.patches.end());
}

BOOST_AUTO_TEST_CASE(test_key_frame_patches)
{
    BOOST_TEST_MESSAGE("###### TEST KEYFRAME PATCHES ######");
    uint64_t idx = 0;
    ::base::Time ts;
    cv::Mat img = cv::imread("test/data/frame_00000000.png", cv::IMREAD_GRAYSCALE);
    eds::mapping::IDepthMap2d depthmap;
    base::Affine3d tf = base::Affine3d::Identity();
    cv::Mat K, D, R_rect, P;
    K = cv::Mat_<double>::eye(3, 3);
    D = cv::Mat_<double>::zeros(4, 1);
    readCalibration(K, D, "test/data/calib.txt");
    std::cout<<"Image size: "<<img.size()<<std::endl;

    tracking::KeyFrame kf(idx, ts, img, depthmap, K, D, R_rect, P, "radtan",
                ::eds::tracking::MAX, 1.0 /*min_depth*/, 2.0/*max_depth*/,
                100.0 /*threshold*/, 10.0 /*percent_points */, tf);

    BOOST_CHECK_EQUAL(kf.coord.size(), kf.patches.size());
    cv::Mat patch_viz = kf.viz(kf.patches[0], false);
    BOOST_TEST_MESSAGE("patch_viz type: "<<eds::utils::type2str(patch_viz.type()));
    cv::imwrite("test/data/out_patch_viz"+std::to_string(kf.coord[0].x)+"_"+std::to_string(kf.coord[0].y)+".png", patch_viz);
}

BOOST_AUTO_TEST_CASE(test_tracker)
{
    BOOST_TEST_MESSAGE("###### TEST TRACKER ######");
    uint64_t idx = 100;
    ::base::Time ts;
    cv::Mat img = cv::imread("test/data/frame_00000000.png", cv::IMREAD_GRAYSCALE);
    std::vector<::base::samples::Event> events = readEvents("test/data/events.txt", 10000);
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

    /** Percent of points to track **/
    tracker_config.percent_points = config["tracker"]["percent_points"].as<float>();
    /** Config for tracker type (only ceres) **/
    tracker_config.type = config["tracker"]["type"].as<std::string>();
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Solver: "<<tracker_config.type);
 
    /** Config the loss **/
    YAML::Node tracker_loss = config["tracker"]["loss_function"];
    std::string loss_name = tracker_loss["type"].as<std::string>();
    tracker_config.loss_type = ::eds::tracking::selectLoss(loss_name);
    tracker_config.loss_params = tracker_loss["param"].as<std::vector<double>>();
   
    /** Config for ceres options **/
    YAML::Node tracker_options = config["tracker"]["options"];
    tracker_config.options.linear_solver_type = ::eds::tracking::selectSolver(tracker_options["solver_type"].as<std::string>());
    tracker_config.options.num_threads = tracker_options["num_threads"].as<int>();
    tracker_config.options.max_num_iterations = tracker_options["max_num_iterations"].as< std::vector<int> >();
    tracker_config.options.function_tolerance = tracker_options["function_tolerance"].as<double>();
    tracker_config.options.minimizer_progress_to_stdout = tracker_options["minimizer_progress_to_stdout"].as<bool>();


    /** Config for mapping **/
    ::eds::mapping::Config mapping_config = ::eds::mapping::readMappingConfig(config["mapping"]);

    /** Create the KeyFrame **/
    std::shared_ptr<eds::tracking::KeyFrame> kf =
            std::make_shared<eds::tracking::KeyFrame>(idx, ts, img, depthmap, K, D, R_rect, P,
                                                    "radtan", eds::tracking::MAX, mapping_config.min_depth, mapping_config.max_depth,
                                                    mapping_config.convergence_sigma2_thresh, tracker_config.percent_points, tf);
    BOOST_CHECK_EQUAL(kf->mag[0].type(), CV_64FC1);
    BOOST_CHECK_EQUAL(kf->idx, idx);

    /** Create the Tracker **/
    tracking::Tracker tracker(kf, tracker_config);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] KF address: "<<std::addressof(*kf));
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] grad address: "<<std::addressof(kf->grad));

    /** Create the EventFrame **/
    tracking::EventFrame ef_1(idx, events, img.rows, img.cols, K, D, R_rect, P, "radtan", tracker_config.options.max_num_iterations.size(), tf);

    /** Optimize **/
    ::base::Transform3d T_kf_ef; T_kf_ef.setIdentity();
    tracker.optimize(0, &(ef_1.event_frame[0]), T_kf_ef, eds::tracking::LOSS_PARAM_METHOD::CONSTANT);
    //std::thread optim_thread_1(&eds::tracking::Tracker::optimize, &tracker, 0, &(ef_1.event_frame[0]), std::ref(T_kf_ef));
    //optim_thread_1.join();
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Optimize duration: "<<tracker.getInfo().meas_time_us<<"[us]");
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Optimize duration: "<<tracker.getInfo().time_seconds<<"[s]");
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Optimize iterations: "<<tracker.getInfo().num_iterations);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] T_kf_ef:\n"<<T_kf_ef.matrix());

    /** Perform Lucas-Kanade tracker **/
    tracker.trackPointsPyr(ef_1.frame[0], 6);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] KLT num points: "<< kf->coord.size());
    BOOST_CHECK_EQUAL(kf->flow.size(), kf->coord.size());
    BOOST_CHECK_EQUAL(kf->flow.size(), kf->inv_depth.size());
    BOOST_CHECK_EQUAL(kf->flow.size(), kf->residuals.size());
    BOOST_CHECK_EQUAL(kf->flow.size(), kf->tracks.size());

    ::base::Transform3d T_kf_ef_ = tracker.getTransform().inverse();
    BOOST_CHECK_CLOSE(T_kf_ef.translation()[0], T_kf_ef_.translation()[0], 1e-09);
    BOOST_CHECK_CLOSE(T_kf_ef.translation()[1], T_kf_ef_.translation()[1], 1e-09);
    BOOST_CHECK_CLOSE(T_kf_ef.translation()[2], T_kf_ef_.translation()[2], 1e-09);

    double mean, std;
    eds::utils::mean_std_vector(kf->residuals, mean, std);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] residual mean: "<<mean<<" std: "<<std);
    double median_1 = eds::utils::n_quantile_vector(kf->residuals, kf->residuals.size()/2);
    double three_quantile_1 = eds::utils::n_quantile_vector(kf->residuals, kf->residuals.size()/3);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] residual median: "<<median_1<<" 3-qth: "<<three_quantile_1);

    /** Check online computation of the loss parameter **/
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Tracker loss param: "<<tracker.config.loss_params[0]);
    BOOST_CHECK_EQUAL(tracker.config.loss_params[0], tracker_config.loss_params[0]);
 
    /** Viz the residual **/
    cv::Mat residuals_viz = kf->residualsViz();
    cv::imwrite("test/data/out_redisuals_viz.png", residuals_viz);

    /** Create the EventFrame **/
    tracking::EventFrame ef_2(idx, events, img.rows, img.cols, K, D, R_rect, P, "radtan", tracker_config.options.max_num_iterations.size(), tf);

    /** Optimize **/
    tracker.optimize(0, &(ef_2.event_frame[0]), T_kf_ef, ::eds::tracking::LOSS_PARAM_METHOD::STD);
    //std::thread optim_thread_2(&eds::tracking::Tracker::optimize, &tracker, 0, &(ef_2.event_frame[0]), std::ref(T_kf_ef));
    //optim_thread_2.join();
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Optimize duration: "<<tracker.getInfo().meas_time_us<<"[us]");
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Optimize duration: "<<tracker.getInfo().time_seconds<<"[s]");
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Optimize iterations: "<<tracker.getInfo().num_iterations);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] T_kf_ef:\n"<<T_kf_ef.matrix());

    /** Check online computation of the loss parameter **/
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Tracker loss param: "<<tracker.config.loss_params[0]);
    eds::utils::mean_std_vector(kf->residuals, mean, std);
    BOOST_CHECK_EQUAL(tracker.config.loss_params[0], 1.345*std);
 
    T_kf_ef_ = tracker.getTransform().inverse();
    BOOST_CHECK_CLOSE(T_kf_ef.translation()[0], T_kf_ef_.translation()[0], 1e-09);
    BOOST_CHECK_CLOSE(T_kf_ef.translation()[1], T_kf_ef_.translation()[1], 1e-09);
    BOOST_CHECK_CLOSE(T_kf_ef.translation()[2], T_kf_ef_.translation()[2], 1e-09);

    eds::utils::mean_std_vector(kf->residuals, mean, std);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] residual mean: "<<mean<<" std: "<<std);

    double median_2 = eds::utils::n_quantile_vector(kf->residuals, kf->residuals.size()/2);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] residual median: "<<median_2<<" std: "<<std);

    /** Viz the inverse depth on the image **/
    cv::Mat inv_depth_viz = kf->idepthmapViz();
    cv::imwrite("test/data/out_inv_depth_viz.png", inv_depth_viz);

    /** Perform Lucas-Kanade tracker **/
    uint16_t patch_radius = 30;
    tracker.trackPoints(ef_2.frame[0], patch_radius);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] KLT num points: "<< kf->coord.size());
    BOOST_CHECK_EQUAL(kf->flow.size(), kf->coord.size());
    BOOST_CHECK_EQUAL(kf->flow.size(), kf->inv_depth.size());
    BOOST_CHECK_EQUAL(kf->flow.size(), kf->residuals.size());
    BOOST_CHECK_EQUAL(kf->flow.size(), kf->tracks.size());

    /** Save optical flow image **/
    cv::Mat of_viz = ::eds::utils::flowArrowsOnImage(kf->img, kf->coord, kf->tracks);
    cv::imwrite("test/data/out_optical_flow_viz.png", of_viz);

    /** Visualize the patches for the tracking **/
    int point_to_viz = 1300;
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] VIZ Point: "<<kf->coord[point_to_viz]);
    std::vector<cv::Point2d> coord = tracker.getCoord(false);
    /** Warpped gradient images **/
    cv::Mat grad_x = kf->getGradient_x(coord, "bilinear");
    cv::Mat grad_y = kf->getGradient_y(coord, "bilinear");

    /** Gradient patches **/
    std::vector<cv::Mat> grad_patches_x, grad_patches_y;
    eds::utils::splitImageInPatches(grad_x, coord, grad_patches_x, patch_radius);
    eds::utils::splitImageInPatches(grad_y, coord, grad_patches_y, patch_radius);

    /** Event frame patches **/
    std::vector<cv::Mat> event_patches;
    eds::utils::splitImageInPatches(ef_2.frame[0], coord, event_patches, patch_radius);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] grad_patches_x: "<<eds::utils::type2str(grad_patches_x[point_to_viz].type()));
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] grad_patches_y: "<<eds::utils::type2str(grad_patches_y[point_to_viz].type()));
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] event_patches: "<<eds::utils::type2str(event_patches[point_to_viz].type()));
    cv::Mat out = kf->viz(grad_patches_x[point_to_viz]);
    cv::Mat out_color; cv::cvtColor(out, out_color, cv::COLOR_GRAY2RGB);
    out_color.at<cv::Vec3b>(cv::Point(grad_patches_x[point_to_viz].cols/2, grad_patches_x[point_to_viz].rows/2)) = cv::Vec3b(0.0, 0.0, 255.0); //red
    cv::imwrite("test/data/out_grad_patch_x.png", out_color);
    out = kf->viz(grad_patches_y[point_to_viz]);
    cv::cvtColor(out, out_color, cv::COLOR_GRAY2RGB);
    out_color.at<cv::Vec3b>(cv::Point(grad_patches_y[point_to_viz].cols/2, grad_patches_y[point_to_viz].rows/2)) = cv::Vec3b(0.0, 0.0, 255.0); //red
    cv::imwrite("test/data/out_grad_patch_y.png", out_color);
    out = kf->viz(event_patches[point_to_viz]);
    cv::cvtColor(out, out_color, cv::COLOR_GRAY2RGB);
    out_color.at<cv::Vec3b>(cv::Point(event_patches[point_to_viz].cols/2, event_patches[point_to_viz].rows/2)) = cv::Vec3b(0.0, 0.0, 255.0); //red
    cv::imwrite("test/data/out_event_patch.png", out_color);
    out = kf->viz(kf->img);
    cv::cvtColor(out, out_color, cv::COLOR_GRAY2RGB);
    out_color.at<cv::Vec3b>(kf->coord[point_to_viz]) = cv::Vec3b(0.0, 0.0, 255.0); //red
    cv::imwrite("test/data/out_img_viz_point.png", out_color);
}

BOOST_AUTO_TEST_CASE(test_resize)
{
    BOOST_TEST_MESSAGE("###### TEST RESIZE TRACKER ######");
    uint64_t idx = 100;
    ::base::Time ts;
    cv::Mat img = cv::imread("test/data/frame_00000000.png", cv::IMREAD_GRAYSCALE);
    std::vector<::base::samples::Event> events = readEvents("test/data/events.txt", 10000);
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

    /** Percent of points to track **/
    tracker_config.percent_points = config["tracker"]["percent_points"].as<float>();
    /** Config for tracker type (only ceres) **/
    tracker_config.type = config["tracker"]["type"].as<std::string>();
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Solver: "<<tracker_config.type);

    /** Config the loss **/
    YAML::Node tracker_loss = config["tracker"]["loss_function"];
    std::string loss_name = tracker_loss["type"].as<std::string>();
    tracker_config.loss_type = ::eds::tracking::selectLoss(loss_name);
    tracker_config.loss_params = tracker_loss["param"].as<std::vector<double>>();

    /** Config for ceres options **/
    YAML::Node tracker_options = config["tracker"]["options"];
    tracker_config.options.linear_solver_type = ::eds::tracking::selectSolver(tracker_options["solver_type"].as<std::string>());
    tracker_config.options.num_threads = tracker_options["num_threads"].as<int>();
    tracker_config.options.max_num_iterations = tracker_options["max_num_iterations"].as< std::vector<int> >();
    tracker_config.options.function_tolerance = tracker_options["function_tolerance"].as<double>();
    tracker_config.options.minimizer_progress_to_stdout = tracker_options["minimizer_progress_to_stdout"].as<bool>();


    /** Config for mapping **/
    ::eds::mapping::Config mapping_config = ::eds::mapping::readMappingConfig(config["mapping"]);

    /** Create the KeyFrame **/
    std::shared_ptr<eds::tracking::KeyFrame> kf =
            std::make_shared<eds::tracking::KeyFrame>(idx, ts, img, depthmap, K, D, R_rect, P,
                                                    "radtan", eds::tracking::MAX, mapping_config.min_depth, mapping_config.max_depth,
                                                    mapping_config.convergence_sigma2_thresh, tracker_config.percent_points, tf, cv::Size(120, 90));
    BOOST_CHECK_EQUAL(kf->mag[0].type(), CV_64FC1);
    BOOST_CHECK_EQUAL(kf->idx, idx);
    cv::imwrite("test/data/out_img_resize.png", ::eds::utils::viz(kf->img));
    cv::imwrite("test/data/out_log_img_resize.png", ::eds::utils::viz(kf->log_img[0]));
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] KF K:\n"<<kf->K);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] KF K_ref:\n"<<kf->K_ref);

    /** Create the Tracker **/
    tracking::Tracker tracker(kf, tracker_config);

    /** Create the EventFrame **/
    tracking::EventFrame ef_1(idx, events, img.rows, img.cols, K, D, R_rect, P, "radtan", tracker_config.options.max_num_iterations.size(), tf, cv::Size(120, 90));
    cv::imwrite("test/data/out_event_frame_resize.png", ::eds::utils::viz(ef_1.frame[0]));
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] EF K:\n"<<ef_1.K);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] EF K_ref:\n"<<ef_1.K_ref);

    /** Optimize **/
    ::base::Transform3d T_kf_ef; T_kf_ef.setIdentity();
    tracker.optimize(0, &(ef_1.event_frame[0]), T_kf_ef, eds::tracking::LOSS_PARAM_METHOD::CONSTANT);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Optimize duration: "<<tracker.getInfo().meas_time_us<<"[us]");
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Optimize duration: "<<tracker.getInfo().time_seconds<<"[s]");
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] Optimize iterations: "<<tracker.getInfo().num_iterations);
}