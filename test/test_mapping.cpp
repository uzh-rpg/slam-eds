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
using namespace cv;

BOOST_AUTO_TEST_CASE(test_read_mapping_config)
{
    BOOST_TEST_MESSAGE("###### TEST MAPPING CONFIG ######");

    std::string test_yaml_fn = "test/data/eds.yaml";
    YAML::Node config = YAML::LoadFile(test_yaml_fn);
    ::eds::mapping::Config mapping_config = ::eds::mapping::readMappingConfig(config["mapping"]);
    BOOST_CHECK_EQUAL(mapping_config.min_depth, 1.0);
    BOOST_CHECK_EQUAL(mapping_config.max_depth, 2.0);
    BOOST_CHECK_EQUAL(mapping_config.convergence_sigma2_thresh, 100);
    BOOST_CHECK_EQUAL(mapping_config.sor_active, true);
    BOOST_CHECK_EQUAL(mapping_config.sor_nb_points, 6);
    BOOST_CHECK_EQUAL(mapping_config.sor_radius, 0.05);
}

BOOST_AUTO_TEST_CASE(test_points)
{

    ::eds::mapping::Point2f my_point(10, 10);
    BOOST_TEST_MESSAGE("my_point 2d: "<<cv::Point2f(my_point));
    BOOST_CHECK (my_point.x() == my_point[0]);
    BOOST_CHECK (my_point.y() == my_point[1]);
    BOOST_CHECK (my_point[0] == my_point[1]);

    ::eds::mapping::Point3f my_point_3d(10, 10, 10);
    BOOST_TEST_MESSAGE("my_point 3d: "<<cv::Point3f(my_point_3d));
    BOOST_CHECK ((my_point_3d[0] == my_point_3d[1]) && (my_point_3d[1] == my_point_3d[2]));

    cv::Point2f cv_p_0 (12, 12), cv_p_1(12, 12);
    BOOST_CHECK (cv_p_0 == cv_p_1);
    ::eds::mapping::Point2f my_p_0(cv_p_0), my_p_1(cv_p_1);
    BOOST_CHECK (my_p_0 == my_p_1);
    BOOST_CHECK (my_p_0.x() == my_p_0[0]);
    BOOST_CHECK (my_p_0.y() == my_p_0[1]);
    BOOST_TEST_MESSAGE("cv_point: "<< cv_p_0);
    BOOST_TEST_MESSAGE("my_point from cv point: "<< my_p_0[0]<<" "<< my_p_0[1]);

    cv::Point3f cv_3dp_0 (rand(), rand(), rand());
    cv::Point3f cv_3dp_1 (cv_3dp_0.x, cv_3dp_0.y, cv_3dp_0.z);
    BOOST_CHECK (cv_3dp_0 == cv_3dp_1);
    ::eds::mapping::Point3f my_3dp_0(cv_3dp_0), my_3dp_1(cv_3dp_1);
    BOOST_CHECK (my_3dp_0 == my_3dp_1);
    BOOST_CHECK (my_3dp_0.x() == my_3dp_0[0]);
    BOOST_CHECK (my_3dp_0.y() == my_3dp_0[1]);
    BOOST_CHECK (my_3dp_0.z() == my_3dp_0[2]);
    BOOST_TEST_MESSAGE("cv_point_3d: "<< cv_3dp_0);
    BOOST_TEST_MESSAGE("my_point_3d from cv point: "<< my_3dp_0[0]<<" "<< my_3dp_0[1]<<" "<<my_3dp_0[2]);
}

BOOST_AUTO_TEST_CASE(test_map_to_vectors)
{
    std::map<int, std::string> map({{0, "hola"}, {1, "adios"}, {2, "bien"}});

    std::vector<std::pair<int, std::string>> v = ::eds::utils::mapToVector(map);

    std::pair<std::vector<int>, std::vector<std::string>> p = ::eds::utils::mapToVectors(map);
    auto it_m=map.begin();
    auto it_v=v.begin();
    int i = 0;
    for(; it_m!=map.end() || it_v!=v.end(); ++it_m, ++it_v)
    {
        BOOST_TEST_MESSAGE("first: "<< it_m->first<<" second: "<<it_m->second);
        BOOST_CHECK_EQUAL(it_m->first, it_v->first);
        BOOST_CHECK_EQUAL(it_m->second, it_v->second);
        BOOST_CHECK_EQUAL(it_m->first, p.first[i]);
        BOOST_CHECK_EQUAL(it_m->second, p.second[i]);
        ++i;
    }
}

BOOST_AUTO_TEST_CASE(test_vector_unique_random)
{
    std::vector<int> v(50);
    ::eds::utils::random_unique(v.begin(), v.end(), 3);
}

BOOST_AUTO_TEST_CASE(test_vector_norm)
{
    BOOST_TEST_MESSAGE("###### TEST VECTOR NORM ######");
    std::vector<int> v = {1, 2, 3};
    BOOST_CHECK_EQUAL(::eds::utils::vectorNorm(v.begin(), v.end()), sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]));
}

BOOST_AUTO_TEST_CASE(test_inverse_depth_map)
{

    ::eds::mapping::IDepthMap2d depthmap;
    double idp = 0.5;
    int num_points = 100;

    for (int x=0; x<num_points; ++x)
    {
        for (int y=0; y<num_points; ++y)
        {
            depthmap.insert(static_cast<double>(x), static_cast<double>(y), idp);
        } 
    }

    BOOST_TEST_MESSAGE("inverse map size: "<<depthmap.size());
    BOOST_CHECK (depthmap.size() == (size_t)(num_points * num_points));

    depthmap.remove(0);
    BOOST_TEST_MESSAGE("inverse map size: "<<depthmap.size());
    BOOST_CHECK (depthmap.size() == (size_t)(num_points * num_points)-1);

    depthmap.clear();
    BOOST_CHECK (depthmap.size() == 0);
}

BOOST_AUTO_TEST_CASE(test_depth_from_distance_image)
{
    ::base::samples::DistanceImage distance_image;
    int height, width; height=180; width=240;
    distance_image.height = height;
    distance_image.width = width;

    for(auto i=0; i<(height*width); ++i)
    {
        distance_image.data.push_back(2.00);
    } 


    ::eds::mapping::IDepthMap2d depthmap;
    depthmap.fromDistanceImage(distance_image);
    BOOST_TEST_MESSAGE("inverse map size: "<<depthmap.size());
    BOOST_CHECK (depthmap.size() == distance_image.data.size());
    BOOST_CHECK (depthmap.idepth[0] == 1.0/distance_image.data[0]);
    BOOST_CHECK (!depthmap.empty());
    BOOST_CHECK (depthmap.empty() != true);
}

BOOST_AUTO_TEST_CASE(test_kd_tree)
{

    const int seed = 10; 
	srand(seed);

	// generate space
	const int width = 500;
	const int height = 500;
	cv::Mat img = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);

	// generate points
	const int npoints = 100;
	std::vector<::eds::mapping::Point2d> points(npoints);
	for (int i = 0; i < npoints; i++)
	{
		const int x = rand() % width;
		const int y = rand() % height;
		points[i] = ::eds::mapping::Point2d(x, y);
	}

	for (const auto& pt : points)
		cv::circle(img, cv::Point2d(pt), 1, cv::Scalar(0, 255, 255), -1);

	// build k-d tree
	::eds::mapping::KDTree<eds::mapping::Point2d> kdtree(points);

	// generate query (center of the space)
	const eds::mapping::Point2d query(0.5 * width, 0.5 * height);
	cv::circle(img, cv::Point2d(query), 1, cv::Scalar(0, 0, 255), -1);

	// nearest neigbor search
	const cv::Mat I0 = img.clone();
	const int idx = kdtree.nnSearch(query);
	cv::circle(I0, cv::Point2d(points[idx]), 1, cv::Scalar(255, 255, 0), -1);
	cv::line(I0, cv::Point2d(query), cv::Point2d(points[idx]), cv::Scalar(0, 0, 255));

	// k-nearest neigbors search
	const cv::Mat I1 = img.clone();
	const int k = 10;
	const std::vector<int> knnIndices = kdtree.knnSearch(query, k);
	for (int i : knnIndices)
	{
		cv::circle(I1, cv::Point2d(points[i]), 1, cv::Scalar(255, 255, 0), -1);
		cv::line(I1, cv::Point2d(query), cv::Point2d(points[i]), cv::Scalar(0, 0, 255));
	}
	
	// radius search
	const cv::Mat I2 = img.clone();
	const double radius = 50;
	const std::vector<int> radIndices = kdtree.radiusSearch(query, radius);
	for (int i : radIndices)
		cv::circle(I2, cv::Point2d(points[i]), 1, cv::Scalar(255, 255, 0), -1);
	cv::circle(I2, cv::Point2d(query), cvRound(radius), cv::Scalar(0, 0, 255));
    
	// show results
	//cv::imshow("Nearest neigbor search", I0);
	//cv::imshow("K-nearest neigbors search (k = 10)", I1);
	//cv::imshow("Radius search (radius = 50)", I2);

	//cv::waitKey(0);

    cv::imwrite("test/data/out_nearest_kdtree.jpg", I0);

    BOOST_CHECK (points.size() == npoints);
    BOOST_CHECK (kdtree.size() == points.size());
    BOOST_CHECK (kdtree.validate() == true);
    BOOST_TEST_MESSAGE("KDTree size : "<<kdtree.size());
    kdtree.clear();
    BOOST_CHECK (kdtree.size() == 0);


}

BOOST_AUTO_TEST_CASE(test_kd_tree_sparse)
{
    const int seed = 10;
    srand(seed);

    // generate space
    const int width = 500;
    const int height = 500;
    cv::Mat img = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);

    // generate points
    const int npoints = 100;
    size_t n_elements = npoints/2;
    std::vector<::eds::mapping::Point2d> points(npoints);
    for (int i = 0; i < npoints; i++)
    {
        const int x = rand() % width;
        const int y = rand() % height;
        points[i] = ::eds::mapping::Point2d(x, y);
    }

    for (size_t i=0; i<n_elements; ++i)
        cv::circle(img, cv::Point2d(points[i]), 1, cv::Scalar(0, 255, 255), -1);

    // build k-d tree with the first n elements
    auto points_begin = points.begin();
    auto points_end = points.begin();
    std::advance(points_end, n_elements);
    BOOST_CHECK_EQUAL(std::distance(points_begin, points_end), n_elements);
    ::eds::mapping::KDTree<eds::mapping::Point2d> kdtree(points_begin, points_end);

    const cv::Mat I0 = img.clone();
    for (size_t i=0; i<(npoints-n_elements); ++i)
    {
        // generate query (center of the space)
        const eds::mapping::Point2d query = points[n_elements + i];
        cv::circle(img, cv::Point2d(query), 1, cv::Scalar(0, 0, 255), -1); //red color


        // nearest neigbor search
        const int idx = kdtree.nnSearch(query);
        cv::circle(I0, cv::Point2d(points[idx]), 1, cv::Scalar(255, 255, 0), -1);
        cv::line(I0, cv::Point2d(query), cv::Point2d(points[idx]), cv::Scalar(0, 0, 255));
    }

    cv::imwrite("test/data/out_nearest_kdtree_sparse.jpg", I0);

    BOOST_CHECK (points.size() == npoints);
    BOOST_CHECK (kdtree.size() == n_elements);
    BOOST_CHECK (kdtree.validate() == true);
    BOOST_TEST_MESSAGE("KDTree size : "<<kdtree.size());
    kdtree.clear();
    BOOST_CHECK (kdtree.size() == 0);
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

BOOST_AUTO_TEST_CASE(test_kf_depth_from_distance_image)
{
    ::base::samples::DistanceImage distance_image;
    int height, width; height=180; width=240;
    distance_image.height = height;
    distance_image.width = width;

    /* initialize random seed: */
    srand (time(NULL) );

    /* generate secret number: */
    double d_rand = rand() % 10 + 1;

    for(auto i=0; i<(height*width); ++i)
    {
        distance_image.data.push_back(d_rand);
    }


    ::eds::mapping::IDepthMap2d depthmap;
    depthmap.fromDistanceImage(distance_image);
    BOOST_TEST_MESSAGE("inverse map size: "<<depthmap.size());
    BOOST_CHECK (depthmap.size() == distance_image.data.size());
    BOOST_CHECK (depthmap.idepth[0] == 1.0/distance_image.data[0]);
    BOOST_CHECK (!depthmap.empty());
    BOOST_CHECK (depthmap.empty() != true);

    uint64_t idx = 0;
    ::base::Time ts;
    cv::Mat img = cv::imread("test/data/frame_00000000.png", cv::IMREAD_GRAYSCALE);
    base::Affine3d tf = base::Affine3d::Identity();
    cv::Mat K, D, R_rect, P;
    K = cv::Mat_<double>::eye(3, 3);
    D = cv::Mat_<double>::zeros(4, 1);
    readCalibration(K, D, "test/data/calib.txt");
    std::cout<<"Image size: "<<img.size()<<std::endl;

    tracking::KeyFrame kf(idx, ts, img, depthmap, K, D, R_rect, P, "radtan",
                    ::eds::tracking::MAX, 1.0 /*min_depth*/, 2.0/*max_depth*/,
                    100.0 /*threshold*/, 10.0 /*percent_points */, tf);
    BOOST_CHECK (kf.coord.size() == kf.inv_depth.size()); //  check size
    BOOST_CHECK (depthmap.idepth[0] == ::eds::mapping::mu(kf.inv_depth[0])); // all the depth are the same
    BOOST_TEST_MESSAGE("Inverse depth [0]: "<< ::eds::mapping::mu(kf.inv_depth[0]));

    std::vector<::base::Point> pcl = kf.getDepthMap();
    BOOST_CHECK (pcl.size() == kf.inv_depth.size());
    for (auto p=pcl.begin(); p<pcl.end(); ++p)
        BOOST_CHECK_EQUAL ((*p)[2], d_rand);

}


BOOST_AUTO_TEST_CASE(test_small_eigen_access_test)
{
    Eigen::Vector3d v(123, 3456, 76);
    BOOST_TEST_MESSAGE("v: "<<v[0]<<","<<v[1]<<","<<v[2]);
    BOOST_TEST_MESSAGE("v: "<<v.data()[0]<<","<<v.data()[1]<<","<<v.data()[2]);
    BOOST_TEST_MESSAGE("v: "<<&(v.data())[0]<<","<<&(v.data())[1]<<","<<&(v.data())[2]);
    BOOST_CHECK_EQUAL (v[2], v.data()[2]);
}

BOOST_AUTO_TEST_CASE(test_mapping_insert_kf)
{
    BOOST_TEST_MESSAGE("###### TEST GLOBAL_MAP INSERT KF######");
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
    ::eds::tracking::Config tracking_config = ::eds::tracking::readTrackingConfig(config["tracker"]);
    ::eds::mapping::Config mapping_config = ::eds::mapping::readMappingConfig(config["mapping"]);
    ::eds::bundles::Config bundles_config = ::eds::bundles::readBundlesConfig(config["bundles"]);

    ::eds::calib::CameraInfo cam_info;
    cam_info.height = img.rows; cam_info.width = img.cols;
    cam_info.D = D;
    cam_info.intrinsics = std::vector<double> {K.at<double>(0,0), K.at<double>(1,1),
                                K.at<double>(0,2), K.at<double>(1,2)};

    /** Create the KeyFrame **/
    std::shared_ptr<eds::tracking::KeyFrame> kf =
            std::make_shared<eds::tracking::KeyFrame>(idx, ts, img, depthmap, K, D, R_rect, P,
            "radtan", eds::tracking::MAX, mapping_config.min_depth, mapping_config.max_depth,
            mapping_config.convergence_sigma2_thresh, 10.0, tf);

    /** Global Map **/
    eds::mapping::GlobalMap global_map(mapping_config, cam_info, bundles_config,
                                        tracking_config.percent_points);

    /** Check than global map returns and empty idepthmap when the KF is not in GlobalMap **/
    global_map.getIDepthMap(kf->idx, depthmap, true);
    BOOST_CHECK_EQUAL(depthmap.empty(), true);

    /** Insert KF **/
    global_map.insert(kf);

    for (auto it : global_map.camera_kfs)
        BOOST_TEST_MESSAGE("global map host idx:\n"<<it.first);

    eds::mapping::KeyFrameInfo &kf_info = global_map.camera_kfs[100];
    BOOST_TEST_MESSAGE("K matrix in map\n"<<kf_info.K);
    BOOST_CHECK_EQUAL(kf_info.K(0,0), kf->K_ref.at<double>(0,0));
    BOOST_CHECK_EQUAL(global_map.point_kfs[100].size(), kf->coord.size());

    /** Test patches **/
    for (auto it : global_map.camera_kfs)
    {
        BOOST_TEST_MESSAGE("KF["<<it.first<<"] in Global Map with: "<< global_map.point_kfs[it.first].size()<<" points");
        int i = 0;
        for (auto ip : global_map.point_kfs[it.first])
        {
            size_t p_in_map = ip.patch.size();
            size_t p_in_kf = kf->bundle_patches[i].size();
            BOOST_CHECK_EQUAL(p_in_map, p_in_kf);
            //BOOST_TEST_MESSAGE("\tpatch ["<<i<<"] size "<<p_in_kf);
            for (size_t j=0; j<p_in_map; ++j)
                BOOST_CHECK_EQUAL(ip.patch[j], kf->bundle_patches[i][j]);
            ++i;
        }
    }

    /** Test the draw Image from Global Map PointKFS **/
    img = ::eds::utils::drawValuesPointInfo(global_map.point_kfs.at(kf->idx), cam_info.height, cam_info.width, "bilinear", 0.0);
    cv::Mat color_img = kf->viz(img, true);
    cv::imwrite("test/data/out_img_grad_point_info.jpg", color_img);

    /** Test the return of inverse depth map **/
    global_map.getIDepthMap(kf->idx, depthmap, true);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] depthmap coord "<<depthmap.coord.size()<<" depth "<<depthmap.idepth.size());
}

BOOST_AUTO_TEST_CASE(test_mapping_selected_points)
{
    BOOST_TEST_MESSAGE("###### TEST GLOBAL MAP SELECTED POINTS ######");
    uint64_t idx = 100;
    ::base::Time ts;
    cv::Mat img = cv::imread("test/data/frame_00000000.png", cv::IMREAD_GRAYSCALE);
    eds::mapping::IDepthMap2d depthmap;
    base::Affine3d tf = base::Affine3d::Identity();
    cv::Mat K, D, R_rect, P;
    K = cv::Mat_<double>::eye(3, 3);
    D = cv::Mat_<double>::zeros(4, 1);
    readCalibration(K, D, "test/data/calib.txt");
    ::eds::calib::CameraInfo cam_info;
    cam_info.height = img.rows; cam_info.width = img.cols;
    cam_info.D = D; cam_info.intrinsics = std::vector<double> {10, 10, 10, 10};

    std::string test_yaml_fn = "test/data/eds.yaml";
    YAML::Node config = YAML::LoadFile(test_yaml_fn);
    ::eds::tracking::Config tracking_config = ::eds::tracking::readTrackingConfig(config["tracker"]);
    ::eds::mapping::Config mapping_config = ::eds::mapping::readMappingConfig(config["mapping"]);
    ::eds::bundles::Config bundles_config = ::eds::bundles::readBundlesConfig(config["bundles"]);

    /** Create the KeyFrame **/
    std::shared_ptr<eds::tracking::KeyFrame> kf =
            std::make_shared<eds::tracking::KeyFrame>(idx, ts, img, depthmap, K, D, R_rect, P,
            "radtan", eds::tracking::MAX, mapping_config.min_depth, mapping_config.max_depth,
            mapping_config.convergence_sigma2_thresh, 10.0, tf);

    /** Global Map **/
    std::shared_ptr<eds::mapping::GlobalMap> global_map =
        std::make_shared<eds::mapping::GlobalMap>(mapping_config, cam_info, bundles_config,
        tracking_config.percent_points);

    /** Insert KF **/
    global_map->insert(kf);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] global map total active points: "<<global_map->total_active_points);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] global map active points per KF: "<<global_map->active_points_per_kf);
}

BOOST_AUTO_TEST_CASE(test_mapping_optical_flow)
{
    BOOST_TEST_MESSAGE("###### TEST MAP OPTICAL FLOW ######");
    uint64_t idx = 100;
    ::base::Time ts;
    cv::Mat img = cv::imread("test/data/frame_00000000.png", cv::IMREAD_GRAYSCALE);
    eds::mapping::IDepthMap2d depthmap;
    base::Affine3d tf = base::Affine3d::Identity();
    cv::Mat K, D, R_rect, P;
    K = cv::Mat_<double>::eye(3, 3);
    D = cv::Mat_<double>::zeros(4, 1);
    readCalibration(K, D, "test/data/calib.txt");
    ::eds::calib::CameraInfo cam_info;
    cam_info.height = img.rows; cam_info.width = img.cols;
    cam_info.D = D; cam_info.intrinsics = std::vector<double> {10, 10, 10, 10};

    std::string test_yaml_fn = "test/data/eds.yaml";
    YAML::Node config = YAML::LoadFile(test_yaml_fn);
    ::eds::mapping::Config mapping_config = ::eds::mapping::readMappingConfig(config["mapping"]);
 
    /** Create the KeyFrame **/
    std::shared_ptr<eds::tracking::KeyFrame> kf =
            std::make_shared<eds::tracking::KeyFrame>(idx, ts, img, depthmap, K, D, R_rect, P,
            "radtan", eds::tracking::MAX, mapping_config.min_depth, mapping_config.max_depth,
            mapping_config.convergence_sigma2_thresh, 10.0, tf);

    /** Points tracks dummy value **/
    for (auto it=kf->tracks.begin(); it!=kf->tracks.end(); ++it)
        *it = Eigen::Vector2d(10.0, 10.0);

    /** Save optical flow image **/
    cv::Mat of_viz = ::eds::utils::flowArrowsOnImage(kf->img, kf->coord, kf->tracks);
    cv::imwrite("test/data/out_optical_flow_mapping_viz.png", of_viz);
}

BOOST_AUTO_TEST_CASE(test_mapping_depth_points_constructor)
{
    BOOST_TEST_MESSAGE("###### TEST MAP DEPTH POINTS ######");
    cv::Mat K, D;
    K = cv::Mat_<double>::eye(3, 3);
    D = cv::Mat_<double>::zeros(4, 1);
    readCalibration(K, D, "test/data/calib.txt");

    /** Create the Depth points **/
    uint32_t num_points = 1000;
    double min_depth = 0.5;
    double max_depth = 10.0;
    double convergence_thresh = 100.0;

    ::eds::mapping::DepthPoints depth_points (K, num_points, min_depth, max_depth, convergence_thresh);
    BOOST_CHECK_EQUAL(depth_points.size(), num_points);
    BOOST_CHECK_EQUAL(depth_points.depthRange(), (max_depth - min_depth));
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] depth_points[10] mu:"<<depth_points[10][0]<<", sigma2:"
                        <<depth_points[10][1]<<", a:"<<depth_points[10][2]<<", b:"<<depth_points[10][3]);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] depth_points[10] mu:"<<::eds::mapping::mu(depth_points[10])<<", sigma2:"
                        <<::eds::mapping::sigma2(depth_points[10])<<", a:"<<::eds::mapping::a(depth_points[10])
                        <<", b:"<<::eds::mapping::b(depth_points[10]));

    BOOST_TEST_MESSAGE("[BOOST MESSAGE] K:\n"<< depth_points.K());
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] intrinsics fx["<< depth_points.fx()<<"] fy["<<depth_points.fy()
                        <<"] cx["<<depth_points.cx()<<"] cy["<<depth_points.cy()<<"]");
    BOOST_CHECK_EQUAL(depth_points.fx(), K.at<double>(0,0));
    BOOST_CHECK_EQUAL(depth_points.fy(), K.at<double>(1,1));
    BOOST_CHECK_EQUAL(depth_points.cx(), K.at<double>(0,2));
    BOOST_CHECK_EQUAL(depth_points.cy(), K.at<double>(1,2));

    for (auto it=depth_points.begin(); it!=depth_points.end(); ++it)
    {
        BOOST_CHECK_EQUAL(eds::mapping::mu(*it), 1.0/((max_depth - min_depth)/2.0));
    }

    /** Mean and st_dev **/
    double mean, st_dev;
    depth_points.meanIDepth(mean, st_dev);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] mean: "<< mean <<" std: "<<st_dev);
    BOOST_CHECK_CLOSE(mean, 1.0/((max_depth - min_depth)/2.0), 1e-06);

    /** Median and Q3 **/
    double median, third_q;
    depth_points.medianIDepth(median, third_q);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] median(q_2): "<< median <<" q_3: "<<third_q);
    BOOST_CHECK_CLOSE(mean, median, 1e-06);

    /** Direct access to data **/
    eds::mapping::mu(depth_points[10]) = 10.0;
    BOOST_CHECK_EQUAL(eds::mapping::mu(depth_points[10]), 10.0);

    std::vector<cv::Point2d> kf_coord(num_points), ef_coord(num_points);
    ::base::Transform3d T_kf_ef(Eigen::Quaterniond(Eigen::AngleAxisd(0.5*M_PI, Eigen::Vector3d::UnitZ())));
    T_kf_ef.translation() << 1.0, 2.0, 0.3;

    depth_points.update(T_kf_ef, kf_coord, ef_coord);
}

BOOST_AUTO_TEST_CASE(test_mapping_depth_points)
{
    BOOST_TEST_MESSAGE("###### TEST MAP DEPTH POINTS ######");
    cv::Mat K, D;
    K = cv::Mat_<double>::eye(3, 3);
    D = cv::Mat_<double>::zeros(4, 1);
    readCalibration(K, D, "test/data/calib.txt");

    /** Create the Depth points **/
    srand((unsigned)time(NULL));
    uint32_t num_points = 10;
    std::vector<double>init_depth(num_points);
    double min_depth = 0.5;
    double max_depth = 10.0;
    for (auto it=init_depth.begin(); it!=init_depth.end(); ++it)
    {
        (*it) = rand() % 10 + 1;
    }

    /** Create the depth points **/
    double convergence_thresh = 100.0;
    ::eds::mapping::DepthPoints depth_points (K, init_depth, min_depth, max_depth, convergence_thresh);

    BOOST_CHECK_EQUAL(depth_points.size(), init_depth.size());
    BOOST_CHECK_EQUAL(depth_points.size(), num_points);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] K:\n"<< depth_points.K());

    /** Check the initial points values **/
    auto it = depth_points.begin();
    for (size_t i=0; i<depth_points.size(); ++i, ++it)
    {
        BOOST_CHECK_EQUAL(eds::mapping::mu(depth_points[i]), init_depth[i]);
        BOOST_CHECK_EQUAL(eds::mapping::mu(*it), init_depth[i]);
    } 

    /** Simple update **/ 
    std::vector<cv::Point2d> kf_coord(num_points), ef_coord(num_points);
    ::base::Transform3d T_kf_ef(Eigen::Quaterniond::Identity());
    T_kf_ef.translation() << 0.5, 0.0, 0.0;

    depth_points.update(T_kf_ef, kf_coord, ef_coord);

    /** Get vector of inverse depth **/
    std::vector<double> mu;
    depth_points.getIDepth(mu);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] return idepth size: "<< mu.size());

    /** Erase data **/
    it = depth_points.begin();
    for (; it!=depth_points.end();)
    {
        it = depth_points.erase(it);   
    }
    BOOST_CHECK_EQUAL(depth_points.size(), 0);
    BOOST_CHECK_EQUAL(depth_points.empty(), true);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] size: "<< depth_points.size());
 
    /** Play with random number generation **/
    srand(time(0));  // Initialize random number generator.
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] rand() [0-2] "<< static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(2.0-0.0)));

}