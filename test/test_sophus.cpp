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
#include <eds/tracking/Types.hpp>

#include <yaml-cpp/yaml.h>
#include <thread>
#include <iostream>
using namespace eds;


BOOST_AUTO_TEST_CASE(test_sophus)
{
    BOOST_TEST_MESSAGE("###### TEST SOPHUS ######");

    base::Transform3d t (Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
    t.translation() << 2.0, 2.0, 2.0;

    SE3 se3(t.rotation(), t.translation());
    BOOST_TEST_MESSAGE("Transform 3D T:\n"<<t.translation());
    BOOST_TEST_MESSAGE("Transform 3D rotation:\n"<<t.rotation());
    BOOST_TEST_MESSAGE("SE3 :\n"<<se3.matrix());
    BOOST_TEST_MESSAGE("SE3 log()\n"<<se3.log());
}

BOOST_AUTO_TEST_CASE(test_sophus_log_and_exp)
{
    BOOST_TEST_MESSAGE("###### TEST SOPHUS ######");
    base::Transform3d t_0 (Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0)); //w, x, y, z
    t_0.translation() << 2.0, 2.0, 2.0;
    base::Transform3d t_1 (Eigen::Quaterniond(0.9238795, 0.0, 0.0, 0.3826834));//w, x, y, z
    t_1.translation() << 2.0, 2.0, 3.0;

    SE3 se3_0(t_0.rotation(), t_0.translation());
    SE3 se3_1(t_1.rotation(), t_1.translation());
    SE3 delta_se3 = se3_0.inverse() * se3_1;
    BOOST_TEST_MESSAGE("SE3_0 :\n"<<se3_0.matrix3x4());
    BOOST_TEST_MESSAGE("SE3_1 :\n"<<se3_1.matrix3x4());
    BOOST_TEST_MESSAGE("SE3 delta\n"<<delta_se3.matrix3x4());

    base::Vector6d delta_disp = delta_se3.log();
    BOOST_TEST_MESSAGE("SE3 delta log()\n"<<delta_disp);

    SE3 result = SE3::exp(delta_disp);
    BOOST_TEST_MESSAGE("SE3 result exp()\n"<<result.matrix3x4());
}

BOOST_AUTO_TEST_CASE(test_sophus_mean_se3)
{

    BOOST_TEST_MESSAGE("###### TEST SOPHUS ######");
    SE3MW50 se3; 
    for (int i=0; i<1000; ++i)
    {
        base::Transform3d t(Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));//w, x, y, z
        double noise = ((double) rand() / (RAND_MAX));
        t.translation() << 0.0, 0.0, noise;
        SE3 insert (t.rotation(), t.translation());
        std::cout<<"INSERT\n"<<insert.matrix3x4()<<std::endl;
        se3.push(SE3(insert));
        /** You can also get the quaternion
         * Eigen::Quaterniond q = se3.mean().unit_quaternion();
        std::cout<<"Quaternion:\n" <<q.x() <<q.y()<<q.z()<<q.w()<<std::endl;**/
        std::cout<<"MEAN["<<i<<"]\n "<<se3.mean().matrix3x4()<<std::endl;
        std::cout<<"******************"<<std::endl;
    }
}