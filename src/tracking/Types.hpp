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

#ifndef _EDS_TRACKING_TYPES_HPP_
#define _EDS_TRACKING_TYPES_HPP_

#include <eds/tracking/Config.hpp>

/** Sophus **/
#include <eds/sophus/sim3.hpp>
#include <eds/sophus/se2.hpp>
#include <eds/sophus/se3.hpp>
#include <eds/sophus/sophus.hpp>

#include <opencv2/opencv.hpp>
#include <base/Eigen.hpp>
#include <base/Float.hpp>
#include <vector>


namespace eds { namespace tracking {

struct KFImagePyr
{
    size_t w;// image width
    size_t h;// image height
    double fx, fy, cx, cy; // instrinsics
    cv::Mat img; // undistort image
    std::vector<cv::Point2d> coord;// coordinates of points
    std::vector<cv::Point2d> norm_coord;// normalized coordinates of points
    std::vector<cv::Point2d> grad;// x-y gradient of the points
    std::vector<double> idp;// inverse depth of the points
};

struct ManageKFImagePyr
{
    uint8_t num_level; //number of pyramid levels
    std::vector<::eds::tracking::KFImagePyr> image; // the pyramed images
};

struct EventImagePyr
{
    size_t w;// image width
    size_t h;// image height
    double fx, fy, cx, cy;// instrinsics
    cv::Mat frame;// Event frame (integration of events) no normalized
    std::vector<double> event_frame;// event_frame = frame / norm
    double norm; // norm of the event frame
};

struct ManageEventImagePyr
{
    uint8_t num_level; //number of pyramid levels
    std::vector<::eds::tracking::EventImagePyr> image;
};

} // tracking namespace

typedef Sophus::SE3d SE3;
typedef Sophus::Sim3d Sim3;
typedef Sophus::SO3d SO3;

/** SE3 Moving window **/
template <int N>
class SE3MW
{
private:
    typedef typename std::array<SE3, N> vector_type;
    typedef typename vector_type::iterator iterator;
    typedef typename vector_type::const_iterator const_iterator;

public:
    vector_type data;
    size_t head, tail;

public:
    SE3MW() { for(auto &it : data) it = SE3(); head = tail = 0;};

    void advance()
    {
        /** Advance indexes **/
        if (++(tail) == N)
        {
            tail = 0;
        }

        if (++(head) == N)
        {
            head = 0;
        }

        /** When full **/
        if (head == tail)
        {
            if (++(tail) == N)
            {
                tail = 0;
            }
        }
    };

    void push(const SE3 &se3)
    {
        data[head] = se3;
        advance();
    };

    SE3 mean()
    {
        int number = 0;
        base::Vector6d mean_log = base::Vector6d::Zero();
        for (size_t i=tail; i != head; i = (i+1)%N)
        {
            mean_log += data[i].log();
            number++;
        }
        //std::cout<<"number: "<<number<<std::endl;
        SE3 result = SE3::exp(mean_log/number);
        return result;
    };

    inline iterator begin() noexcept {return data.begin();};
    inline const_iterator cbegin() const noexcept {return data.cbegin();};
    inline iterator end() noexcept {return data.end();};
    inline const_iterator cend() const noexcept {return data.cend();};
};

typedef SE3MW<10> SE3MW10;
typedef SE3MW<20> SE3MW20;
typedef SE3MW<50> SE3MW50;

} // end namespace
#endif // _EDS_TRACKING_TYPES_HPP_

