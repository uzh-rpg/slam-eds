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

#ifndef _EDS_MAPPING_TYPES_HPP_
#define _EDS_MAPPING_TYPES_HPP_

#include <eds/tracking/Config.hpp>

#include <opencv2/opencv.hpp>
#include <base/Float.hpp>
#include <base/samples/DistanceImage.hpp>
#include <base/samples/Pointcloud.hpp>
#include <array>
#include <vector>
#include <iostream>
#include <random>

namespace eds { namespace mapping {

// user-defined point type
// inherits std::array in order to use operator[]
template<typename T>
class Point : public std::array<T, 2>
{
    public:

    // dimension of space (or "k" of k-d tree)
    // KDTree class accesses this member
    static const int DIM = 2;

    // the constructors
    Point() {}

    Point(const cv::Point_<T> p)
    {
        (*this) = Point(p.x, p.y);
    }

    Point(const T &x,  const T &y)
    {
        (*this)[0] = x;
        (*this)[1] = y;
    }

    T& x() {return (*this)[0];}
    T& y() {return (*this)[1];}

    // conversion to OpenCV Point
    operator cv::Point_<T>() const { return cv::Point_<T>((*this)[0], (*this)[1]); }
};

typedef Point<int> Point2i;
typedef Point<float> Point2f;
typedef Point<double> Point2d;

// user-defined point type
// inherits std::array in order to use operator[]
template<typename T>
class Point3: public std::array<T, 3>
{
    public:

    // dimension of space (or "k" of k-d tree)
    // KDTree class accesses this member
    static const int DIM = 3;

    // the constructors
    Point3() {}
    Point3(const cv::Point3_<T> p)
    {
        (*this) = Point3(p.x, p.y, p.z);
    }
    Point3(const T &x,  const T &y, const T &z)
    {
        (*this)[0] = x;
        (*this)[1] = y;
        (*this)[2] = z;
    }

    T& x() {return (*this)[0];}
    T& y() {return (*this)[1];}
    T& z() {return (*this)[2];}

    // conversion to OpenCV Point3
    operator cv::Point3_<T>() const { return cv::Point3_<T>((*this)[0], (*this)[1], (*this)[2]); }
};

typedef Point3<int> Point3i;
typedef Point3<float> Point3f;
typedef Point3<double> Point3d;

template<typename T>
class IDepthMap
{
    public:

    T fx, fy, cx, cy;
    std::vector< Point< T > > coord; // pixel coordinates of the map
    std::vector< T > idepth; // inverse depth

    public:
    IDepthMap():fx(1.0), fy(1.0), cx(0.0), cy(0.0){};
    IDepthMap(const std::vector<T> &intrinsics):fx(intrinsics[0]), fy(intrinsics[1]), cx(intrinsics[2]), cy(intrinsics[3]){};
    IDepthMap(T &fx_, T &fy_, T &cx_, T &cy_):fx(fx_), fy(fy_), cx(cx_), cy(cy_){};
    // Insert a 2D point in the image coord and
    // its associated inverse depth
    void inline insert (const T &x, const T &y, const T &idp)
    {
        coord.push_back(Point<T> (x, y));
        idepth.push_back(idp);
    }

    void remove (const int &index)
    {
        coord.erase(coord.begin() + index);
        idepth.erase(idepth.begin() + index);
    }

    void clear()
    {
        coord.clear();
        idepth.clear();
    }

    size_t size()
    {
        assert(coord.size() == idepth.size());
        return coord.size();
    }

    bool empty()
    {
        return this->size() == 0;
    }

    void fromDistanceImage(const ::base::samples::DistanceImage &dimg, const double &perturbance = 0.0)
    {
        /** Get median if necesary **/
        double median = 0.00; //depth median in distance img
        std::default_random_engine generator;// for the perturbation generation
        if (perturbance != 0.00)
        {
            std::vector<::base::samples::DistanceImage::scalar> data = dimg.data;
            std::nth_element(data.begin(), data.begin() + data.size()/2, data.end());
            median = data[data.size()/2];
            std::cout<<"MEDIAN GT DEPTH: "<< median<<" WITH PERTURBANCE PERCENT:"<<perturbance* 100.0<<" GIVES:"<<perturbance * median<<std::endl;
        }
        std::normal_distribution<double> distribution(0.0, perturbance * median);

        /** Clear the previous information **/
        coord.clear(); idepth.clear();

        /** Populate the inverse depth from the distance image **/
        for (int x=0; x<dimg.width; ++x)
        {
            for(int y=0; y<dimg.height; ++y)
            {
                const ::base::samples::DistanceImage::scalar d = dimg.data[dimg.width*y+x];
                if(::base::isNaN(d) != true)
                {
                    if (perturbance > 0.0)
                        insert(static_cast<T>(x), static_cast<T>(y), static_cast<T>(1.0/(d + distribution(generator))));
                    else
                        insert(static_cast<T>(x), static_cast<T>(y), static_cast<T>(1.0/d));
                }
            }
        }

        fx = static_cast<T>(1.0/dimg.scale_x); fy =  static_cast<T>(1.0/dimg.scale_y);
        cx = static_cast<T>(-dimg.center_x/dimg.scale_x);
        cy = static_cast<T>(-dimg.center_y/dimg.scale_y);
    }

    void fromDepthmapImage(const cv::Mat &img, const std::vector<double> &intrinsics, const double &min_depth, const double &max_depth)
    {
        /** Get intrinsics **/
        fx = intrinsics[0]; fy = intrinsics[1];
        cx = intrinsics[2]; cy = intrinsics[3];
        double min_inv_depth = 1.0/max_depth;
        double max_inv_depth = 1.0/min_depth;

        /** Compute to the depth range **/
        cv::Mat depthmap; img.convertTo(depthmap, CV_64FC1);
        double min, max; cv::minMaxLoc(depthmap, &min, &max);
        depthmap = (depthmap - min) / (max - min); // 0 - 1 image
        depthmap = (depthmap * (max_inv_depth-min_inv_depth)) + min_inv_depth; // min_depth - max_depth image

        /** Clear the previous information **/
        coord.clear(); idepth.clear();

        cv::minMaxLoc(depthmap, &min, &max);
        std::cout<<"[IDEPTHMAP] depthmap size: "<<img.size()<<" min: "<<min<<" max: "<<max<<std::endl;

        /** Populate the inverse depth from the distance image **/
        for (int x=0; x<depthmap.cols; ++x)
        {
            for(int y=0; y<depthmap.rows; ++y)
            {
                double inv_depth = depthmap.at<double>(y, x);
                if(::base::isNaN<double>(inv_depth) != true)
                {
                    insert(static_cast<T>(x), static_cast<T>(y), static_cast<T>(inv_depth));
                }
            }
        }
    }

    void toPointCloud(::base::samples::Pointcloud &pcl, const std::vector<double> &intrinsics={})
    {
        /** Get intrinsics **/
        if (!intrinsics.empty())
        {
            fx = intrinsics[0]; fy = intrinsics[1];
            cx = intrinsics[2]; cy = intrinsics[3];
        }

        pcl.points.clear(); pcl.colors.clear();

        for (size_t i=0; i<coord.size(); ++i)
        {
            T d_i = 1.0/idepth[i];
            pcl.points.push_back(::base::Point(d_i*((coord[i][0]-cx)/fx), d_i*((coord[i][1]-cy)/fy), d_i));
        }
    }

    inline void fromPointCloud(const ::base::samples::Pointcloud &pcl, const cv::Size &img_size, const std::vector<double> &intrinsics={})
    {
        return fromPoints(pcl.points, img_size, intrinsics);
    }

    inline void fromPoints(const std::vector<base::Point> &points, const cv::Size &img_size, const std::vector<double> &intrinsics={})
    {
        /** Get intrinsics **/
        if (!intrinsics.empty())
        {
            fx = intrinsics[0]; fy = intrinsics[1];
            cx = intrinsics[2]; cy = intrinsics[3];
        }

        /** Clear existing points (coord and idepth) **/
        clear();

        /** For all the points in map **/
        for (auto &it : points)
        {
            /** Project the point on the frame. x-y pixel coord, z is inverse depth **/
            cv::Point3d px(fx * (it[0]/it[2]) + cx, fy * (it[1]/it[2]) + cy, 1.0/it[2]);

            /** Check if the projected point is in the frame **/
            bool inlier = ((px.x >= 0.0) and (px.x < img_size.width)) and ((px.y >= 0.0) and (px.y < img_size.height));

            /** Push the point in to the vector **/
            if (inlier)
            {
                insert(static_cast<T>(px.x), static_cast<T>(px.y), static_cast<T>(px.z)); //pixel coordinates and inverse depth
            }
        }
    }
};

typedef IDepthMap<float> IDepthMap2f;
typedef IDepthMap<double> IDepthMap2d;


/** Typedef for keyframe information **/
/** [translation, quaternion, linear and angular velocities, img, height, width, K] **/
struct KeyFrameInfo
{
    KeyFrameInfo():ts(::base::Time()), t(Eigen::Vector3d::Zero()), q(Eigen::Quaterniond::Identity()),
                   v(::base::Vector6d::Zero()), height(0), width(0), K(Eigen::Matrix3d::Identity()){};
    KeyFrameInfo(const ::base::Time ts_, const Eigen::Vector3d &t_, const Eigen::Quaterniond &q_, const ::base::Vector6d &v_,
                 const std::vector<double> &img_, const uint16_t &height_, const uint16_t &width_,
                 const Eigen::Matrix3d &K_, const double &a_, const double &b_, const double &exposure_time_=1.0)
                  :ts(ts_), t(t_), q(q_), v(v_), img(img_), height(height_), width(width_), K(K_),
                  a(a_), b(b_), exposure_time(exposure_time_){};


    ::base::Time ts; //time
    Eigen::Vector3d t; //translation
    Eigen::Quaterniond q; //quaternion
    ::base::Vector6d v; //velocities
    std::vector<double> img; //image
    uint16_t height; uint16_t width; //image size
    Eigen::Matrix3d K; //intrinsics
    double a; //affine brightness transfer parameter a
    double b; //affine brightness transfer parameter b
    double exposure_time; //frame exposure time (if available)
};

/** Typedef for active points informatioon **/
/** [2D pixel coord, 2D norm coord, d x d patch, inverse depth, intensity value] **/
struct PointInfo
{
    PointInfo(){};
    PointInfo(const ::eds::mapping::Point2d &coord_, const ::eds::mapping::Point2d &norm_coord_,
             const std::vector<double> &patch_, const double &inv_depth_, const double &intensity_,
             const double &residual_, const double &gradient_)
            :coord(coord_), norm_coord(norm_coord_), patch(patch_), inv_depth(inv_depth_),
            intensity(intensity_), residual(residual_), gradient(gradient_){};

    ::eds::mapping::Point2d coord; //pixel coord
    ::eds::mapping::Point2d norm_coord; //normalized coord
    std::vector<double> patch; //bundle patch of this point
    double inv_depth; //inverse depth for all points in the patch
    double intensity; //image intensity
    double residual; //point residual
    double gradient; //gradient magnitude

    static bool residual_order(PointInfo &p_i, PointInfo &p_j)
    {
        return p_i.residual < p_j.residual; //lower residual the better
    }

    static bool gradient_order(PointInfo &p_i, PointInfo &p_j)
    {
        return p_i.gradient > p_j.gradient; //higher gradient the better
    }

    friend std::ostream& operator<<(std::ostream& os, const PointInfo& dt);
};

inline std::ostream& operator<<(std::ostream& os, const PointInfo& p)
{
    os <<"coord: "<<p.coord[0]<<","<<p.coord[1]<<"inv_depth: "<<p.inv_depth;
    return os;
}

struct PointShortInfo
{
    PointShortInfo(){};
    PointShortInfo(const uint64_t &kf_id_, const uint32_t &id_, const double &residual_)
                    :kf_id(kf_id_), id(id_), residual(residual_){};
    uint64_t kf_id;
    uint32_t id;
    double residual;

    static bool residual_order(PointShortInfo &p_i, PointShortInfo &p_j)
    {
        return p_i.residual < p_j.residual; //lower residual the better
    }
};

} // mapping namespace
} // end namespace


#endif // _EDS_MAPPING_TYPES_HPP_

