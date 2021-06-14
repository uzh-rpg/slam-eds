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

#ifndef _EDS_UTILS_HPP_
#define _EDS_UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <map>
#include <assert.h>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>

/** Base types **/
#include <base/Eigen.hpp>
#include <base/Time.hpp>
#include <base/samples/Twist.hpp>
#include <base/samples/Event.hpp>
#include <base/samples/EventArray.hpp>

/** EDS Types **/
#include <eds/utils/Calib.hpp>
#include <eds/mapping/Types.hpp>

namespace eds { namespace utils {

    /** Trajectory alignment **/
    template<typename T, int N>
    class Alignment
    {
    public:
        static const int DIM = 3;
        typedef Eigen::Matrix<T, DIM+1, N> MatrixX;
        Eigen::Matrix<T, DIM+1, DIM+1> cR_t;

    private:
        int index;
        MatrixX model/*src*/, data/*dst*/;

    public:
        // the constructors
        Alignment()
        {
            index = 0;
            model = MatrixX::Zero();
            data = MatrixX::Zero();
            model.row(DIM) = Eigen::Matrix<T, 1, Eigen::Dynamic>::Constant(N, T(1));
            data.row(DIM) = Eigen::Matrix<T, 1, Eigen::Dynamic>::Constant(N, T(1));
            cR_t = Eigen::Matrix<T, DIM+1, DIM+1>::Identity();
        }

        Eigen::Matrix<T, DIM+1, DIM+1> realign(Eigen::Matrix<T, DIM, 1> &model_vector, Eigen::Matrix<T, DIM, 1> &data_vector)
        {
            /** Add new information **/
            model.col(index) << model_vector(0), model_vector(1), model_vector(2), 1;
            data.col(index) << data_vector(0), data_vector(1), data_vector(2), 1;
            index = (index + 1) % N;

            /** Compute Umeyama **/
            Eigen::Block<MatrixX, DIM, N> model_block(model,0,0,DIM,N);
            Eigen::Block<MatrixX, DIM, N> data_block(data,0,0,DIM,N);
            cR_t =  umeyama(model_block, data_block);
            return cR_t;
        }
    };

    enum SIMILARITY_MEASURE{NCC, ZNCC, SSD, NSSD, ZSSD, SAD, ZSAD};

    std::string type2str(int type);

    cv::Mat drawValuesPoints(const std::vector<cv::Point2d> &points, const std::vector<int8_t> &values, const int height, const int width, const std::string &method = "nn", const float s=0.5, const bool &use_exp_weights=false);

    cv::Mat drawValuesPoints(const std::vector<cv::Point2d> &points, const std::vector<double> &values, const int height, const int width, const std::string &method = "nn", const float s=0.5);
 
    cv::Mat drawValuesPoints(const std::vector<::eds::mapping::Point2d> &points, const std::vector<double> &values, const int height, const int width, const std::string &method = "nn", const float s=0.5);

    cv::Mat drawValuesPointInfo(const std::vector<::eds::mapping::PointInfo> &points_info,
                        const int height, const int width, const std::string &method = "nn", const float s = 0.5);

    cv::Mat drawValuesPoints(std::vector<cv::Point2d>::const_iterator &points_begin, std::vector<cv::Point2d>::const_iterator &points_end,
                         std::vector<double>::const_iterator &values_begin, std::vector<double>::const_iterator &values_end,
                         const int height, const int width, const std::string &method, const float s);

    void visualize_gradient(const cv::Mat& src, cv::Mat& dst);

    void color_map(cv::Mat& input /*CV_32FC1*/, cv::Mat& dest, int color_map);

    double medianMat(cv::Mat &in);
    
    cv::Mat linspace(float x0, float x1, int n);

    void sortMatrixRowsByIndices(cv::InputArray _src, cv::InputArray _indices, cv::OutputArray _dst);

    cv::Mat sortMatrixRowsByIndices(cv::InputArray src, cv::InputArray indices);

    cv::Mat argsort(cv::InputArray _src, bool ascending=true);

    template <typename _Tp> static cv::Mat interp1_(const cv::Mat& X_, const cv::Mat& Y_, const cv::Mat& XI);

    cv::Mat interp1(cv::InputArray _x, cv::InputArray _Y, cv::InputArray _xi);

    void splitImageInPatches(const cv::Mat &image, const std::vector<cv::Point2d> &coord,
                            std::vector<cv::Mat> &patches, const uint16_t &patch_radius=7,
                            const int &border_type = cv::BORDER_DEFAULT, const uint8_t &border_value = 255);

    void splitImageInPatches(const cv::Mat &image, const std::vector<::eds::mapping::PointInfo> &points,
                            std::vector<cv::Mat> &patches, const uint16_t &patch_radius=7,
                            const int &border_type = cv::BORDER_DEFAULT, const uint8_t &border_value = 255);

    void pyramidPatches(const cv::Mat &patch, std::vector<cv::Mat> &pyr_patches, const size_t num_level = 3);

    void computeBundlePatches(const std::vector<cv::Mat> &patches, std::vector< std::vector<uchar> > &bundle_patches);

    void computeBundlePatches(const std::vector<cv::Mat> &patches, std::vector< std::vector<double> > &bundle_patches);

    Eigen::Vector2d kltTracker(cv::Mat &Ix, cv::Mat &Iy, cv::Mat &E);

    bool kltRefinement(const cv::Point2d &coord, Eigen::Vector2d &f, const cv::Mat &model_patch,
                    const cv::Mat &event_frame, const double &outlier_threshold,
                    const int &border_type = cv::BORDER_DEFAULT, const uint8_t &border_value = 255);

    cv::Mat flowArrowsOnImage(const cv::Mat &img, const std::vector<cv::Point2d> &coord,
                              const std::vector<Eigen::Vector2d> &flow,
                              const cv::Vec3d &color=cv::Vec3d(0.0, 255.0, 0.0),
                              const size_t &skip_amount = 10.0);

    cv::Point2d searchAlongEpiline(const cv::Size &size, const cv::Mat &img, const cv::Mat &patch, const cv::Vec3d &line,
                            const base::Transform3d &T_ef_kf, const cv::Point2d &norm_coord, const double &idepth,
                            const double &sigma, const cv::Mat &K, const ::eds::utils::SIMILARITY_MEASURE &method=ZNCC);

    cv::Mat viz(const cv::Mat &img, bool color=false);

    cv::Mat epilinesViz (const cv::Mat &img, const std::vector<cv::Vec3d> &lines, const size_t &skip_amount);

    cv::Point2d matchTemplate(const cv::Mat &img, const cv::Mat &templ, const int &match_method=cv::TM_SQDIFF);

    cv::Mat epilineImage(const cv::Mat &img, const cv::Vec3d &line, const uint16_t &height);

    void getCalibration(const ::eds::calib::CameraInfo &cam_info, int &w_out, int &h_out, Eigen::Matrix3f &K_out, Eigen::Vector4f &D_out, Eigen::Matrix3f &R_rect_out, Eigen::Matrix3f &K_ref_out);

    void getUndistortImage(const std::string &distortion_model, cv::Mat &input, cv::Mat &output, cv::Mat &K, cv::Mat &K_ref, cv::Mat &D);

    inline void compute_flow(const double &xp, const double &yp, const Eigen::Vector3d &vx, const Eigen::Vector3d &wx,
                            const double &idp, Eigen::Vector2d &result)
    {
        result[0] = (-idp*vx[0]) + (xp*idp*vx[2])
                        + (xp*yp*wx[0]) - (1.0+std::pow(xp, 2))*wx[1] + (yp*wx[2]);

        result[1] = (-idp*vx[1]) + (yp*idp*vx[2])
                        + (1.0+std::pow(yp, 2))*wx[0] - (xp*yp*wx[1]) - (xp*wx[2]);
    };

    inline double ncc(const cv::Mat &mat1, const cv::Mat &mat2)
    {
        cv::Mat patch1_squared = mat1.mul(mat1);
        cv::Mat patch2_squared = mat2.mul(mat2);
        cv::Mat patch12 = mat1.mul(mat2);

        return cv::sum(patch12)[0]/(cv::sqrt(cv::sum(patch1_squared)[0]) * cv::sqrt(cv::sum(patch2_squared)[0]));
    };

    inline double zncc(const cv::Mat &mat1, const cv::Mat &mat2)
    {
        cv::Mat patch1 = mat1 - cv::mean(mat1);
        cv::Mat patch2 = mat2 - cv::mean(mat2);

        cv::Mat patch1_squared = patch1.mul(patch1);
        cv::Mat patch2_squared = patch2.mul(patch2);
        cv::Mat patch12 = patch1.mul(patch2);

        return cv::sum(patch12)[0]/(cv::sqrt(cv::sum(patch1_squared)[0]) * cv::sqrt(cv::sum(patch2_squared)[0]));
    };

    inline double ssd(const cv::Mat &mat1, const cv::Mat &mat2)
    {
        return cv::norm(mat1, mat2, cv::NORM_L2SQR);
    };

    inline double nssd(const cv::Mat &mat1, const cv::Mat &mat2)
    {
        cv::Mat mat1_squared = mat1.mul(mat1);
        cv::Mat mat2_squared = mat2.mul(mat2);

        cv::Mat diff = (mat1 - mat2);
        cv::Mat diff_squared = diff.mul(diff);

        return cv::sum(diff_squared)[0]/(cv::sqrt(cv::sum(mat1_squared)[0] * cv::sum(mat2_squared)[0]));
    }

    inline double zssd(const cv::Mat &mat1, const cv::Mat &mat2)
    {
        cv::Mat patch1 = mat1 - cv::mean(mat1);
        cv::Mat patch2 = mat2 - cv::mean(mat2);
        cv::Mat diff_squared = (patch1-patch2).mul(patch1-patch2);
        return cv::sum(diff_squared)[0];
    }

    inline double sad(const cv::Mat &mat1, const cv::Mat &mat2)
    {
        cv::Mat abs_diff; cv::absdiff(mat1, mat2, abs_diff);
        return cv::sum(abs_diff)[0];
    }

    inline double zsad(const cv::Mat &mat1, const cv::Mat &mat2)
    {
        cv::Mat patch1 = mat1 - cv::mean(mat1);
        cv::Mat patch2 = mat2 - cv::mean(mat2);
        cv::Mat abs_diff; cv::absdiff(patch1, patch2, abs_diff);
        return cv::sum(abs_diff)[0];
    }

    template<typename Iter_T>
    double vectorNorm(Iter_T first, Iter_T last)
    {
        return sqrt(inner_product(first, last, first, 0.0L));
    };

    template<typename K, typename V>
    std::vector<std::pair<K, V>> mapToVector(const std::map<K, V> &map)
    {
        std::vector<std::pair<K, V>> v;
        v.resize(map.size());

        std::copy(map.begin(), map.end(), v.begin());

        return v;
    };

    template<typename K, typename V>
    std::pair< std::vector<K>, std::vector<V> > mapToVectors(const std::map<K, V> &map)
    {
        std::vector<K> keys;
        std::vector<V> values;

        std::transform(map.begin(), map.end(),
                std::back_inserter(keys),
                [](const std::pair<K, V> &p) {
                    return p.first;
                });
 
        std::transform(map.begin(), map.end(),
                std::back_inserter(values),
                [](const std::pair<K, V> &p) {
                    return p.second;
                });
 
        return std::make_pair(keys, values);
    };

    template<typename T>    
    void mean_std_vector(const std::vector<T> &vec, T &mu, T &std_dev)
    {
        const size_t sz = vec.size();
        if (sz == 1) {
            mu = vec[0]; std_dev=0.0;
            return;
        }

        // Calculate the mean
        mu = std::accumulate(vec.begin(), vec.end(), 0.0) / sz;

        // Now calculate the variance
        auto variance_func = [&mu, &sz](T accumulator, const T& val) {
            return accumulator + ((val - mu)*(val - mu) / (sz - 1));
        };

        std_dev = std::accumulate(vec.begin(), vec.end(), 0.0, variance_func);
    };

    inline bool keyframe_selection_occlusion(const ::base::Transform3d &delta_pose,
                                    const double &median_depth, const double &threshold = 0.20/*0.12 is VAMR course for normal VO*/)
    {
        std::cout<<"[KF SELECTION OCCLUSION] delta_trans.norm(): "<<delta_pose.translation().norm()<<" depth: "<<median_depth
        <<" criteria: "<<delta_pose.translation().norm()/median_depth <<" > "<<threshold<<std::endl;
        return delta_pose.translation().norm() / median_depth > threshold;
    };

    inline bool keyframe_selection_rotation(const ::base::Transform3d &delta_pose, const double &threshold = 0.174533/3.0 /*5deg*/)
    {
        base::Quaterniond q(delta_pose.rotation());
        Eigen::AngleAxisd angle_axis(q);
        std::cout<<"[KF SELECTION ROTATION] angle: "<< angle_axis.angle() * (180.00/M_PI) <<" threshold: "<<threshold * (180.00/M_PI)<<std::endl;
        return angle_axis.angle() > threshold;
    }

    inline bool keyframe_selection_translation(const ::base::Transform3d &delta_pose, const double &threshold = 0.05 /*5.0 cm*/)
    {
        double dist = delta_pose.translation().norm();
        std::cout<<"[KF SELECTION TRANSLATION] translation: "<< dist <<" threshold: "<<threshold<<std::endl;
        return dist > threshold;
    }

    template<typename T>
    T n_quantile_vector(std::vector<T> &vec, const int n)
    {
        std::nth_element(vec.begin(), vec.begin() + n, vec.end());
        return vec[n];
    };

    template<class bidiiter>
    bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random)
    {
        size_t left = std::distance(begin, end);
        while (num_random--) {
            bidiiter r = begin;
            std::advance(r, rand()%left);
            std::swap(*begin, *r);
            ++begin;
            --left;
        }
        return begin;
    };

    template<class T>
    inline T normPdf(const T x, const T mean, const T sigma)
    {
        T exponent = x - mean;
        exponent *= -exponent;
        exponent /= 2 * sigma * sigma;
        T result = std::exp(exponent);
        result /= sigma * std::sqrt(2 * M_PI);
        return result;
    };

    inline float bilinear(const float* img, const size_t w, const size_t h,
                        const float x, const float y)
    {
        if (x < 0 || y < 0)
            return -1.f;

        size_t px = (size_t)x,
                py = (size_t)y;

        if (px+1 >= w || py+1 >= h)
            return -1.f;


        // Load Pixel Values
        const float *addr = img + px + py*w;
        const float p1 = addr[0],
                    p2 = addr[1],
                    p3 = addr[w],
                    p4 = addr[w+1];

        // Compute Weights
        float fx = x - px,
                fy = y - py,
            fx1 = 1.0f - fx,
            fy1 = 1.0f - fy;

        float w1 = fx1 * fy1,
                w2 = fx  * fy1,
                w3 = fx1 * fy,
                w4 = fx  * fy;

        return p1*w1 + p2*w2 + p3*w3 + p4*w4;
    };


    inline float nn(const float* img, const size_t w, const size_t h,
                    const float x, const float y)
    {
        if (x < 0 || y < 0)
            return -1;

        size_t px = static_cast<size_t>(x + .5f),
                py = static_cast<size_t>(y + .5f);

        if (px+1 >= w || py+1 >= h)
            return -1;

        return img[px + py*w];
    };

    inline std::vector<double> imgToVector(const cv::Mat &img)
    {
        /** row major conversion to a 1D vector **/
        assert(img.channels() == 1);

        int n = img.rows;
        int m = img.cols;
        std::vector<double> res;
        for(int row = 0; row < m; row++)
        {
            for(int col = 0; col < n; col++)
            {
                 res.push_back(img.at<double>(row, col));
            }
        }
        return res;
    };

    inline void decomposeE(cv::Mat E, cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                            cv::Mat_<double> &t1, cv::Mat_<double> &t2)
    {
        cv::SVD svd(E, cv::SVD::MODIFY_A);
        cv::Matx33d W(0, -1, 0,
                    1, 0, 0,
                    0, 0, 1);
        cv::Matx33d Wt(0, 1, 0,
                    -1, 0, 0,
                    0, 0, 1);
        R1 = svd.u * cv::Mat(W) * svd.vt;
        R2 = svd.u * cv::Mat(Wt) * svd.vt;
        t1 = svd.u.col(2);
        t2 = -svd.u.col(2);
    };

    inline void drawPointsOnImage(const std::vector<cv::Point2d> &points, cv::Mat &img)
    {
        /** Asertion only in debug mode **/
        assert(img.cols > 0);
        assert(img.rows > 0);

        std::cout<<eds::utils::type2str(img.type())<<std::endl;
        img.convertTo(img, CV_8UC1, 255, 0);
        std::cout<<eds::utils::type2str(img.type())<<std::endl;
        cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
        std::cout<<eds::utils::type2str(img.type())<<std::endl;

        auto clip = [](const int n, const int lower, const int upper)
        {
            return std::max(lower, std::min(n, upper));
        };

        for (auto it_x = points.begin(); it_x != points.end(); ++it_x)
        {
            int x = clip(it_x->x, 0, img.cols - 1);
            int y = clip(it_x->y, 0, img.rows - 1);
            //std::cout<<"x :"<<x<<"y: "<<y<<std::endl;
            cv::Vec3b & color = img.at<cv::Vec3b>(y, x);
            color[0] = 0; color[1] = 255; color[2] = 0;
        }
        return;
    };

    inline void veloIntegration(const ::base::Transform3d &T_kf_ef, std::vector<::base::samples::Twist> &data,
                               Eigen::Matrix<double, 4, 4> &oldomega4, ::base::Vector3d &px, ::base::Quaterniond &qx)
    {
        /** Get the current tracker transformation **/
        ::base::Transform3d T_delta (::base::Transform3d::Identity());

        std::cout<<"** [IMU_INTEGRATION] size[" <<data.size() <<"] ";
        /** Lienar and angular velocity **/
        if (data.size() > 0)
        {
            auto it_start = data.begin();
            auto it_end = data.begin();
            std::advance(it_end, data.size()-1);
            double dt = (it_end->time.toSeconds() - it_start->time.toSeconds());
            std::cout<<"** [IMU_INTEGRATION] delta_t: " <<dt <<std::endl;

            ::base::Vector3d linvelo = ::base::Vector3d::Zero();
            ::base::Vector3d angvelo = ::base::Vector3d::Zero();
            for (auto &it : data)
            {
                angvelo += it.angular;
                linvelo += it.linear; 
            }

            /** Mean values in camera frame (without gravity) **/
            angvelo /= data.size(); linvelo /= data.size();
            Eigen::Vector4d quat = Eigen::Vector4d::Zero();
            Eigen::Matrix<double, 4, 4> omega4;
            omega4 << 0,-angvelo(0), -angvelo(1), -angvelo(2),
                    angvelo(0), 0, angvelo(2), -angvelo(1),
                    angvelo(1), -angvelo(2), 0, angvelo(0),
                    angvelo(2), angvelo(1), -angvelo(0), 0;

            quat = (Eigen::Matrix<double,4,4>::Identity() +(0.75 * omega4 *dt)-(0.25 * oldomega4 * dt) -
            ((1.0/6.0) * angvelo.squaredNorm() * pow(dt,2) *  Eigen::Matrix<double,4,4>::Identity()) -
            ((1.0/24.0) * omega4 * oldomega4 * pow(dt,2)) - ((1.0/48.0) * angvelo.squaredNorm() * omega4 * pow(dt,3))) * quat;

            Eigen::Quaterniond q4;
            q4.w() = quat(0);
            q4.x() = quat(1);
            q4.y() = quat(2);
            q4.z() = quat(3);
            q4.normalize();

            oldomega4 = omega4;
            T_delta.rotate(q4);
            T_delta.translation() = linvelo * dt;
        }
        std::cout<<"** [IMU_INTEGRATION] T_DELTA:\n"<< T_delta.matrix()<<std::endl;

        base::Transform3d T_result = (T_kf_ef * T_delta).inverse();
        px = T_result.translation();
        qx = ::base::Quaterniond(T_result.rotation());

        /** Remove data in buffer **/
        data.clear();
    };

    //from here https://stackoverflow.com/questions/55178857/convert-a-floating-point-number-to-rgb-vector
    inline cv::Vec3b valueToColor(const double &value)
    {
        double H = value * 2.0f/3.0f;
        double R = fabs(H * 6.0 - 3.0) - 1.0;
        double G = 2.0f - fabs(H * 6.0 - 2.0);
        double B = 2.0f - fabs(H * 6.0 - 4.0); 
        return cv::Vec3b(
            std::max(0.0, std::min(1.0, R)) * 255.0,
            std::max(0.0, std::min(1.0, G)) * 255.0,
            std::max(0.0, std::min(1.0, B)) * 255.0);
    };

    inline void tanhMat(cv::Mat &img)
    {
        double min, max; cv::minMaxLoc(img, &min, &max);
        for(int x=0; x<img.cols; ++x)
        {
            for (int y=0; y<img.rows; ++y)
            {
                img.at<double>(y, x) = ((max - min)/2.0) * tanh(img.at<double>(y, x));
            }
        }
    };

    inline double expWeight(const double &idx, const double &window_size)
    {
        double value = (idx - (window_size/2)) / (window_size/6.0);
        return std::exp(-0.5*value*value);
    };

    inline bool file_exist (const std::string& name)
    {
        struct stat buffer;
        return (stat (name.c_str(), &buffer) == 0); 
    };

    inline void cleanEventFrame(cv::Mat event_frame, const double &thrs = 0.3)
    {
        for (int i=0; i<event_frame.rows; i++)
        {
            for (int j=0; j<event_frame.cols; j++)
            {
                if (abs(event_frame.at<double>(i, j)) < thrs)
                {
                    event_frame.at<double>(i,j) = 0.00;
                }
            }
        }
    }

} //utils namespace
} // end namespace

#endif // _EDS_UTILS_HPP_