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

#include "Utils.hpp"
#include <opencv2/core/eigen.hpp>

namespace eds { namespace utils {

std::string type2str(int type)
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

cv::Mat drawValuesPoints(const std::vector<cv::Point2d> &points, const std::vector<int8_t> &values, const int height, const int width, const std::string &method, const float s, const bool &use_exp_weights)
{
    /** Asertion only in debug mode **/
    assert(height > 0);
    assert(width > 0);
    assert(values.size() == points.size());
    assert((method.compare("nn") == 0) || (method.compare("bilinear") == 0));

    /** Mat image **/
    cv::Mat img = cv::Mat(height, width, CV_64FC1, cv::Scalar(0));
    cv::Size size = img.size();

    auto clip = [](const int n, const int lower, const int upper)
    {
        return std::max(lower, std::min(n, upper));
    };

    double idx = 0; uint32_t window_size = values.size();
    auto it_x = points.begin();
    auto it_p = values.begin();
    while(it_x != points.end() && it_p != values.end())
    {
        double weight = (use_exp_weights)? eds::utils::expWeight(static_cast<double>(idx/window_size), 1.0) : 1.0;
        if (method.compare("nn") == 0)
        {
            cv::Point2i x_int = *it_x;
            //std::cout<<*it_x<<std::endl;
            x_int.x = clip(x_int.x, 0, size.width - 1);
            x_int.y = clip(x_int.y, 0, size.height - 1);

            img.at<double>(x_int) += weight * (*it_p);

        }
        else if (method.compare("bilinear") == 0)
        {   
            int x0 = floor(it_x->x);
            int y0 = floor(it_x->y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            //std::cout<<"x0: "<< x0<<" y0: "<<y0<<" x1: "<< x1<<" y1: "<<y1 <<" pol: "<<std::to_string(*it_p)<<std::endl;
            /** compute the voting weights. Note: assign weight 0 if the point is out of the image **/
            double wa, wb, wc, wd;
            wa = ((x0 < size.width) && (y0 < size.height) && (x0 >= 0) && (y0 >= 0))? (x1 - it_x->x) * (y1 - it_x->y) : 0.0;
            wb = ((x0 < size.width) && (y1 < size.height) && (x0 >= 0) && (y1 >= 0))? (x1 - it_x->x) * (it_x->y - y0) : 0.0;
            wc = ((x1 < size.width) && (y0 < size.height) && (x1 >= 0) && (y0 >= 0))? (it_x->x - x0) * (y1 - it_x->y) : 0.0;
            wd = ((x1 < size.width) && (y1 < size.height) && (x1 >= 0) && (y1 >= 0))? (it_x->x - x0) * (it_x->y - y0) : 0.0;

            x0 = clip(x0, 0, size.width - 1);
            x1 = clip(x1, 0, size.width - 1);
            y0 = clip(y0, 0, size.height - 1);
            y1 = clip(y1, 0, size.height - 1);
            //std::cout<<"wa: "<<wa<<" wb: "<<wb<<" wc: "<<wc<<" wd: "<<wd<<std::endl;

            img.at<double>(y0, x0) += weight * wa * (*it_p);
            img.at<double>(y1, x0) += weight * wb * (*it_p);
            img.at<double>(y0, x1) += weight * wc * (*it_p);
            img.at<double>(y1, x1) += weight * wd * (*it_p);
        }
        ++it_x;
        ++it_p;
        ++idx;
    }

    if(s > 0)
    {
        /** Kernal size depending on image size **/
        int k_h = int((1.7 * 180)/100);
        int k_w = int((1.25 * 240)/100);
        cv::GaussianBlur(img, img, cv::Size(k_w, k_h), s, s);
    }

    return img;
}

cv::Mat drawValuesPoints(const std::vector<cv::Point2d> &points, const std::vector<double> &values, const int height, const int width, const std::string &method, const float s)
{
    /** Asertion only in debug mode **/
    assert(height > 0);
    assert(width > 0);
    assert(values.size() == points.size());
    assert((method.compare("nn") == 0) || (method.compare("bilinear") == 0));

    /** Mat image **/
    cv::Mat img = cv::Mat(height, width, CV_64FC1, cv::Scalar(0));
    cv::Size size = img.size();

    auto clip = [](const int n, const int lower, const int upper)
    {
        return std::max(lower, std::min(n, upper));
    };

    auto it_x = points.begin();
    auto it_p = values.begin();
    while(it_x != points.end() && it_p != values.end())
    {
        if (method.compare("nn") == 0)
        {
            cv::Point2i x_int = *it_x;
            //std::cout<<*it_x<<std::endl;
            x_int.x = clip(x_int.x, 0, size.width - 1);
            x_int.y = clip(x_int.y, 0, size.height - 1);

            img.at<double>(x_int) += (*it_p);
 
        }
        else if (method.compare("bilinear") == 0)
        {   
            int x0 = floor(it_x->x);
            int y0 = floor(it_x->y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            //std::cout<<"x0: "<< x0<<" y0: "<<y0<<" x1: "<< x1<<" y1: "<<y1 <<" pol: "<<std::to_string(*it_p)<<std::endl;
            /** compute the voting weights. Note: assign weight 0 if the point is out of the image **/
            double wa, wb, wc, wd;
            wa = ((x0 < size.width) && (y0 < size.height) && (x0 >= 0) && (y0 >= 0))? (x1 - it_x->x) * (y1 - it_x->y) : 0.0;
            wb = ((x0 < size.width) && (y1 < size.height) && (x0 >= 0) && (y1 >= 0))? (x1 - it_x->x) * (it_x->y - y0) : 0.0;
            wc = ((x1 < size.width) && (y0 < size.height) && (x1 >= 0) && (y0 >= 0))? (it_x->x - x0) * (y1 - it_x->y) : 0.0;
            wd = ((x1 < size.width) && (y1 < size.height) && (x1 >= 0) && (y1 >= 0))? (it_x->x - x0) * (it_x->y - y0) : 0.0;

            x0 = clip(x0, 0, size.width - 1);
            x1 = clip(x1, 0, size.width - 1);
            y0 = clip(y0, 0, size.height - 1);
            y1 = clip(y1, 0, size.height - 1);
            //std::cout<<"wa: "<<wa<<" wb: "<<wb<<" wc: "<<wc<<" wd: "<<wd<<std::endl;

            img.at<double>(y0, x0) += wa * (*it_p);
            img.at<double>(y1, x0) += wb * (*it_p);
            img.at<double>(y0, x1) += wc * (*it_p);
            img.at<double>(y1, x1) += wd * (*it_p);
        }
        ++it_x;
        ++it_p;
    }

    if(s > 0)
    {
        /** Kernal size depending on image size **/
        int k_h = int((1.7 * 180)/100);
        int k_w = int((1.25 * 240)/100);
        cv::GaussianBlur(img, img, cv::Size(k_w, k_h), s, s);
    }

    return img;
}

cv::Mat drawValuesPoints(const std::vector<::eds::mapping::Point2d> &points, const std::vector<double> &values, const int height, const int width, const std::string &method, const float s)
{
    /** Asertion only in debug mode **/
    assert(height > 0);
    assert(width > 0);
    assert(values.size() == points.size());
    assert((method.compare("nn") == 0) || (method.compare("bilinear") == 0));

    /** Mat image **/
    cv::Mat img = cv::Mat(height, width, CV_64FC1, cv::Scalar(0));
    cv::Size size = img.size();

    auto clip = [](const int n, const int lower, const int upper)
    {
        return std::max(lower, std::min(n, upper));
    };

    auto it_x = points.begin();
    auto it_p = values.begin();
    while(it_x != points.end() && it_p != values.end())
    {
        if (method.compare("nn") == 0)
        {
            cv::Point2i x_int = cv::Point2d(*it_x);
            //std::cout<<*it_x<<std::endl;
            x_int.x = clip(x_int.x, 0, size.width - 1);
            x_int.y = clip(x_int.y, 0, size.height - 1);

            img.at<double>(x_int) += (*it_p);
        }
        else if (method.compare("bilinear") == 0)
        {
            int x0 = floor((*it_x)[0]);
            int y0 = floor((*it_x)[1]);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            //std::cout<<"x0: "<< x0<<" y0: "<<y0<<" x1: "<< x1<<" y1: "<<y1 <<" pol: "<<std::to_string(*it_p)<<std::endl;
            /** compute the voting weights. Note: assign weight 0 if the point is out of the image **/
            double wa, wb, wc, wd;
            wa = ((x0 < size.width) && (y0 < size.height) && (x0 >= 0) && (y0 >= 0))? (x1 - (*it_x)[0]) * (y1 - (*it_x)[1]) : 0.0;
            wb = ((x0 < size.width) && (y1 < size.height) && (x0 >= 0) && (y1 >= 0))? (x1 - (*it_x)[0]) * ((*it_x)[1] - y0) : 0.0;
            wc = ((x1 < size.width) && (y0 < size.height) && (x1 >= 0) && (y0 >= 0))? ((*it_x)[0] - x0) * (y1 - (*it_x)[1]) : 0.0;
            wd = ((x1 < size.width) && (y1 < size.height) && (x1 >= 0) && (y1 >= 0))? ((*it_x)[0] - x0) * ((*it_x)[1] - y0) : 0.0;

            x0 = clip(x0, 0, size.width - 1);
            x1 = clip(x1, 0, size.width - 1);
            y0 = clip(y0, 0, size.height - 1);
            y1 = clip(y1, 0, size.height - 1);
            //std::cout<<"wa: "<<wa<<" wb: "<<wb<<" wc: "<<wc<<" wd: "<<wd<<std::endl;

            img.at<double>(y0, x0) += wa * (*it_p);
            img.at<double>(y1, x0) += wb * (*it_p);
            img.at<double>(y0, x1) += wc * (*it_p);
            img.at<double>(y1, x1) += wd * (*it_p);
        }
        ++it_x;
        ++it_p;
    }

    if(s > 0)
    {
        /** Kernal size depending on image size **/
        int k_h = int((1.7 * 180)/100);
        int k_w = int((1.25 * 240)/100);
        cv::GaussianBlur(img, img, cv::Size(k_w, k_h), s, s);
    }

    return img;
}

cv::Mat drawValuesPointInfo(const std::vector<::eds::mapping::PointInfo> &points_info,
                        const int height, const int width, const std::string &method, const float s)
{
    /** Asertion only in debug mode **/
    assert(height > 0);
    assert(width > 0);
    assert((method.compare("nn") == 0) || (method.compare("bilinear") == 0));

    /** Mat image **/
    cv::Mat img = cv::Mat(height, width, CV_64FC1, cv::Scalar(0));
    cv::Size size = img.size();

    auto clip = [](const int n, const int lower, const int upper)
    {
        return std::max(lower, std::min(n, upper));
    };

    auto it_p = points_info.begin();
    while(it_p != points_info.end())
    {
        if (method.compare("nn") == 0)
        {
            cv::Point2i x_int((int)it_p->coord[0], (int)it_p->coord[1]);
            //std::cout<<*it_x<<std::endl;
            x_int.x = clip(x_int.x, 0, size.width - 1);
            x_int.y = clip(x_int.y, 0, size.height - 1);

            img.at<double>(x_int) += (*it_p).gradient;

        }
        else if (method.compare("bilinear") == 0)
        {   
            int x0 = floor(it_p->coord[0]);
            int y0 = floor(it_p->coord[1]);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            //std::cout<<"x0: "<< x0<<" y0: "<<y0<<" x1: "<< x1<<" y1: "<<y1 <<" pol: "<<std::to_string(*it_p)<<std::endl;
            /** compute the voting weights. Note: assign weight 0 if the point is out of the image **/
            double wa, wb, wc, wd;
            wa = ((x0 < size.width) && (y0 < size.height) && (x0 >= 0) && (y0 >= 0))? (x1 - it_p->coord[0] ) * (y1 - it_p->coord[1]) : 0.0;
            wb = ((x0 < size.width) && (y1 < size.height) && (x0 >= 0) && (y1 >= 0))? (x1 - it_p->coord[0] ) * (it_p->coord[1] - y0) : 0.0;
            wc = ((x1 < size.width) && (y0 < size.height) && (x1 >= 0) && (y0 >= 0))? (it_p->coord[0] - x0) * (y1 - it_p->coord[1]) : 0.0;
            wd = ((x1 < size.width) && (y1 < size.height) && (x1 >= 0) && (y1 >= 0))? (it_p->coord[0] - x0) * (it_p->coord[1] - y0) : 0.0;

            x0 = clip(x0, 0, size.width - 1);
            x1 = clip(x1, 0, size.width - 1);
            y0 = clip(y0, 0, size.height - 1);
            y1 = clip(y1, 0, size.height - 1);
            //std::cout<<"wa: "<<wa<<" wb: "<<wb<<" wc: "<<wc<<" wd: "<<wd<<std::endl;

            img.at<double>(y0, x0) += wa * (*it_p).gradient;
            img.at<double>(y1, x0) += wb * (*it_p).gradient;
            img.at<double>(y0, x1) += wc * (*it_p).gradient;
            img.at<double>(y1, x1) += wd * (*it_p).gradient;
        }
        ++it_p;
    }

    if(s > 0)
    {
        /** Kernal size depending on image size **/
        int k_h = int((1.7 * 180)/100);
        int k_w = int((1.25 * 240)/100);
        cv::GaussianBlur(img, img, cv::Size(k_w, k_h), s, s);
    }

    return img;
}

cv::Mat drawValuesPoints(std::vector<cv::Point2d>::const_iterator &points_begin,
                         std::vector<cv::Point2d>::const_iterator &points_end,
                         std::vector<double>::const_iterator &values_begin,
                         std::vector<double>::const_iterator &values_end,
                         const int height, const int width, const std::string &method, const float s)
{
    /** Asertion only in debug mode **/
    assert(height > 0);
    assert(width > 0);
    assert(std::distance(points_begin, points_end) == std::distance(values_begin, values_end));
    assert((method.compare("nn") == 0) || (method.compare("bilinear") == 0));

    /** Mat image **/
    cv::Mat img = cv::Mat(height, width, CV_64FC1, cv::Scalar(0));
    cv::Size size = img.size();

    auto clip = [](const int n, const int lower, const int upper)
    {
        return std::max(lower, std::min(n, upper));
    };

    while(points_begin != points_end && values_begin != values_end)
    {
        if (method.compare("nn") == 0)
        {
            cv::Point2i x_int = *points_begin;
            //std::cout<<*it_x<<std::endl;
            x_int.x = clip(x_int.x, 0, size.width - 1);
            x_int.y = clip(x_int.y, 0, size.height - 1);

            img.at<double>(x_int) += (*values_begin);

        }
        else if (method.compare("bilinear") == 0)
        {
            int x0 = floor(points_begin->x);
            int y0 = floor(points_begin->y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            //std::cout<<"x0: "<< x0<<" y0: "<<y0<<" x1: "<< x1<<" y1: "<<y1 <<" pol: "<<std::to_string(*it_p)<<std::endl;
            /** compute the voting weights. Note: assign weight 0 if the point is out of the image **/
            double wa, wb, wc, wd;
            wa = ((x0 < size.width) && (y0 < size.height) && (x0 >= 0) && (y0 >= 0))? (x1 - points_begin->x) * (y1 - points_begin->y) : 0.0;
            wb = ((x0 < size.width) && (y1 < size.height) && (x0 >= 0) && (y1 >= 0))? (x1 - points_begin->x) * (points_begin->y - y0) : 0.0;
            wc = ((x1 < size.width) && (y0 < size.height) && (x1 >= 0) && (y0 >= 0))? (points_begin->x - x0) * (y1 - points_begin->y) : 0.0;
            wd = ((x1 < size.width) && (y1 < size.height) && (x1 >= 0) && (y1 >= 0))? (points_begin->x - x0) * (points_begin->y - y0) : 0.0;

            x0 = clip(x0, 0, size.width - 1);
            x1 = clip(x1, 0, size.width - 1);
            y0 = clip(y0, 0, size.height - 1);
            y1 = clip(y1, 0, size.height - 1);
            //std::cout<<"wa: "<<wa<<" wb: "<<wb<<" wc: "<<wc<<" wd: "<<wd<<std::endl;

            img.at<double>(y0, x0) += wa * (*values_begin);
            img.at<double>(y1, x0) += wb * (*values_begin);
            img.at<double>(y0, x1) += wc * (*values_begin);
            img.at<double>(y1, x1) += wd * (*values_begin);
        }
        ++points_begin;
        ++values_begin;
    }

    if(s > 0)
    {
        /** Kernal size depending on image size **/
        int k_h = int((1.7 * 180)/100);
        int k_w = int((1.25 * 240)/100);
        cv::GaussianBlur(img, img, cv::Size(k_w, k_h), s, s);
    }

    return img;
}

void visualize_gradient(const cv::Mat& src, cv::Mat& dst)
{
    // split input into x and y gradients
    std::vector<cv::Mat> gradients;
    cv::split(src, gradients);

    // convert to float (cvtColor cannot handle floats :()
    gradients[0].convertTo(gradients[0], CV_32FC1);
    gradients[1].convertTo(gradients[1], CV_32FC1);

    // calculate angle (in degree) and magnitude
    cv::Mat angle; // 0 to 360
    cv::Mat magnitude; // 0 to ???
    cv::cartToPolar(gradients[0], gradients[1], magnitude, angle, true);

    // scale magnitude to [0,1]
    double max_mag, min_mag;
    cv::minMaxLoc(magnitude, &min_mag, &max_mag);
    magnitude /= max_mag;

    // Hue: 0-360 (float)
    // Saturation: 0-1 (float)
    // Value: 0-1 (float)

    // initialize 'value' with 1
    cv::Mat value(src.size(), CV_32FC1, cv::Scalar(1));

    cv::Mat hsv_in[] = { angle, magnitude, value }; // Hue, Saturation, Value
    cv::Mat hsv;
    cv::merge(hsv_in, 3, hsv);

    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
}

void color_map(cv::Mat& input /*CV_32FC1*/, cv::Mat& dest, int color_map)
{
    int num_bar_w=20;
    int color_bar_w=10;
    int vline=10;

    cv::Mat win_mat(cv::Size(input.cols+num_bar_w+num_bar_w+vline, input.rows), CV_8UC3, cv::Scalar(255,255,255));

    //Input image to
    double Min, Max;
    cv::minMaxLoc(input, &Min, &Max);
    int max_int=ceil(Max);

    std::cout<<" [COLOR_MAP] Min "<< Min<<" Max "<< Max<<std::endl;

    input.convertTo(input,CV_8UC3,255.0/(Max-Min),-255.0*Min/(Max-Min));
    input.convertTo(input, CV_8UC3);

    cv::Mat M;
    cv::applyColorMap(input, M, color_map);

    M.copyTo(win_mat(cv::Rect(  0, 0, input.cols, input.rows)));

    //Scale
    cv::Mat num_window(cv::Size(num_bar_w, input.rows), CV_8UC3, cv::Scalar(255,255,255));
    for(int i=0; i<=max_int; i++){
        int j=i*input.rows/max_int;
        //TO-DO: does not work
        cv::putText(num_window, std::to_string(i), cv::Point(5, num_window.rows-j-5),cv::FONT_HERSHEY_SIMPLEX, 0.6 , cv::Scalar(0,0,0), 1 , 2 , false);
    }

    //color bar
    cv::Mat color_bar(cv::Size(color_bar_w, input.rows), CV_8UC3, cv::Scalar(255,255,255));
    cv::Mat cb;
    for(int i=0; i<color_bar.rows; i++)
    {
        for(int j=0; j<color_bar_w; j++)
        {
            int v=255-255*i/color_bar.rows;
            color_bar.at<cv::Vec3b>(i,j)=cv::Vec3b(v,v,v);
        }
    }

    color_bar.convertTo(color_bar, CV_8UC3);
    cv::applyColorMap(color_bar, cb, color_map);
    num_window.copyTo(win_mat(cv::Rect(input.cols+vline+color_bar_w, 0, num_bar_w, input.rows)));
    cb.copyTo(win_mat(cv::Rect(input.cols+vline, 0, color_bar_w, input.rows)));
    dest=win_mat.clone();
}

double medianMat(cv::Mat &_in)
{
    cv::Mat in = _in.clone();
    in = in.reshape(0,1);
    std::vector<double> vec_from_mat;
    in.copyTo(vec_from_mat);
    std::nth_element(vec_from_mat.begin(), vec_from_mat.begin() + vec_from_mat.size() / 2, vec_from_mat.end());
    return vec_from_mat[vec_from_mat.size() / 2];
}

cv::Mat linspace(float x0, float x1, int n)
{
    cv::Mat pts(n, 1, CV_32FC1);
    float step = (x1-x0)/(n-1);
    for(int i = 0; i < n; i++)
        pts.at<float>(i,0) = x0+i*step;
    return pts;
}

//------------------------------------------------------------------------------
// cv::sortMatrixRowsByIndices
//------------------------------------------------------------------------------
void sortMatrixRowsByIndices(cv::InputArray _src, cv::InputArray _indices, cv::OutputArray _dst)
{
    if(_indices.getMat().type() != CV_32SC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "cv::sortRowsByIndices only works on integer indices!");
    cv::Mat src = _src.getMat();
    std::vector<int> indices = _indices.getMat();
    _dst.create(src.rows, src.cols, src.type());
    cv::Mat dst = _dst.getMat();
    for(size_t idx = 0; idx < indices.size(); idx++) {
        cv::Mat originalRow = src.row(indices[idx]);
        cv::Mat sortedRow = dst.row((int)idx);
        originalRow.copyTo(sortedRow);
    }
}

cv::Mat sortMatrixRowsByIndices(cv::InputArray src, cv::InputArray indices)
{
    cv::Mat dst;
    sortMatrixRowsByIndices(src, indices, dst);
    return dst;
}


cv::Mat argsort(cv::InputArray _src, bool ascending)
{
    cv::Mat src = _src.getMat();
    if (src.rows != 1 && src.cols != 1)
        CV_Error(cv::Error::StsBadArg, "cv::argsort only sorts 1D matrices.");
    int flags = cv::SORT_EVERY_ROW | (ascending ? cv::SORT_ASCENDING : cv::SORT_DESCENDING);
    cv::Mat sorted_indices;
    sortIdx(src.reshape(1,1),sorted_indices,flags);
    return sorted_indices;
}

template <typename _Tp> static
cv::Mat interp1_(const cv::Mat& X_, const cv::Mat& Y_, const cv::Mat& XI)
{
    int n = XI.rows;
    // sort input table
    std::vector<int> sort_indices = argsort(X_);

    cv::Mat X = sortMatrixRowsByIndices(X_,sort_indices);
    cv::Mat Y = sortMatrixRowsByIndices(Y_,sort_indices);
    // interpolated values
    cv::Mat yi = cv::Mat::zeros(XI.size(), XI.type());
    for(int i = 0; i < n; i++) {
        int low = 0;
        int high = X.rows - 1;
        // set bounds
        if(XI.at<_Tp>(i,0) < X.at<_Tp>(low, 0))
            high = 1;
        if(XI.at<_Tp>(i,0) > X.at<_Tp>(high, 0))
            low = high - 1;
        // binary search
        while((high-low)>1) {
            const int c = low + ((high - low) >> 1);
            if(XI.at<_Tp>(i,0) > X.at<_Tp>(c,0)) {
                low = c;
            } else {
                high = c;
            }
        }
        // linear interpolation
        yi.at<_Tp>(i,0) += Y.at<_Tp>(low,0)
        + (XI.at<_Tp>(i,0) - X.at<_Tp>(low,0))
        * (Y.at<_Tp>(high,0) - Y.at<_Tp>(low,0))
        / (X.at<_Tp>(high,0) - X.at<_Tp>(low,0));
    }
    return yi;
}

cv::Mat interp1(cv::InputArray _x, cv::InputArray _Y, cv::InputArray _xi)
{
    // get matrices
    cv::Mat x = _x.getMat();
    cv::Mat Y = _Y.getMat();
    cv::Mat xi = _xi.getMat();
    // check types & alignment
    CV_Assert((x.type() == Y.type()) && (Y.type() == xi.type()));
    CV_Assert((x.cols == 1) && (x.rows == Y.rows) && (x.cols == Y.cols));
    // call templated interp1
    switch(x.type()) {
        case CV_8SC1: return interp1_<char>(x,Y,xi); break;
        case CV_8UC1: return interp1_<unsigned char>(x,Y,xi); break;
        case CV_16SC1: return interp1_<short>(x,Y,xi); break;
        case CV_16UC1: return interp1_<unsigned short>(x,Y,xi); break;
        case CV_32SC1: return interp1_<int>(x,Y,xi); break;
        case CV_32FC1: return interp1_<float>(x,Y,xi); break;
        case CV_64FC1: return interp1_<double>(x,Y,xi); break;
    }
    CV_Error(cv::Error::StsUnsupportedFormat, "");
    return cv::Mat::zeros(xi.size(), xi.type());
}


void splitImageInPatches(const cv::Mat &image, const std::vector<cv::Point2d> &coord,
                        std::vector<cv::Mat> &patches, const uint16_t &patch_radius,
                        const int &border_type, const uint8_t &border_value)
{
    /** Patch size based on radius **/
    double patch_size = 2 * patch_radius + 1;

    if (image.channels() > 1.0)
            throw std::runtime_error("[ERROR] SPLIT IMAGE IN PATCHES: image number of channels > 1");

    /** Create a bigger model image (padding) **/
    cv::Mat img;
    cv::copyMakeBorder(image, img, patch_radius, patch_radius, patch_radius, patch_radius, border_type, border_value);

    patches.clear();
    /** Get the pathes for the gradients **/
    for (auto it=coord.begin(); it!=coord.end(); ++it)
    {
        cv::Point2d p (it->x, it->y);
        cv::Rect grid_rect(p.x, p.y, patch_size, patch_size);
        /** better to create a variable for continuous (isContinuous) data in opencv **/
        cv::Mat patch = img(grid_rect);
        patches.push_back(patch);
    }
    std::cout<<"[UTILS] number of patches: "<<patches.size()<<" of size "<<patch_size<<" x "<<patch_size<<std::endl;
}

void splitImageInPatches(const cv::Mat &image, const std::vector<::eds::mapping::PointInfo> &point,
                        std::vector<cv::Mat> &patches, const uint16_t &patch_radius,
                        const int &border_type, const uint8_t &border_value)
{
    /** Patch size based on radius **/
    double patch_size = 2 * patch_radius + 1;

    if (image.channels() > 1.0)
            throw std::runtime_error("[ERROR] SPLIT IMAGE IN PATCHES POINT INFO: image number of channels > 1");

    /** Create a bigger model image (padding) **/
    cv::Mat img;
    cv::copyMakeBorder(image, img, patch_radius, patch_radius, patch_radius, patch_radius, border_type, border_value);

    patches.clear();
    /** Get the pathes for the gradients **/
    for (auto it=point.begin(); it!=point.end(); ++it)
    {
        cv::Point2d p = cv::Point2d(it->coord);
        cv::Rect grid_rect(p.x, p.y, patch_size, patch_size);
        /** better to create a variable for continuous (isContinuous) data in opencv **/
        cv::Mat patch = img(grid_rect);
        patches.push_back(patch);
    }
    std::cout<<"[UTILS] number of patches: "<<patches.size()<<" of size "<<patch_size<<" x "<<patch_size<<std::endl;
}

void pyramidPatches(const cv::Mat &patch, std::vector<cv::Mat> &pyr_patches, const size_t num_level)
{
    pyr_patches.clear();
    pyr_patches.push_back(patch);
    for (size_t i=1; i<num_level; ++i)
    {
        cv::Mat level_i;
        double scale = std::pow(2.0, static_cast<double>(i));
        cv::pyrDown(pyr_patches[i-1], level_i, cv::Size(patch.cols/(int)scale, patch.rows/(int)scale));
        pyr_patches.push_back(level_i);
    }
}

void computeBundlePatches(const std::vector<cv::Mat> &patches, std::vector< std::vector<uchar> > &bundle_patches)
{
    /** Clear bundle patches **/
    bundle_patches.clear();

    /** Get the Bundle patch for each existing patch **/
    for (auto it:patches)
    {
        /** The patch has to be squared and be at least 5x5 **/
        if ((it.cols != it.rows) || (it.rows < 5) || (it.channels() > 1))
        {
            throw std::runtime_error("[ERROR] COMPUTE BUNDLES PATCHES UCHAR: patch size at least 5x5 and one single channel");
        }
        cv::Point center(it.cols/2, it.rows/2); //x-y coord of center point
        //std::cout<<"center patch point uchar: "<<center<<std::endl;

        /** Compute DSO patch pattern **/
        std::vector<uchar> dso_patch;
        dso_patch.push_back(it.at<uchar>(center.y+2, center.x));//y-x point1
        dso_patch.push_back(it.at<uchar>(center.y+1, center.x-1));//y-x point2
        dso_patch.push_back(it.at<uchar>(center.y, center.x-2));//y-x point3
        dso_patch.push_back(it.at<uchar>(center.y-1, center.x-1));//y-x point4
        dso_patch.push_back(it.at<uchar>(center.y-2, center.x));//y-x point5
        dso_patch.push_back(it.at<uchar>(center.y-1, center.x+1));//y-x point6
        dso_patch.push_back(it.at<uchar>(center.y, center.x+2));//y-x point7
        dso_patch.push_back(it.at<uchar>(center.y, center.x));//y-x point8
        bundle_patches.push_back(dso_patch);
    }
}

void computeBundlePatches(const std::vector<cv::Mat> &patches, std::vector< std::vector<double> > &bundle_patches)
{
    /** Clear bundle patches **/
    bundle_patches.clear();

    /** Get the Bundle patch for each existing patch **/
    for (auto it:patches)
    {
        /** The patch has to be squared and be at least 5x5 **/
        if ((it.cols != it.rows) || (it.rows < 5) || (it.channels() > 1))
        {
            throw std::runtime_error("[ERROR] COMPUTE BUNDLES PATCHES DOUBLE: patch size at least 5x5 and one single channel");
        }
        cv::Point center(it.cols/2, it.rows/2); //x-y coord of center point
        //std::cout<<"center patch point double: "<<center<<std::endl;

        /** Compute DSO patch pattern **/
        std::vector<double> dso_patch;
        dso_patch.push_back(it.at<double>(center.y+2, center.x));//y-x point1
        dso_patch.push_back(it.at<double>(center.y+1, center.x-1));//y-x point2
        dso_patch.push_back(it.at<double>(center.y, center.x-2));//y-x point3
        dso_patch.push_back(it.at<double>(center.y-1, center.x-1));//y-x point4
        dso_patch.push_back(it.at<double>(center.y-2, center.x));//y-x point5
        dso_patch.push_back(it.at<double>(center.y-1, center.x+1));//y-x point6
        dso_patch.push_back(it.at<double>(center.y, center.x+2));//y-x point7
        dso_patch.push_back(it.at<double>(center.y, center.x));//y-x point8
        bundle_patches.push_back(dso_patch);
    }
}

Eigen::Vector2d kltTracker(cv::Mat &Ix, cv::Mat &Iy, cv::Mat &It)
{
    assert(Ix.size() == Iy.size());
    assert(It.size() == Ix.size());
    assert(Ix.channels() == 1);
    assert(Iy.channels() == 1);
    assert(It.channels() == 1);

    cv::Mat Ixx = Ix.mul(Ix);
    cv::Mat Iyy = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);
    cv::Mat Ixt = Ix.mul(It);
    cv::Mat Iyt = Iy.mul(It);

    double s_Ixx = cv::sum(Ixx)[0];
    double s_Iyy = cv::sum(Iyy)[0];
    double s_Ixy = cv::sum(Ixy)[0];
    double s_Ixt = cv::sum(Ixt)[0];
    double s_Iyt = cv::sum(Iyt)[0];

    Eigen::Matrix2d M; M<<s_Ixx, s_Ixy, s_Ixy, s_Iyy;
    Eigen::Vector2d b; b<<s_Ixt, s_Iyt;

    return -M.inverse() * b;
}

bool kltRefinement (const cv::Point2d &coord, Eigen::Vector2d &f, const cv::Mat &model_patch,
                const cv::Mat &event_frame, const double &outlier_threshold, const int &border_type,
                const uint8_t &border_value)
{
    /** USE AS: bool inlier = ::eds::utils::kltRefinement(coord[i], this->kf->flow[i], model_patches[i],
            event_frame, cv::norm(event_patches[i], cv::NORM_L2SQR)); **/

    uint16_t patch_size = model_patch.cols;

    /** coordinate candidate **/
    double angle = std::atan2(f[1], f[0]); //atan (y/x)
    Eigen::Vector2d direction; direction << cos(angle), sin(angle); //x-y direction
    cv::Point2d coord_tip (direction[0], direction[1]); // assumes 1px displacement

    /** Create a bigger model image (padding) **/
    cv::Mat img; uint16_t patch_radius = patch_size/2;
    cv::copyMakeBorder(event_frame, img, patch_radius+1, patch_radius+1, patch_radius+1, patch_radius+1, border_type, border_value);
    std::cout<<"patch_size: "<<patch_size<<" patch_radius: "<<patch_radius<<" img.size: "<<img.size()<<std::endl;

    /** Compute the candidate points to explore **/
    std::vector<cv::Point2d> target_coord(3);
    if (angle > 0.0 && angle < M_PI/2.0) /** I Quadrant 0<x<90 **/
    {
        std::cout<<"I-Q ";
        target_coord[0] = coord +  cv::Point2d(1.0, 0.0);
        target_coord[1] = coord +  cv::Point2d(1.0, -1.0);
        target_coord[2] = coord +  cv::Point2d(0.0, -1.0);
    }
    else if (angle > M_PI/2.0 && angle < M_PI) /** II Quadrant 90<x<180 **/
    {
        std::cout<<"II-Q ";
        target_coord[0] = coord +  cv::Point2d(0.0, -1.0);
        target_coord[1] = coord +  cv::Point2d(-1.0, -1.0);
        target_coord[2] = coord +  cv::Point2d(-1.0, 0.0);
    }
    else if (angle < -M_PI/2.0 && angle < 0.0) /** III Quadrant 0.0>x<-90 **/
    {
        std::cout<<"III-Q ";
        target_coord[0] = coord +  cv::Point2d(-1.0, 0.0);
        target_coord[1] = coord +  cv::Point2d(-1.0, 1.0);
        target_coord[2] = coord +  cv::Point2d(0.0, 1.0);
    }
    else /** IV Quadrant -90>x<-180**/
    {
        std::cout<<"IV-Q ";
        target_coord[0] = coord +  cv::Point2d(0.0, 1.0);
        target_coord[1] = coord +  cv::Point2d(1.0, 1.0);
        target_coord[2] = coord +  cv::Point2d(1.0, 0.0);
    }

    /** Compute the ssds**/
    std::vector<double> ssds;
    for (size_t i=0; i<target_coord.size(); ++i)
    {
        cv::Rect grid_rect(target_coord[i].x, target_coord[i].y, patch_size, patch_size);
        std::cout<<"angle["<<angle*180/M_PI<<"] target_coord["<<i<<"]: "<<target_coord[i]<<std::endl;
        std::cout<<"to get event_patch"<<std::endl;
        cv::Mat event_patch = img(grid_rect);
        std::cout<<"to compute ssd "<<model_patch.size()<<" - "<<event_patch.size()<<std::endl;
        ssds.push_back(::eds::utils::ssd(model_patch, event_patch));
        std::cout<<"angle["<<angle*180/M_PI<<"] target_coord["<<i<<"]: "<<target_coord[i]<<" ssd: "<<ssds[i]<<std::endl;
    }

    /** Select the one with ninumim ssd **/
    std::vector<double>::iterator it = std::min_element(ssds.begin(), ssds.end());
    cv::Point2d f_ = coord - target_coord[std::distance(ssds.begin() , it)];
    f[0] = f_.x; f[1] = f_.y;
    std::cout<<"min_ssds["<<std::distance(ssds.begin(), it)<<"]: "<<*it<<" threshold["<<outlier_threshold<<"]"<<std::endl;

    return (*it < outlier_threshold)? true: false;
}

cv::Mat flowArrowsOnImage(const cv::Mat &img, const std::vector<cv::Point2d> &coord, const std::vector<Eigen::Vector2d> &flow, const cv::Vec3d &color, const size_t &skip_amount)
{
    assert(coord.size() == flow.size());

    cv::Mat flow_img; img.convertTo(flow_img, CV_8UC1, 255, 0);
    cv::cvtColor(flow_img, flow_img, cv::COLOR_GRAY2RGB);

    /** Draw the arrows **/
    auto it_c = coord.begin();
    auto it_f = flow.begin();
    for (; it_c < coord.end() && it_f <flow.end(); it_c+=skip_amount, it_f+=skip_amount)
    {
        double angle = atan2((*it_f)[1], (*it_f)[0]); //atan (y/x)
        double magnitude = (*it_f).norm(); // optical flow norm
        Eigen::Vector2d direction; direction << cos(angle), sin(angle); //x-y direction
        cv::Point2d arrow_tip (direction[0] * magnitude, direction[1] * magnitude);
        cv::Point2d it_end = (*it_c) + arrow_tip;
        cv::arrowedLine(flow_img, *it_c, it_end, color, 1.0 /*thickness*/,
                        cv::LINE_8 /* line type **/, 0 /*shift*/, 0.15 /*tiplength*/);

    }
    return flow_img;
}

cv::Point2d searchAlongEpiline(const cv::Size &size, const cv::Mat &img, const cv::Mat &patch, const cv::Vec3d &line,
                            const base::Transform3d &T_ef_kf, const cv::Point2d &norm_coord, const double &idepth,
                            const double &sigma, const cv::Mat &K, const ::eds::utils::SIMILARITY_MEASURE &method)
{
    /** Intrinsics **/
    double fx, fy, cx, cy;
    fx = K.at<double>(0,0); fy = K.at<double>(1,1);
    cx = K.at<double>(0,2); cy = K.at<double>(1,2);

    /** Get the event frame coordinates for max and min inverse depth **/
    double idepth_max = idepth+sigma;
    double idepth_min = idepth-sigma;
    Eigen::Vector3d p_mu(1.0/idepth * norm_coord.x, 1.0/idepth * norm_coord.y, 1.0/idepth);
    Eigen::Vector3d p_min(1.0/idepth_min * norm_coord.x, 1.0/idepth_min * norm_coord.y, 1.0/idepth_min);
    Eigen::Vector3d p_max(1.0/idepth_max * norm_coord.x, 1.0/idepth_max * norm_coord.y, 1.0/idepth_max);
    p_mu = T_ef_kf * p_mu; p_min = T_ef_kf * p_min; p_max = T_ef_kf * p_max;
    cv::Point2d coord_mu(fx * (p_mu[0]/p_mu[2]) + cx, fy * (p_mu[1]/p_mu[2]) + cy);
    cv::Point2d coord_min(fx * (p_min[0]/p_min[2]) + cx, fy * (p_min[1]/p_min[2]) + cy);
    cv::Point2d coord_max(fx * (p_max[0]/p_max[2]) + cx, fy * (p_max[1]/p_max[2]) + cy);

    //std::cout<<"IMG SIZE:"<<size<<std::endl;
    //std::cout<<"SEARCH idepth: "<<idepth <<" sigma: "<<sigma<<" idepth_max: "<<idepth_max<<" idepth_min: "<<idepth_min<<std::endl;
    //std::cout<<"SEARCH coord_mu: "<<coord_mu<<" coord_min: "<<coord_min <<" coord_max: "<<coord_max<<std::endl;

    std::vector<double> metric;
    std::vector<cv::Point2d> points;
    for (int i=0; i<size.width; ++i)
    {
        /** Select the point along the line **/
        cv::Point p(i, -(line[2] + line[0] * i) / line[1]);

        /** Check if point is in image size **/
        bool inlier = ((p.x >= 0.0) and (p.x < size.width)) and ((p.y >= 0.0) and (p.y < size.height));

        /** Check if the point is whithin depth sigma range **/
        bool inrange = ((p.x > coord_mu.x-2) and (p.x < coord_mu.x+2)) and ((p.y > coord_mu.y-2) and (p.y < coord_mu.y+2));

        if (inlier and inrange)
        {
            /** Select the patch at this point **/
            cv::Rect grid_rect(p.x, p.y, patch.cols, patch.rows);
            //std::cout<<"SEARCH IDX["<<i<<"] p: "<<p<<" grid_rect: "<<grid_rect<<std::endl;
            cv::Mat img_patch = img(grid_rect);
            //std::cout<<"SEARCH IDX["<<i<<"] img_patch: "<<img_patch.size()<<" patch: "<<patch.size()<<std::endl;
            //cv::imwrite("/tmp/event_patch_"+std::to_string(i)+".png", ::eds::utils::viz(img_patch));

            /** Compute similarity metric **/
            double value = ::base::NaN<double>();
            switch (method)
            {
            case NCC:
                value = ::eds::utils::ncc(img_patch, patch);
                break;
            case ZNCC:
                value = ::eds::utils::zncc(img_patch, patch);
                break;
             case SSD:
                value = ::eds::utils::ssd(img_patch, patch);
                break;
             case NSSD:
                value = ::eds::utils::nssd(img_patch, patch);
                break;
             case ZSSD:
                value = ::eds::utils::zssd(img_patch, patch);
                break;
             case SAD:
                value = ::eds::utils::sad(img_patch, patch);
                break;
             case ZSAD:
                value = ::eds::utils::zsad(img_patch, patch);
                break;
            default:
                break;
            }

            metric.push_back(value);
            points.push_back(cv::Point2d(p.x, p.y));
        }
    }

    /** Get the argmin **/
    if (!metric.empty())
    {
        std::vector<double>::iterator it;
        if (method == NCC || ZNCC)
        {
            it = std::max_element(metric.begin(), metric.end());
        }
        else
        {
            it = std::min_element(metric.begin(), metric.end());
        }
        //std::for_each(metric.begin(), metric.end(), [](double &v){ std::cout<<v<<" "; }); std::cout<<"\n";
        return points[std::distance(metric.begin(), it)];
    }
    else
    {
        return coord_mu;
    }
}

cv::Mat viz(const cv::Mat &img, bool color)
{
    cv::Mat img_viz;
    double min, max;
    cv::minMaxLoc(img, &min, &max);
    cv::Mat norm_img = (img - min)/(max -min);

    norm_img.convertTo(img_viz, CV_8UC1, 255, 0);
    if (color)
    {
        cv::applyColorMap(img_viz, img_viz, cv::COLORMAP_JET);
    }

    return img_viz;
}

cv::Mat epilinesViz (const cv::Mat &img, const std::vector<cv::Vec3d> &lines, const size_t &skip_amount)
{
    cv::Mat img_color;
    cv::cvtColor(img, img_color, cv::COLOR_GRAY2RGB);

    auto it_l = lines.begin();
    for(; it_l < lines.end(); it_l+=skip_amount)
    {
        const cv::Vec3d &l = *it_l;
        cv::Scalar color((double)std::rand() / RAND_MAX * 255,
        (double)std::rand() / RAND_MAX * 255,
        (double)std::rand() / RAND_MAX * 255);
        cv::line(img_color, cv::Point(0, -l[2]/l[1]), cv::Point(img_color.cols, -(l[2] + l[0] * img_color.cols) / l[1]), color);
    }

    return img_color;
}

cv::Point2d matchTemplate(const cv::Mat &img, const cv::Mat &templ, const int &match_method)
{
    cv::Mat img_display = img;
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    cv::Mat result; result.create( result_rows, result_cols, CV_32FC1);
    cv::matchTemplate(img, templ, result, match_method);

    cv::normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    double min_val; double max_val; cv::Point min_loc; cv::Point max_loc; cv::Point match_loc;
    cv::minMaxLoc( result, &min_val, &max_val, &min_loc, &max_loc, cv::Mat());
    std::string method_string;
    if( match_method  == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED )
    {
        match_loc = min_loc;
        if (match_method == cv::TM_SQDIFF_NORMED) method_string = "nssd";
        else method_string = "ssd";
    }
    else
    {
        match_loc = max_loc;
        if (match_method == cv::TM_CCOEFF_NORMED ) method_string = "ncc";
        else method_string = "cc";
    }

    cv::rectangle(img_display, match_loc, cv::Point( match_loc.x + templ.cols , match_loc.y + templ.rows ), cv::Scalar::all(0), 2, 8, 0);
    cv::rectangle(result, match_loc, cv::Point( match_loc.x + templ.cols , match_loc.y + templ.rows ), cv::Scalar::all(0), 2, 8, 0);
    //cv::imwrite("/tmp/img_display_"+method_string+".png", img_display);
    //cv::imwrite("/tmp/template_matching_"+ method_string+".png", eds::utils::viz(result));
    return cv::Point2d(match_loc.x, match_loc.y);
}

void getCalibration(const ::eds::calib::CameraInfo &cam_info, int &w_out, int &h_out, Eigen::Matrix3f &K_out, Eigen::Vector4f &D_out, Eigen::Matrix3f &R_rect_out, Eigen::Matrix3f &K_ref_out)
{
    cv::Mat K, K_ref, D, R_rect, P;
    R_rect  = cv::Mat_<float>::eye(3, 3);
    K = cv::Mat_<float>::eye(3, 3);
    K.at<float>(0,0) = cam_info.intrinsics[0];
    K.at<float>(1,1) = cam_info.intrinsics[1];
    K.at<float>(0,2) = cam_info.intrinsics[2];
    K.at<float>(1,2) = cam_info.intrinsics[3];

    cv::Size size_in = cv::Size(cam_info.width, cam_info.height);
    cv::Size size_out = cv::Size(cam_info.out_width, cam_info.out_height);

    /** Extract the information **/
    D = cv::Mat_<float>::zeros(4, 1);
    for (size_t i=0; i<cam_info.D.size(); ++i)
    {
        D.at<float>(i, 0) = cam_info.D[i];
    }

    if (cam_info.P.size() == 12)
    {
        P = cv::Mat_<float>::zeros(4, 4);
        for (auto row=0; row<P.rows; ++row)
        {
            for (auto col=0; col<P.cols; ++col)
            {
                P.at<float>(row, col) = cam_info.P[(P.cols*row)+col];
            }
        }
    }

    if (cam_info.R.size() == 9)
    {
        for (auto row=0; row<R_rect.rows; ++row)
        {
            for (auto col=0; col<R_rect.cols; ++col)
            {
                R_rect.at<float>(row, col) = cam_info.R[(R_rect.cols*row)+col];
            }
        }
    }

    /** Compute the information **/
    if (P.total()>0)
        K_ref = P(cv::Rect(0,0,3,3)).clone();
    if (R_rect.total()>0)
        R_rect = R_rect.clone();

    /** Distortion matrices and model **/
    if (K.total() > 0 && D.total() > 0)
    {
        if (K_ref.total() == 0)
        {
            if (cam_info.distortion_model.compare("equidistant") != 0)
            {
               /** radtan model **/
                K_ref = cv::getOptimalNewCameraMatrix(K, D, size_in, 0.0);
            }
            else
            {
                /** Kalibr equidistant model is opencv fisheye **/
                cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, size_in, R_rect, K_ref);
            }
        }
    }
    else
    {
        K_ref = K.clone();
    }

    /** Check if the input image should be rescale **/
    if ((size_out.height == 0 || size_out.width == 0) || (size_out.height > size_in.height || size_out.width > size_in.width))
    {
        size_out.width = size_in.width; size_out.height = size_in.height;
    }
    else
    {
        /** Rescale the input **/
        std::array<float, 2> out_scale;
        out_scale[0] = size_in.width / size_out.width;
        out_scale[1] = size_in.height / size_out.height;
        K.at<float>(0,0) /=  out_scale[0]; K.at<float>(1,1) /=  out_scale[1];
        K.at<float>(0,2) /=  out_scale[0]; K.at<float>(1,2) /=  out_scale[1];
        K_ref.at<float>(0,0) /=  out_scale[0]; K_ref.at<float>(1,1) /=  out_scale[1];
        K_ref.at<float>(0,2) /=  out_scale[0]; K_ref.at<float>(1,2) /=  out_scale[1];
    }

    /** Save in the output **/
    cv::cv2eigen(K, K_out);
    cv::cv2eigen(D, D_out);
    cv::cv2eigen(R_rect, R_rect_out);
    cv::cv2eigen(K_ref, K_ref_out);
    w_out = size_out.width; h_out = size_out.height;
}

void getUndistortImage(const std::string &distortion_model, cv::Mat &input, cv::Mat &output, cv::Mat &K, cv::Mat &K_ref, cv::Mat &D)
{
    if (distortion_model.compare("equidistant") != 0)
    {
        /** radtan model **/
        cv::undistort(input, output, K, D, K_ref);
    }
    else
    {
        /** Kalibr equidistant model is opencv fisheye **/
        cv::fisheye::undistortImage(input, output, K, D, K_ref);
    }
}

}} //end namespace eds::utils
