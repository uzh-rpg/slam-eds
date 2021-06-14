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

#include "Colormap.hpp"

namespace eds { namespace utils {

void ColorMap::operator()(cv::InputArray _src, cv::OutputArray _dst) const
{
    if(_lut.total() != 256)
        CV_Error(cv::Error::StsAssert, "cv::LUT only supports tables of size 256.");
    cv::Mat src = _src.getMat();
    if(src.type() != CV_8UC1  &&  src.type() != CV_8UC3)
        CV_Error(cv::Error::StsBadArg, "cv::ColorMap only supports source images of type CV_8UC1 or CV_8UC3");
    // Turn into a BGR matrix into its grayscale representation.
    if(src.type() == CV_8UC3)
        cvtColor(src.clone(), src, cv::COLOR_BGR2GRAY);
    cv::cvtColor(src.clone(), src, cv::COLOR_GRAY2BGR);
    // Apply the ColorMap.
    cv::LUT(src, _lut, _dst);
}

cv::Mat ColorMap::linear_colormap(cv::InputArray X,
        cv::InputArray r, cv::InputArray g, cv::InputArray b,
        cv::InputArray xi) {
    cv::Mat lut, lut8;
    cv::Mat planes[] = {
            interp1(X, b, xi),
            interp1(X, g, xi),
            interp1(X, r, xi)};
    merge(planes, 3, lut);
    lut.convertTo(lut8, CV_8U, 255.);
    return lut8;
}

void BlueWhiteRed::init(int n)
{

    static const float r[] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    static const float g[] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    static const float b[] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.85714286, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.57142857, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    cv::Mat X = linspace(0,1,128);
    this->_lut = eds::utils::ColorMap::linear_colormap(X,
            cv::Mat(128,1, CV_32FC1, (void*)r).clone(), // red
            cv::Mat(128,1, CV_32FC1, (void*)g).clone(), // green
            cv::Mat(128,1, CV_32FC1, (void*)b).clone(), // blue
            n);  // number of sample points
}

void BlueWhiteBlack::init(int n)
{

    static const float r[] = {0., 0.03149606, 0.06299213, 0.09448819, 0.12598425, 0.15748031, 0.18897638, 0.22047244, 0.2519685 , 0.28346457, 0.31496063, 0.34645669, 0.37795276, 0.40944882, 0.44094488, 0.47244094, 0.50393701, 0.53543307, 0.56692913, 0.5984252 , 0.62992126, 0.66929134, 0.7007874 , 0.73228346, 0.76377953, 0.79527559, 0.82677165, 0.85826772, 0.88976378, 0.92125984, 0.95275591, 0.98425197, 0.98425197, 0.95275591, 0.92125984, 0.88976378, 0.85826772, 0.82677165, 0.79527559, 0.76377953, 0.73228346, 0.7007874 , 0.66141732, 0.62992126, 0.5984252 , 0.56692913, 0.53543307, 0.50393701, 0.47244094, 0.44094488, 0.40944882, 0.37795276, 0.34645669, 0.31496063, 0.28346457, 0.2519685 , 0.22047244, 0.18897638, 0.15748031, 0.12598425, 0.09448819, 0.06299213, 0.03149606, 0.0};
    static const float g[] = {0., 0.03149606, 0.06299213, 0.09448819, 0.12598425, 0.15748031, 0.18897638, 0.22047244, 0.2519685 , 0.28346457, 0.31496063, 0.34645669, 0.37795276, 0.40944882, 0.44094488, 0.47244094, 0.50393701, 0.53543307, 0.56692913, 0.5984252 , 0.62992126, 0.66929134, 0.7007874 , 0.73228346, 0.76377953, 0.79527559, 0.82677165, 0.85826772, 0.88976378, 0.92125984, 0.95275591, 0.98425197, 0.98425197, 0.95275591, 0.92125984, 0.88976378, 0.85826772, 0.82677165, 0.79527559, 0.76377953, 0.73228346, 0.7007874 , 0.66141732, 0.62992126, 0.5984252 , 0.56692913, 0.53543307, 0.50393701, 0.47244094, 0.44094488, 0.40944882, 0.37795276, 0.34645669, 0.31496063, 0.28346457, 0.2519685 , 0.22047244, 0.18897638, 0.15748031, 0.12598425, 0.09448819, 0.06299213, 0.03149606, 0.0};
    static const float b[] = {0., 0.03149606, 0.06299213, 0.09448819, 0.12598425, 0.15748031, 0.18897638, 0.22047244, 0.2519685 , 0.28346457, 0.31496063, 0.34645669, 0.37795276, 0.40944882, 0.44094488, 0.47244094, 0.50393701, 0.53543307, 0.56692913, 0.5984252 , 0.62992126, 0.66929134, 0.7007874 , 0.73228346, 0.76377953, 0.79527559, 0.82677165, 0.85826772, 0.88976378, 0.92125984, 0.95275591, 0.98425197, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    cv::Mat X = linspace(0,1,64);
    this->_lut = eds::utils::ColorMap::linear_colormap(X,
            cv::Mat(64,1, CV_32FC1, (void*)r).clone(), // red
            cv::Mat(64,1, CV_32FC1, (void*)g).clone(), // green
            cv::Mat(64,1, CV_32FC1, (void*)b).clone(), // blue
            n);  // number of sample points
}

void GreenWhiteRed::init(int n)
{

    static const float r[] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.98425197, 0.95275591, 0.92125984, 0.88976378, 0.85826772, 0.82677165, 0.79527559, 0.76377953, 0.73228346, 0.7007874 , 0.66141732, 0.62992126, 0.5984252 , 0.56692913, 0.53543307, 0.50393701, 0.47244094, 0.44094488, 0.40944882, 0.37795276, 0.34645669, 0.31496063, 0.28346457, 0.2519685 , 0.22047244, 0.18897638, 0.15748031, 0.12598425, 0.09448819, 0.06299213, 0.03149606, 0.};
    static const float g[] = {0., 0.03149606, 0.06299213, 0.09448819, 0.12598425, 0.15748031, 0.18897638, 0.22047244, 0.2519685 , 0.28346457, 0.31496063, 0.34645669, 0.37795276, 0.40944882, 0.44094488, 0.47244094, 0.50393701, 0.53543307, 0.56692913, 0.5984252 , 0.62992126, 0.66929134, 0.7007874 , 0.73228346, 0.76377953, 0.79527559, 0.82677165, 0.85826772, 0.88976378, 0.92125984, 0.95275591, 0.98425197, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    static const float b[] = {0., 0.03149606, 0.06299213, 0.09448819, 0.12598425, 0.15748031, 0.18897638, 0.22047244, 0.2519685 , 0.28346457, 0.31496063, 0.34645669, 0.37795276, 0.40944882, 0.44094488, 0.47244094, 0.50393701, 0.53543307, 0.56692913, 0.5984252 , 0.62992126, 0.66929134, 0.7007874 , 0.73228346, 0.76377953, 0.79527559, 0.82677165, 0.85826772, 0.88976378, 0.92125984, 0.95275591, 0.98425197, 0.98425197, 0.95275591, 0.92125984, 0.88976378, 0.85826772, 0.82677165, 0.79527559, 0.76377953, 0.73228346, 0.7007874 , 0.66141732, 0.62992126, 0.5984252 , 0.56692913, 0.53543307, 0.50393701, 0.47244094, 0.44094488, 0.40944882, 0.37795276, 0.34645669, 0.31496063, 0.28346457, 0.2519685 , 0.22047244, 0.18897638, 0.15748031, 0.12598425, 0.09448819, 0.06299213, 0.03149606, 0.};
    cv::Mat X = linspace(0,1,64);
    this->_lut = eds::utils::ColorMap::linear_colormap(X,
            cv::Mat(64,1, CV_32FC1, (void*)r).clone(), // red
            cv::Mat(64,1, CV_32FC1, (void*)g).clone(), // green
            cv::Mat(64,1, CV_32FC1, (void*)b).clone(), // blue
            n);  // number of sample points
}
}} //end namespace eds::utils