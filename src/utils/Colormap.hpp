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


#ifndef _EDS_COLORMAP_HPP_
#define _EDS_COLORMAP_HPP_

#include <eds/utils/Utils.hpp>

namespace eds { namespace utils {

    class ColorMap
    {

    protected:
        cv::Mat _lut;

    public:
        virtual ~ColorMap() {}

        // Applies the colormap on a given image.
        //
        // This function expects BGR-aligned data of type CV_8UC1 or CV_8UC3.
        // Throws an error for wrong-aligned lookup table, which must be
        // of size 256 in the latest OpenCV release (2.3.1).
        void operator()(cv::InputArray src, cv::OutputArray dst) const;

        // Setup base map to interpolate from.
        //not used: virtual void init(int n) = 0;

        // Interpolates from a base colormap.
        static cv::Mat linear_colormap(cv::InputArray X,
                cv::InputArray r, cv::InputArray g, cv::InputArray b,
                int n) {
            return linear_colormap(X,r,g,b,linspace(0,1,n));
        }

        // Interpolates from a base colormap.
        static cv::Mat linear_colormap(cv::InputArray X,
                cv::InputArray r, cv::InputArray g, cv::InputArray b,
                float begin, float end, float n) {
            return linear_colormap(X,r,g,b,linspace(begin,end, cvRound(n)));
        }

        // Interpolates from a base colormap.
        static cv::Mat linear_colormap(cv::InputArray X,
                cv::InputArray r, cv::InputArray g, cv::InputArray b,
                cv::InputArray xi);
    };



    class BlueWhiteRed : public ColorMap
    {
        public:
            BlueWhiteRed():ColorMap() {
                init(256);
            }
            BlueWhiteRed(int n) : ColorMap() {
                init(n);
            }
            void init(int n);
    };

    class BlueWhiteBlack : public ColorMap
    {
        public:
            BlueWhiteBlack():ColorMap() {
                init(256);
            }
            BlueWhiteBlack(int n) : ColorMap() {
                init(n);
            }
            void init(int n);
    };

    class GreenWhiteRed : public ColorMap
    {
        public:
            GreenWhiteRed():ColorMap() {
                init(256);
            }
            GreenWhiteRed(int n) : ColorMap() {
                init(n);
            }
            void init(int n);
    };

} //utils namespace
} // end namespace

#endif // _EDS_COLORMAP_HPP_