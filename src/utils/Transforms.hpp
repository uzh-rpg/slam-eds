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



#ifndef _EDS_TRANSFORMS_HPP_
#define _EDS_TRANSFORMS_HPP_

/** Base types **/
#include <base/Eigen.hpp>
#include <base/Time.hpp>
#include <math.h>

namespace eds { namespace transforms {

    inline Eigen::Matrix3d axis_angle(Eigen::Vector3d &axis, double &theta)
    {
        Eigen::Matrix3d R;
        if (theta*theta > 1e-06)
        {
            double wx = axis[0]; double wy = axis[1]; double wz = axis[2];
            double costheta = cos(theta); double sintheta = sin(theta);

            double c_1 = 1.0 - costheta;
            double wx_sintheta = wx * sintheta;
            double wy_sintheta = wy * sintheta;
            double wz_sintheta = wz * sintheta;
            double C00 = c_1 * wx * wx;
            double C01 = c_1 * wx * wy;
            double C02 = c_1 * wx * wz;
            double C11 = c_1 * wy * wy;
            double C12 = c_1 * wy * wz;
            double C22 = c_1 * wz * wz;

            R(0,0) =     costheta + C00;
            R(1,0) =  wz_sintheta + C01;
            R(2,0) = -wy_sintheta + C02;
            R(0,1) = -wz_sintheta + C01;
            R(1,1) =     costheta + C11;
            R(2,1) =  wx_sintheta + C12;
            R(0,2) =  wy_sintheta + C02;
            R(1,2) = -wx_sintheta + C12;
            R(2,2) =     costheta + C22;
            return R;
        }
        else
        {
            Eigen::Vector3d rotvec = axis*theta;
            R(0,0) = 1.0;
            R(1,0) = rotvec[2];
            R(2,0) = -rotvec[1];
            R(0,1) = -rotvec[2];
            R(1,1) = 1.0;
            R(2,1) = rotvec[0];
            R(0,2) = rotvec[1];
            R(1,2) = -rotvec[0];
            R(2,2) = 1.0;
            return R;

        }
    };

} //transforms namespace
} // end namespace

#endif

