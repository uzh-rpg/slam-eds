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
 * along with this program. If not, see <http://www.gnu.org/licenses/>
 */

#include <eds/utils/settings.h>
#include <eds/utils/NumType.h>
#include <eds/utils/FrameShell.h>
#include <eds/utils/globalCalib.h>
#include <eds/tracking/HessianBlocks.h>
#include <eds/tracking/ImmaturePoint.h>
#include <eds/io/OutputMaps.h> 

namespace dso { namespace io
{

template<typename Derived>
bool is_not_nan(const Eigen::MatrixBase<Derived>& x)
{
	return ((x.array() == x.array())).all();
}

base::samples::Pointcloud getMap(const dso::FrameHessian *fh, dso::CalibHessian *hcalib, const base::Vector4d color, const bool &single_point)
{
    base::samples::Pointcloud pcl;

    float fx = hcalib->fxl();
    float fy = hcalib->fyl();
    float cx = hcalib->cxl();
    float cy = hcalib->cyl();

    base::Transform3d T_w_kf = dso::SE3ToBaseTransform(fh->get_worldToCam_evalPT()).inverse();

    for(::dso::PointHessian* p : fh->pointHessians)
    {
        double d_i = 1.0/p->idepth_scaled;
        if (is_not_nan(color))
        {
            /** Push points in the world frame coordinate **/
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-cx)/fx, d_i*(p->v-cy)/fy, d_i));
            pcl.colors.push_back(color);//point intensity
        }
        else
        {
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-cx)/fx, d_i*(p->v-cy)/fy, d_i));//u-v point8
            //pcl.colors.push_back(::base::Vector4d(p->color[7], p->color[7], p->color[7], 1.0));//point intensity
        }

        if (!single_point)
        {
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-cx)/fx, d_i*(p->v+2-cy)/fy, d_i));//x-y point1
            if (is_not_nan(color)) pcl.colors.push_back(color);//point intensity
            //pcl.colors.push_back(::base::Vector4d(p->color[0], p->color[0], p->color[0], 1.0));//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-1-cx)/fx, d_i*(p->v+1-cy)/fy, d_i ));//x-y point2
            if (is_not_nan(color)) pcl.colors.push_back(color);//point intensity
            //pcl.colors.push_back(::base::Vector4d(p->color[1], p->color[1], p->color[1], 1.0));//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-2-cx)/fx, d_i*(p->v-cy)/fy, d_i ));//x-y point3
            if (is_not_nan(color)) pcl.colors.push_back(color);//point intensity
            //pcl.colors.push_back(::base::Vector4d(p->color[2], p->color[2], p->color[2], 1.0));//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-1-cx)/fx, d_i*(p->v-1-cy)/fy, d_i ));//x-y point4
            if (is_not_nan(color)) pcl.colors.push_back(color);//point intensity
            //pcl.colors.push_back(::base::Vector4d(p->color[3], p->color[3], p->color[3], 1.0));//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-cx)/fx, d_i*(p->v-2-cy)/fy, d_i ));//x-y point5
            if (is_not_nan(color)) pcl.colors.push_back(color);//point intensity
            //pcl.colors.push_back(::base::Vector4d(p->color[4], p->color[4], p->color[4], 1.0));//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u+1-cx)/fx, d_i*(p->v-1-cy)/fy, d_i ));//x-y point6
            if (is_not_nan(color)) pcl.colors.push_back(color);//point intensity
            //pcl.colors.push_back(::base::Vector4d(p->color[5], p->color[5], p->color[5], 1.0));//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u+2-cx)/fx, d_i*(p->v-cy)/fy, d_i ));//x-y point7
            if (is_not_nan(color)) pcl.colors.push_back(color);//point intensity
            //pcl.colors.push_back(::base::Vector4d(p->color[6], p->color[6], p->color[6], 1.0));//point intensity
        }
    }

    return pcl;
}

base::samples::Pointcloud getImmatureMap(const dso::FrameHessian *fh, dso::CalibHessian *hcalib, const bool &single_point)
{
    base::samples::Pointcloud pcl;

    float fx = hcalib->fxl();
    float fy = hcalib->fyl();
    float cx = hcalib->cxl();
    float cy = hcalib->cyl();

    base::Transform3d T_w_kf = dso::SE3ToBaseTransform(fh->get_worldToCam_evalPT()).inverse();

    for(::dso::ImmaturePoint* p : fh->immaturePoints)
    {
        /** This is the depth. Same maner as Hessian points are initialized from Immature Points **/
        double d_i = 1.0/(0.5 * (p->idepth_max+p->idepth_min));

        if (d_i < 0 || !std::isfinite(d_i)) continue;

        ::base::Vector4d color;
        if(p->lastTraceStatus==dso::ImmaturePointStatus::IPS_GOOD)
            color = ::base::Vector4d(0,1.0,0, 1.0);//GREEN
        if(p->lastTraceStatus==dso::ImmaturePointStatus::IPS_OOB)
            color = ::base::Vector4d(1.0,0,0,1.0);//RED
        if(p->lastTraceStatus==dso::ImmaturePointStatus::IPS_OUTLIER)
            color = ::base::Vector4d(0,0,1.0,1.0);//BLUE
        if(p->lastTraceStatus==dso::ImmaturePointStatus::IPS_SKIPPED)
            color = ::base::Vector4d(0.0,1.0,1.0,1.0);//CYAN
        if(p->lastTraceStatus==dso::ImmaturePointStatus::IPS_BADCONDITION)
            color = ::base::Vector4d(1.0,1.0,1.0,1.0); //WHITE
        if(p->lastTraceStatus==dso::ImmaturePointStatus::IPS_UNINITIALIZED)
            color = ::base::Vector4d(0,0,0,1.0);//BLACK

        pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-cx)/fx, d_i*(p->v-cy)/fy, d_i));//u-v point8
        pcl.colors.push_back(color);//point intensity

        if (!single_point)
        {
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-cx)/fx, d_i*(p->v+2-cy)/fy, d_i));//x-y point1
            pcl.colors.push_back(color);//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-1-cx)/fx, d_i*(p->v+1-cy)/fy, d_i ));//x-y point2
            pcl.colors.push_back(color);//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-2-cx)/fx, d_i*(p->v-cy)/fy, d_i ));//x-y point3
            pcl.colors.push_back(color);//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-1-cx)/fx, d_i*(p->v-1-cy)/fy, d_i ));//x-y point4
            pcl.colors.push_back(color);//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u-cx)/fx, d_i*(p->v-2-cy)/fy, d_i ));//x-y point5
            pcl.colors.push_back(color);//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u+1-cx)/fx, d_i*(p->v-1-cy)/fy, d_i ));//x-y point6
            pcl.colors.push_back(color);//point intensity
            pcl.points.push_back(T_w_kf * ::base::Point(d_i*(p->u+2-cx)/fx, d_i*(p->v-cy)/fy, d_i ));//x-y point7
            pcl.colors.push_back(color);//point intensity
        }
    }

    return pcl;
}

}
}//end of dso namespace
