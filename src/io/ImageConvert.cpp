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

#include <frame_helper/FrameHelper.h>
#include <eds/io/ImageConvert.h>

namespace eds { namespace io
{

dso::MinimalImageB3* toMinimalImageB3(const Eigen::Vector3f *fd, const int &w, const int &h)
{
    int wh = w*h;
    dso::MinimalImageB3* img_b3 = new dso::MinimalImageB3(w, h);
    for(int i=0;i<wh;i++)
    {
        int c = fd[i][0]*0.9f;
        if(c>255) c=255;
        img_b3->at(i) = ::dso::Vec3b(c,c,c);
    }
    return img_b3;
}

void MinimalImageB3ToFrame(const dso::MinimalImageB3 *img_b3, const ::base::Time &timestamp, ::base::samples::frame::Frame &frame)
{
    cv::Mat img =  cv::Mat(img_b3->h, img_b3->w, CV_8UC3, img_b3->data);
    //cv::imwrite("/tmp/dso_img_"+std::to_string(this->frame_idx)+".png", frame);

    frame.image.clear();
    frame_helper::FrameHelper::copyMatToFrame(img, frame);
    frame.time = timestamp;

}

}
}//end of dso namespace
