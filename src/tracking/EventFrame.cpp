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

#include "EventFrame.hpp"

using namespace eds::tracking;

EventFrame::EventFrame(const ::eds::calib::Camera &cam, const ::eds::calib::Camera &newcam, const std::string &distortion_model)
{
    this->K = cam.K.clone();
    this->D = cam.D.clone();
    this->R_rect = newcam.R * cam.R.t();
    this->K_ref =  newcam.K.clone();
    this->distortion_model = distortion_model;

    if (distortion_model.compare("equidistant") != 0)
    {
        cv::initUndistortRectifyMap(this->K, this->D,
                            this->R_rect, this->K_ref,
                            newcam.size,//undistorted image size
                            CV_32FC1,
                            this->mapx, this->mapy);
    }
    else
    {
        cv::fisheye::initUndistortRectifyMap(this->K, this->D,
                            this->R_rect, this->K_ref,
                            newcam.size,//undistorted image size
                            CV_32FC1,
                            this->mapx, this->mapy);
    }
    /** Create look-up table for all possible coordinates in events
     * We need to do this since initUndistortRectifyMap create the inverse
     * mapping for remap. Which is not useful for undistroting events.
     * We do this only once in the constructor **/
    std::vector<cv::Point2f> coord, undist_coord; // H x W points
    for (int y=0; y<newcam.size.height; ++y)
    {
        for (int x=0; x<newcam.size.width; ++x)
        {
            coord.push_back(cv::Point2f(x, y));
        }
    }

    if (distortion_model.compare("equidistant") != 0)
    {
        cv::undistortPoints(coord, undist_coord, this->K, this->D, this->R_rect, this->K_ref);
    }
    else
    {
        cv::fisheye::undistortPoints(coord, undist_coord, this->K, this->D, this->R_rect, this->K_ref);
    }

    /** Reshape to get the forward maps **/
    this->fwd_mapx = cv::Mat_<float>::eye(newcam.size.height, newcam.size.width);
    this->fwd_mapy = cv::Mat_<float>::eye(newcam.size.height, newcam.size.width);
    int idx = 0; for (auto &it : undist_coord)
    {
        float y = idx/newcam.size.width;
        float x = idx - ((int)y * newcam.size.width);
        this->fwd_mapx.at<float>(y, x) = it.x;
        this->fwd_mapy.at<float>(y, x) = it.y;
        idx++;
    }

    /* Scale to one (no resize) **/
    this->out_scale[0] = 1.0; this->out_scale[1] = 1.0;

    /** Check if the input image should be downscaled **/
    if ((newcam.out_size.height != 0 || newcam.out_size.width != 0) || (newcam.out_size.height != newcam.size.height || newcam.out_size.width != newcam.size.width))
    {
        /** Downrescale the input **/
        this->out_scale[0] = (double)newcam.size.width / (double)newcam.out_size.width;
        this->out_scale[1] = (double)newcam.size.height / (double)newcam.out_size.height;
        this->K.at<double>(0,0) /=  this->out_scale[0]; this->K.at<double>(1,1) /=  this->out_scale[1];
        this->K.at<double>(0,2) /=  this->out_scale[0]; this->K.at<double>(1,2) /=  this->out_scale[1];
        this->K_ref.at<double>(0,0) /=  this->out_scale[0]; this->K_ref.at<double>(1,1) /=  this->out_scale[1];
        this->K_ref.at<double>(0,2) /=  this->out_scale[0]; this->K_ref.at<double>(1,2) /=  this->out_scale[1];
    }

    std::cout<<"** EVENTFRAME: CAMERA CALIB: **"<<std::endl;
    std::cout<<"Model: "<<distortion_model<<std::endl;
    std::cout<<"Size: "<<newcam.size<<std::endl;
    std::cout<<"Out Size: "<<newcam.out_size<<std::endl;
    std::cout<<"K:\n"<<this->K<<std::endl;
    std::cout<<"D:\n"<<this->D<<std::endl;
    std::cout<<"R:\n"<<this->R_rect<<std::endl;
    std::cout<<"K_ref:\n"<<this->K_ref<<std::endl;
    std::cout<<"OUT SCALE ["<<this->out_scale[0]<<","<<this->out_scale[1]<<"]"<<std::endl;

    std::cout<<"mapx: "<<this->mapx.rows<<" x "<<this->mapx.cols;
    std::cout<<" forward mapx: "<<this->fwd_mapx.rows<<" x "<<this->fwd_mapx.cols<<std::endl;
    std::cout<<"mapy: "<<this->mapy.rows<<" x "<<this->mapy.cols;
    std::cout<<" forward mapy: "<<this->fwd_mapy.rows<<" x "<<this->fwd_mapy.cols<<std::endl;
}

EventFrame::EventFrame(const uint64_t &idx, const std::vector<base::samples::Event> &events,
                    const ::eds::calib::CameraInfo &cam_info, const int &num_levels,
                    const ::base::Affine3d &T,
                    const cv::Size &out_size)
{
    cv::Mat K, D, R_rect, P;
    R_rect  = cv::Mat_<double>::eye(3, 3);
    K = cv::Mat_<double>::eye(3, 3);
    K.at<double>(0,0) = cam_info.intrinsics[0];
    K.at<double>(1,1) = cam_info.intrinsics[1];
    K.at<double>(0,2) = cam_info.intrinsics[2];
    K.at<double>(1,2) = cam_info.intrinsics[3];

    D = cv::Mat_<double>::zeros(4, 1);
    for (size_t i=0; i<cam_info.D.size(); ++i)
    {
        D.at<double>(i, 0) = cam_info.D[i];
    }

    if (cam_info.P.size() == 12)
    {
        P = cv::Mat_<double>::zeros(4, 4);
        for (auto row=0; row<P.rows; ++row)
        {
            for (auto col=0; col<P.cols; ++col)
            {
                P.at<double>(row, col) = cam_info.P[(P.cols*row)+col];
            }
        }
    }

    if (cam_info.R.size() == 9)
    {
        for (auto row=0; row<R_rect.rows; ++row)
        {
            for (auto col=0; col<R_rect.cols; ++col)
            {
                R_rect.at<double>(row, col) = cam_info.R[(R_rect.cols*row)+col];
            }
        }
    }

    (*this) = EventFrame(idx, events, cam_info.height, cam_info.width, K, D, R_rect, P, cam_info.distortion_model, num_levels, T, out_size);
}

EventFrame::EventFrame(const uint64_t &idx, const std::vector<base::samples::Event> &events, const uint16_t height, const uint16_t width,
            cv::Mat &K, cv::Mat &D, cv::Mat &R_rect, cv::Mat &P, const std::string distortion_model, const int &num_levels,
            const ::base::Affine3d &T, const cv::Size &out_size)
            :idx(idx), height(height), width(width), distortion_model(distortion_model), T_w_ef(T)
{

    if (P.total()>0)
        this->K_ref = P(cv::Rect(0,0,3,3));
    if (R_rect.total()>0)
        this->R_rect = R_rect.clone();

    /** Distortion **/
    if (K.total() > 0 && D.total() > 0)
    {
        this->K = K.clone();
        this->D = D.clone();
        if (this->K_ref.total() == 0)
        {
            if (distortion_model.compare("equidistant") != 0)
            {
                this->K_ref = cv::getOptimalNewCameraMatrix(this->K, this->D, cv::Size(width, height), 0.0);
            }
            else
            {
                cv::fisheye::estimateNewCameraMatrixForUndistortRectify(this->K, this->D, cv::Size(width, height), this->R_rect, this->K_ref);
            }
        }
    }
    else
    {
        this->K_ref = K.clone();
    }


    /** Get the events and polarity **/
    for (auto it=events.begin(); it!=events.end(); ++it)
    {
        this->coord.push_back(cv::Point2d(it->x, it->y));
        this->pol.push_back((it->polarity)?1:-1);
        if (it == events.begin())
            this->first_time = it->ts;
        else if ((it+1) == events.end())
            this->last_time = it->ts;
    }

    if (first_time.toMicroseconds() > last_time.toMicroseconds())
    {
        std::string error_message = std::string("[EVENT_FRAME] FATAL ERROR Event time[0] > event time [N-1] ");
        throw std::runtime_error(error_message);
    }

    /** Frame time as the median event time **/
    auto it = events.begin(); std::advance(it, events.size()/2);
    this->time = it->ts;

    /** Delta time of thsi event frame **/
    this->delta_time = (last_time - first_time);

    /** Undistort the points **/
    if (distortion_model.compare("equidistant") != 0)
    {
        cv::undistortPoints(this->coord, this->undist_coord, this->K, this->D, this->R_rect, this->K_ref);
    }
    else
    {
        cv::fisheye::undistortPoints(this->coord, this->undist_coord, this->K, this->D, this->R_rect, this->K_ref);
    }

    /**  Compute brightness change event frame (undistorted) per pyramid level **/
    cv::Mat event_img = eds::utils::drawValuesPoints(this->undist_coord, this->pol, height, width, "bilinear", 0.5, true);
    cv::Size size = event_img.size();

    /** Check if the input image should be rescale **/
    if ((out_size.height == 0 || out_size.width == 0) || (out_size.height > size.height || out_size.width > size.width))
    {
        this->out_scale[0] = 1.0; this->out_scale[1] = 1.0;
        std::cout<<"[EVENT_FRAME] out_scale["<<this->out_scale[0]<<","<<this->out_scale[1]<<"] out_size: "<<size<<std::endl;
    }
    else
    {
        /** Rescale the input **/
        this->out_scale[0] = (double)size.width / (double)out_size.width;
        this->out_scale[1] = (double)size.height / (double)out_size.height;
        std::cout<<"[EVENT_FRAME] out_scale["<<this->out_scale[0]<<","<<this->out_scale[1]<<"] out_size: "<<out_size<<std::endl;
        this->width /= this->out_scale[0]; this->height /= this->out_scale[1];
        this->K.at<double>(0,0) /=  this->out_scale[0]; this->K.at<double>(1,1) /=  this->out_scale[1];
        this->K.at<double>(0,2) /=  this->out_scale[0]; this->K.at<double>(1,2) /=  this->out_scale[1];
        this->K_ref.at<double>(0,0) /=  this->out_scale[0]; this->K_ref.at<double>(1,1) /=  this->out_scale[1];
        this->K_ref.at<double>(0,2) /=  this->out_scale[0]; this->K_ref.at<double>(1,2) /=  this->out_scale[1];
        cv::resize(event_img, event_img, out_size, cv::INTER_CUBIC);
        size = event_img.size();
    }

    this->frame.push_back(event_img);
    for (int i=1; i<num_levels; ++i)
    {
        cv::Mat event_img_dilate, event_img_erode;
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*i + 1, 2*i + 1), cv::Point(i, i));
        cv::dilate(event_img, event_img_dilate, element);
        cv::erode(event_img, event_img_erode, element);
        this->frame.push_back(event_img_dilate + event_img_erode);
    }

    /** Norm of the event frame **/
    for (auto f=frame.begin(); f!=frame.end(); ++f)
    {
        double n_f = cv::norm (*f);
        this->norm.push_back(n_f);
    }

    /** Normalized event frame **/
    int id = 0; this->event_frame.clear();
    for (auto &it : this->frame)
    {
        std::vector<double> frame_item;
        for (int row=0; row<it.rows; row++)
        {
            for (int col=0; col<it.cols;col++)
            {
                /** When using PhotometricError cost function  **/
                frame_item.push_back(it.at<double>(row,col)/this->norm[id]);
                /** When using PhotometricErrorNC cost function **/
                //frame_item.push_back(it.at<double>(row,col));
            }
        }
        this->event_frame.push_back(frame_item);
        ++id;
    }
    std::cout<<"[EVENT_FRAME] Created ID["<<this->idx<<"] with: "<<events.size()
                <<" events. start time "<<first_time.toMicroseconds()<<" end time "
                <<last_time.toMicroseconds()<<std::endl;
    std::cout<<"[EVENT_FRAME] event frame ["<<std::addressof(this->event_frame)<<"] size:"<<this->event_frame.size()<<std::endl;

}

void EventFrame::create(const uint64_t &idx, const std::vector<base::samples::Event> &events,
                    const ::eds::calib::CameraInfo &cam_info, const int &num_levels,
                    const ::base::Affine3d &T,
                    const cv::Size &out_size)
{
    return this->create(idx, events, cam_info.height, cam_info.width, num_levels, T, out_size);
}

void EventFrame::create(const uint64_t &idx, const std::vector<base::samples::Event> &events,
            const uint16_t height, const uint16_t width, const int &num_levels,
            const ::base::Affine3d &T, const cv::Size &out_size)
{
    /** Clean before inserting**/
    this->clear();

    /** Input idx and size **/
    this->idx = idx; this->height = height; this->width = width; this->T_w_ef = T;

    /** Get the coordinates, undistort coordinates, events and polarity **/
    for (auto it=events.begin(); it!=events.end(); ++it)
    {
        this->coord.push_back(cv::Point2d(it->x, it->y));
        this->undist_coord.push_back(cv::Point2d(this->fwd_mapx.at<float>(it->y, it->x),
                                                 this->fwd_mapy.at<float>(it->y, it->x)));
        this->pol.push_back((it->polarity)?1:-1);
        if (it == events.begin())
            this->first_time = it->ts;
        else if ((it+1) == events.end())
            this->last_time = it->ts;
    }

    if (first_time.toMicroseconds() > last_time.toMicroseconds())
    {
        std::string error_message = std::string("[EVENT_FRAME] FATAL ERROR Event time[0] > event time [N-1] ");
        throw std::runtime_error(error_message);
    }

    /** Frame time as the median event time **/
    auto it = events.begin(); std::advance(it, events.size()/2);
    this->time = it->ts;

    /** Delta time of this event frame **/
    this->delta_time = (last_time - first_time);

    /**  Compute brightness change event frame (undistorted) per pyramid level **/
    cv::Mat event_img = eds::utils::drawValuesPoints(this->undist_coord, this->pol, height, width, "bilinear", 0.5, true);
    cv::Size size = event_img.size();

    if (this->out_scale[0] != 1 || this->out_scale[1] != 1)
    {
        this->width /= this->out_scale[0]; this->height /= this->out_scale[1];
        cv::resize(event_img, event_img, out_size, cv::INTER_CUBIC);
        size = event_img.size();
    }

    this->frame.push_back(event_img);
    for (int i=1; i<num_levels; ++i)
    {
        cv::Mat event_img_dilate, event_img_erode;
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*i + 1, 2*i + 1), cv::Point(i, i));
        cv::dilate(event_img, event_img_dilate, element);
        cv::erode(event_img, event_img_erode, element);
        this->frame.push_back(event_img_dilate + event_img_erode);
    }

    /** Norm of the event frame **/
    for (auto f=frame.begin(); f!=frame.end(); ++f)
    {
        double n_f = cv::norm (*f);
        this->norm.push_back(n_f);
    }

    /** Normalized event frame **/
    int id = 0; this->event_frame.clear();
    for (auto &it : this->frame)
    {
        std::vector<double> frame_item;
        for (int row=0; row<it.rows; row++)
        {
            for (int col=0; col<it.cols;col++)
            {
                /** When using PhotometricError cost function  **/
                frame_item.push_back(it.at<double>(row,col)/this->norm[id]);
                /** When using PhotometricErrorNC cost function **/
                //frame_item.push_back(it.at<double>(row,col));
            }
        }
        this->event_frame.push_back(frame_item);
        ++id;
    }
    std::cout<<"[EVENT_FRAME] Created ID["<<this->idx<<"] with: "<<events.size()
                <<" events. start time "<<first_time.toMicroseconds()<<" end time "
                <<last_time.toMicroseconds()<<std::endl;
    std::cout<<"[EVENT_FRAME] event frame ["<<std::addressof(this->event_frame)<<"] size:"<<this->event_frame.size()<<" image size[0]: "<<this->event_frame[0].size()<<std::endl;

}

void EventFrame::clear()
{
    this->coord.clear();
    this->undist_coord.clear();
    this->pol.clear();
    this->frame.clear();
    this->event_frame.clear();
    this->norm.clear();
}

cv::Mat EventFrame::viz(size_t id, bool color)
{
    assert(id < this->frame.size());
    cv::Mat events_viz;
    double min, max;
    cv::Mat &img = this->frame[id];
    cv::minMaxLoc(img, &min, &max);
    cv::Mat norm_img = (img - min)/(max -min);

    norm_img.convertTo(events_viz, CV_8UC1, 255, 0);
    if (color)
    {
        eds::utils::BlueWhiteRed bwr;
        bwr(events_viz, events_viz);
    }

    return events_viz;
}

cv::Mat EventFrame::getEventFrame(const size_t &id)
{
    /* get the image from the std vector **/
    cv::Mat event_mat = cv::Mat(this->height, this->width, CV_64FC1);

    std::memcpy(event_mat.data, this->event_frame[id].data(), this->event_frame[id].size()*sizeof(double));

    return event_mat;
}

cv::Mat EventFrame::getEventFrameViz(const size_t &id, bool color)
{
    cv::Mat img = this->getEventFrame();

    double min, max;
    cv::minMaxLoc(img, &min, &max);
    double bmax = 0.7 * max;
    cv::Mat norm_img = (img + bmax)/(2.0* bmax);//(img - min)/(max -min); //between 0 - 1

    cv::Mat events_viz;
    norm_img.convertTo(events_viz, CV_8UC1, 255, 0);
    if (color)
    {
        eds::utils::BlueWhiteRed bwr;
        bwr(events_viz, events_viz);
    }

    return events_viz;
}

void EventFrame::setPose(const ::base::Transform3d& pose)
{
    this->T_w_ef = pose;
}

::base::Transform3d& EventFrame::getPose()
{
    return this->T_w_ef;
}

::base::Matrix4d EventFrame::getPoseMatrix()
{
    return this->T_w_ef.matrix();
}

std::pair<Eigen::Vector3d, Eigen::Quaterniond> EventFrame::getTransQuater()
{
    return std::make_pair(this->T_w_ef.translation(), Eigen::Quaterniond(this->T_w_ef.rotation()));
}

cv::Mat EventFrame::epilinesViz (const std::vector<cv::Point2d> &coord, const cv::Mat &F, const size_t &skip_amount)
{
    std::vector<cv::Vec3d> lines;
    cv::computeCorrespondEpilines(coord, 1, F, lines);

    cv::Mat img_color = this->getEventFrameViz(0, false);
    cv::cvtColor(img_color, img_color, cv::COLOR_GRAY2RGB);

    auto it_l = lines.begin();
    for(; it_l < lines.end(); it_l+=skip_amount)
    {
        cv::Vec3d &l = *it_l; //lines[1000]
        cv::Scalar color((double)std::rand() / RAND_MAX * 255,
        (double)std::rand() / RAND_MAX * 255,
        (double)std::rand() / RAND_MAX * 255);
        cv::line(img_color, cv::Point(0, -l[2]/l[1]), cv::Point(img_color.cols, -(l[2] + l[0] * img_color.cols) / l[1]), color);
    }

    return img_color;
}

cv::Mat EventFrame::pyramidViz(const bool &color)
{
    size_t n = this->frame.size();
    cv::Mat img = cv::Mat(n * this->height, this->width, CV_64FC1, cv::Scalar(0));

    for (size_t i=0; i<n; ++i)
    {
        cv::Size size = this->frame[i].size();
        this->frame[i].copyTo(img(cv::Rect(0, i*size.height, size.width, size.height)));
    }

    return ::eds::utils::viz(img);
}