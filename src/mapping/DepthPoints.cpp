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

#include "DepthPoints.hpp"

#include <iostream>
#include <opencv2/core/eigen.hpp>
using namespace eds::mapping;

DepthPoints::DepthPoints()
{
    /** Undifined values constructor **/
    this->K_ = cv::Mat_<double>(3,3) * ::base::NaN<double>();
    this->fx_ = this->K_.at<double>(0,0); this->fy_ = this->K_.at<double>(1,1);
    this->cx_ = this->K_.at<double>(0,2); this->cy_ = this->K_.at<double>(1,2);
    this->mu_range  = ::base::NaN<double>();
    this->px_error_angle = this->getAngleError(this->px_noise);
    this->seed_mu_range = ::base::NaN<double>();
    this->convergence_sigma2_thresh = ::base::NaN<double>();
    this->data.clear();
}

DepthPoints::DepthPoints(const cv::Mat &K, const uint32_t &num_points, const double &min_depth, const double &max_depth,
                        const double &threshold, const double &init_a, const double &init_b)
{
    this->init(K, num_points, min_depth, max_depth, threshold, init_a, init_b);
}

DepthPoints::DepthPoints(const cv::Mat &K, const std::vector<double> &inv_depth, const double &min_depth, const double &max_depth,
                const double &threshold, const double &init_a, const double &init_b)
{
    this->init(K, inv_depth, min_depth, max_depth, threshold, init_a, init_b);
}

void DepthPoints::init(const cv::Mat &K, const uint32_t &num_points, const double &min_depth, const double &max_depth,
                        const double &threshold, const double &init_a, const double &init_b)
{
    this->K_ = K.clone();
    this->fx_ = K.at<double>(0,0); this->fy_ = K.at<double>(1,1);
    this->cx_ = K.at<double>(0,2); this->cy_ = K.at<double>(1,2);
    this->mu_range  = max_depth-min_depth;
    this->px_error_angle = this->getAngleError(this->px_noise);
    this->seed_mu_range = eds::mapping::getMeanRangeFromDepthMinMax(min_depth, max_depth);
    this->convergence_sigma2_thresh = threshold;
    std::cout<<"[DEPTH_POINTS] px_noise: "<<this->px_noise<<" px_error_angle: "<<this->px_error_angle<<std::endl;
    std::cout<<"[DEPTH_POINTS] init_depth: "<<(max_depth-min_depth)/2.0<<" mu_range: "<<this->mu_range<<" init_sigma2: "<<this->mu_range * this->mu_range<< std::endl;

    /** Initial inverse depth to mean depth, initial sigma_2 is mu_range^2 is the range when we don't have much information about depth**/
    data_type init_value; init_value << 1.0/((max_depth-min_depth)/2.0), this->mu_range * this->mu_range, init_a, init_b /*this is probability of outliers*/;
    this->data.clear(); this->data.resize(num_points, init_value);
}

void DepthPoints::init(const cv::Mat &K, const std::vector<double> &inv_depth, const double &min_depth, const double &max_depth,
                const double &threshold, const double &init_a, const double &init_b)
{
    this->K_ = K.clone();
    this->fx_ = K.at<double>(0,0); this->fy_ = K.at<double>(1,1);
    this->cx_ = K.at<double>(0,2); this->cy_ = K.at<double>(1,2);
    this->mu_range  = max_depth-min_depth;
    this->px_error_angle = this->getAngleError(this->px_noise);
    this->seed_mu_range = eds::mapping::getMeanRangeFromDepthMinMax(min_depth, max_depth);
    this->convergence_sigma2_thresh = threshold;
    std::cout<<"[DEPTH_POINTS] px_noise: "<<this->px_noise<<" px_error_angle: "<<this->px_error_angle<<std::endl;

    /** Depth is comming from the global map, we set the initial sigma^2 = small mu_range
     * *TO-DO: make the inital sigma^2 as an argument with different */
    this->data.clear();
    double init_sigma2 = (this->mu_range * this->mu_range)/36.0;
    std::cout<<"[DEPTH_POINTS] mu_range: "<<this->mu_range<<" init_sigma2: "<<init_sigma2<< std::endl;
    for (auto it : inv_depth)
    {
        this->data.push_back(data_type(it, init_sigma2, init_a, init_b));
    }
}

void DepthPoints::update(const ::base::Transform3d &T_kf_ef, const std::vector<cv::Point2d> &kf_coord, const std::vector<cv::Point2d> &ef_coord,
                const eds::mapping::DEPTH_FILTER &filter)
{
    assert(this->data.size() == kf_coord.size());
    assert(kf_coord.size() == ef_coord.size());

    /** Projection matrix for Keyframe K[I|0] **/
    cv::Mat P_kf; cv::hconcat(this->K_, cv::Vec3d(0.0, 0.0, 0.0), P_kf);

    /** Projection matrix for Eventframe K[R|t] **/
    cv::Mat T_ef_kf; cv::eigen2cv(T_kf_ef.inverse().matrix(), T_ef_kf); //3x4 matrix
    T_ef_kf = T_ef_kf.colRange(0, 4).rowRange(0, 3); //3x4 matrix
    cv::Mat P_ef = this->K_ * T_ef_kf;//3x4 matrix

    std::cout<<"[DEPTH_POINTS] P_kf:\n"<<P_kf<<std::endl;
    std::cout<<"[DEPTH_POINTS] P_ef:\n"<<P_ef<<std::endl;
    std::cout<<"[DEPTH_POINTS] T_ef_kf:\n"<<T_ef_kf<<std::endl;
    cv::Mat t_ef_kf = T_ef_kf.colRange(3, 4).rowRange(0, 3);
    std::cout<<"[DEPTH_POINTS] t_ef_kf:\n"<<t_ef_kf<<std::endl;

    auto it_data = this->data.begin();
    auto it_kf = kf_coord.begin();
    auto it_ef = ef_coord.begin();
    for (; it_kf != kf_coord.end() && it_ef != ef_coord.end(); ++it_kf, ++it_ef, ++it_data)
    {
        /** Calculate the inverse depth from traingulation **/
        cv::Vec3d x_kf ((*it_kf).x, (*it_kf).y, 1.0);
        cv::Vec3d x_ef ((*it_ef).x, (*it_ef).y, 1.0);
        double inv_depth;
        this->invDepthTwoPointsEucl(cv::Mat(x_kf, false), cv::Mat(x_ef, false), P_kf, P_ef, inv_depth);
        double depth = 1.0/inv_depth;

        /** Uncertainty depth sigma (tau) **/
        Eigen::Vector2d x_norm((x_ef[0]-this->cx_)/this->fx_, (x_ef[1]-this->cy_)/this->fy_);
        double depth_sigma = this->computeTau(T_kf_ef, x_norm, depth, this->px_error_angle);
        //std::cout<<"depth["<<depth<<"] tau["<<depth_sigma<<"] sigma2 ["<<this->getSigma2FromDepthSigma(depth, depth_sigma)<<
        //            "] x_norm["<<x_norm[0]<<","<<x_norm[1]<<","<<x_norm[2]<<"] converge: "<< this->isConverged(*it_data, this->mu_range, this->convergence_sigma2_thresh)<<"\n";

        /** Update estimates using the filter **/
        this->filterVogiatzis(inv_depth, this->getSigma2FromDepthSigma(depth, depth_sigma), this->mu_range, *it_data);
        //eds::mapping::mu(*it_data) = inv_depth;
    }
}

void DepthPoints::update(const ::base::Transform3d &T_kf_ef, const std::vector<cv::Point2d> &kf_coord, const std::vector<Eigen::Vector2d> &tracks,
                const eds::mapping::DEPTH_FILTER &filter)
{
    assert(this->data.size() == kf_coord.size());
    assert(kf_coord.size() == tracks.size());

    /** Projection matrix for Keyframe K[I|0] **/
    cv::Mat P_kf; cv::hconcat(this->K_, cv::Vec3d(0.0, 0.0, 0.0), P_kf);

    /** Projection matrix for Eventframe K[R|t] **/
    cv::Mat T_ef_kf; cv::eigen2cv(T_kf_ef.inverse().matrix(), T_ef_kf); //3x4 matrix
    T_ef_kf = T_ef_kf.colRange(0, 4).rowRange(0, 3); //3x4 matrix
    cv::Mat P_ef = this->K_ * T_ef_kf;//3x4 matrix

    std::cout<<"[DEPTH_POINTS] P_kf:\n"<<P_kf<<std::endl;
    std::cout<<"[DEPTH_POINTS] P_ef:\n"<<P_ef<<std::endl;
    std::cout<<"[DEPTH_POINTS] T_ef_kf:\n"<<T_ef_kf<<std::endl;
    cv::Mat t_ef_kf = T_ef_kf.colRange(3, 4).rowRange(0, 3);
    std::cout<<"[DEPTH_POINTS] t_ef_kf:\n"<<t_ef_kf<<std::endl;

    auto it_data = this->data.begin();
    auto it_kf = kf_coord.begin();
    auto it_tr = tracks.begin();
    for (; it_kf != kf_coord.end() && it_tr != tracks.end(); ++it_kf, ++it_tr, ++it_data)
    {
        /** Calculate the inverse depth from traingulation **/
        cv::Vec3d x_kf ((*it_kf).x, (*it_kf).y, 1.0);
        cv::Vec3d x_ef ((*it_kf).x + (*it_tr)[0], (*it_kf).y + (*it_tr)[1], 1.0);
        double inv_depth;
        this->invDepthTwoPointsEucl(cv::Mat(x_kf, false), cv::Mat(x_ef, false), P_kf, P_ef, inv_depth);
        double depth = 1.0/inv_depth;

        /** Uncertainty depth sigma (tau) **/
        Eigen::Vector2d x_norm((x_ef[0]-this->cx_)/this->fx_, (x_ef[1]-this->cy_)/this->fy_);
        double depth_sigma = this->computeTau(T_kf_ef, x_norm, depth, this->px_error_angle);
        //std::cout<<"depth["<<depth<<"] tau["<<depth_sigma<<"] sigma2 ["<<this->getSigma2FromDepthSigma(depth, depth_sigma)<<
        //            "] x_norm["<<x_norm[0]<<","<<x_norm[1]<<","<<x_norm[2]<<"] converge: "<< this->isConverged(*it_data, this->mu_range, this->convergence_sigma2_thresh)<<"\n";

        /** Update estimates using the filter **/
        this->filterVogiatzis(inv_depth, this->getSigma2FromDepthSigma(depth, depth_sigma), this->mu_range, *it_data);
    }
}

bool DepthPoints::filterVogiatzis(const double &z, const double &tau2, const double &mu_range, data_type &state)
{
    double &mu = ::eds::mapping::mu(state);
    double &sigma2 = ::eds::mapping::sigma2(state);
    double &a = ::eds::mapping::a(state);
    double &b = ::eds::mapping::b(state);

    const double norm_scale = std::sqrt(sigma2 + tau2);
    if(std::isnan(norm_scale))
    {
        std::cout<<"[VOGIATZIS] Update Seed: Sigma2+Tau2 is NaN"<<std::endl;
        return false;
    }

    const double oldsigma2 = sigma2;
    const double s2 = 1.0/(1.0/sigma2 + 1.0/tau2);
    const double m = s2*(mu/sigma2 + z/tau2);
    const double uniform_x = 1.0/mu_range;

    double C1 = a/(a+b) * ::eds::utils::normPdf<double>(z, mu, norm_scale);
    double C2 = b/(a+b) * uniform_x;
    const double normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    const double f = C1*(a+1.0)/(a+b+1.0) + C2*a/(a+b+1.0);
    const double e = C1*(a+1.0)*(a+2.0)/((a+b+1.0)*(a+b+2.0))
                    + C2*a*(a+1.0)/((a+b+1.0)*(a+b+2.0));

    // update parameters
    const double mu_new = C1*m+C2*mu;
    sigma2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_new*mu_new;
    mu = mu_new;
    a = (e - f) / (f - e / f);
    b = a * (1.0 - f) / f;

    // TODO: This happens sometimes.
    if(sigma2 < 0.0)
    {
        std::cout<<"[VOGIATZIS] Seed sigma2 is negative!"<<std::endl;
        sigma2 = oldsigma2;
    }
    if(mu < 0.0)
    {
        std::cout<<"[VOGIATZIS] Seed diverged! mu is negative!!"<<std::endl;
        mu = 1.0;
        return false;
    }
    return true;
}

void DepthPoints::getIDepth(std::vector<double> &x)
{
    x.clear();
    for (auto &it: this->data)
    {
        x.push_back(::eds::mapping::mu(it));
    }
}

std::vector<double> DepthPoints::getIDepth()
{
    std::vector<double> x;
    for (auto &it: this->data)
    {
        x.push_back(::eds::mapping::mu(it));
    }
    return x;
}

void DepthPoints::meanIDepth(double &mean, double &st_dev)
{
    std::vector<double> x = this->getIDepth();
    ::eds::utils::mean_std_vector(x, mean, st_dev);
}

void DepthPoints::medianIDepth(double &median, double &third_q)
{
    std::vector<double> x = this->getIDepth();
    median = eds::utils::n_quantile_vector(x, x.size()/2);
    third_q = eds::utils::n_quantile_vector(x, x.size()/3);
}

// Implementation of [] operator.  This function must return a
// reference as array element can be put on left side
DepthPoints::data_type& DepthPoints::operator[](size_t index)
{
    if (index >= this->data.size())
    {
        std::cout<< "[ERROR] index out of bound, exiting"<<std::endl;
        exit(0);
    }
    return this->data[index];
}

cv::Mat DepthPoints::sigmaViz(const cv::Mat &img, const std::vector<cv::Point2d> &coord,
                            double &min_sigma, double &max_sigma)
{
    /** Get the sigma **/
    std::vector<double> sigmas;
    for (auto it : this->data)
    {
        sigmas.push_back(std::sqrt(::eds::mapping::sigma2(it)));
    }
    /** get the min and max values **/
    min_sigma = * std::min_element(std::begin(sigmas), std::end(sigmas));
    max_sigma = * std::max_element(std::begin(sigmas), std::end(sigmas));
 
    /** Draw the sigmas **/
    cv::Mat sigma_img = eds::utils::drawValuesPoints(coord, sigmas, img.rows, img.cols, "nn", 0.0);

    /** Convert between 0-1 scale **/
    double min, max;
    cv::minMaxLoc(sigma_img, &min, &max);
    cv::Mat norm_img = (sigma_img - min)/(max-min);

    cv::Mat sigma_viz;
    cv::minMaxLoc(norm_img, &min, &max);
    norm_img.convertTo(sigma_viz, CV_8UC1, 255, 0);
    cv::minMaxLoc(sigma_viz, &min, &max);
    cv::applyColorMap(sigma_viz, sigma_viz, cv::COLORMAP_JET);

    cv::Mat img_color; img.convertTo(img_color, CV_8UC1, 255, 0);
    cv::cvtColor(img_color, img_color, cv::COLOR_GRAY2RGB);
    for (auto it=coord.begin(); it!=coord.end(); ++it)
    {
        cv::Vec3b & point = img_color.at<cv::Vec3b>(*it);
        point = sigma_viz.at<cv::Vec3b>(*it);
    }

    return sigma_viz;
}

cv::Mat DepthPoints::convergenceViz(const cv::Mat &img, const std::vector<cv::Point2d> &coord)
{
    assert(this->data.size() == coord.size());

    /** Get the convergence **/
    std::vector<bool> converge;
    for (auto it : this->data)
    {
        converge.push_back(this->isConverged(it, this->mu_range, this->convergence_sigma2_thresh));
    }

    /** Green color for convergence (true), Red color for not convergence (false) **/
    cv::Vec3b color_positive = cv::Vec3b(0.0, 255.0, 0.0); //Green
    cv::Vec3b color_negative = cv::Vec3b(0.0, 0.0, 255.0); //Red

    /** Write on the image  **/
    cv::Mat img_color; img.convertTo(img_color, CV_8UC1, 255, 0);
    cv::cvtColor(img_color, img_color, cv::COLOR_GRAY2RGB);
    auto it_conv = converge.begin();
    auto it_point = coord.begin();
    for (; it_point!=coord.end(); ++it_point, ++it_conv)
    {
        if (*it_conv)
            img_color.at<cv::Vec3b>(*it_point) = color_positive;
        else
            img_color.at<cv::Vec3b>(*it_point) = color_negative;
    }

    return img_color;
}

void DepthPoints::linTriangTwoPointsEucl(const cv::Mat& ox1, const cv::Mat& ox2,
                                const cv::Mat& oP1, const cv::Mat& oP2, cv::Mat& oX3d)
{
    cv::Mat oM1 = oP1.colRange(0,3); // First 3x3 submatrix of P1
    cv::Mat oM2 = oP2.colRange(0,3); // First 3x3 submatrix of P2
    cv::Mat oInvM1 = oM1.inv(cv::DECOMP_SVD);
    cv::Mat oInvM2 = oM2.inv(cv::DECOMP_SVD);
    // Compute optical center of camera 1
    cv::Mat oC1(4,1,CV_64FC1);
    oC1.at<double>(3,0)=1; // Euclidize it.
    cv::Mat oC1euc = oC1.rowRange(0,3);
    oC1euc = -oInvM1*(oP1.col(3));
    // Compute epipole on image plane 2
    cv::Mat oEpipole2 = oP2 * oC1;
    // Compute projection of point in the plane at infinity
    cv::Mat ox1p = oM2*(oInvM1*ox1);
    // Compute depth of the 3D point with respect to camera 2
    cv::Mat oAux1 = oEpipole2.cross(ox1p);
    cv::Mat oAux2 = ox2.cross(ox1p);
    double depth2 = oAux1.dot(oAux2) / oAux2.dot(oAux2);
    // Compute optical center of camera 2
    cv::Mat oC2(4,1,CV_64FC1);
    oC2.at<double>(3,0)=1; // Euclidize it.
    cv::Mat oC2euc = oC2.rowRange(0,3);
    oC2euc = -oInvM2*(oP2.col(3));
    cv::Mat oXtmp(4,1,CV_64FC1);
    oXtmp.at<double>(3,0)=0;
    cv::Mat oXtmpeuc = oXtmp.rowRange(0,3);
    oXtmpeuc = oInvM2*ox2;
    // Compute 3D point. See Hartley and Zissermann 2004, eq's. (6.13) and (6.14)
    cv::addWeighted(oXtmp,depth2,oC2,1.0,0,oX3d);
}

void DepthPoints::invDepthTwoPointsEucl(const cv::Mat& ox1, const cv::Mat& ox2,
                                const cv::Mat& oP1, const cv::Mat& oP2, double& inv_depth)
{
    cv::Mat oM1 = oP1.colRange(0,3); // First 3x3 submatrix of P1
    cv::Mat oM2 = oP2.colRange(0,3); // First 3x3 submatrix of P2
    cv::Mat oInvM1 = oM1.inv(cv::DECOMP_SVD);
    cv::Mat oInvM2 = oM2.inv(cv::DECOMP_SVD);

    /** Compute optical center of camera 1 (Keyframe) **/
    cv::Mat oC1(4,1,CV_64FC1);
    oC1.at<double>(3,0)=1; // Euclidize it.
    cv::Mat oC1euc = oC1.rowRange(0,3);
    oC1euc = -oInvM1*(oP1.col(3));

    /** Compute epipole on image plane 2 (Eventframe) **/
    cv::Mat oEpipole2 = oP2 * oC1;
    /** Compute projection of point 1 in the plane at infinity **/
    cv::Mat ox1p = oM2*(oInvM1*ox1);

    /** Compute depth of the 3D point with respect to camera 1 (Key frame) **/
    cv::Mat aux1 = ox1p.cross(ox2); //m_l' x m_r
    cv::Mat aux2 = ox2.cross(oEpipole2); //m_r x e_r

    /** Inverse depth in camera 1 **/
    inv_depth = aux1.dot(aux2) / aux2.dot(aux2);
}
