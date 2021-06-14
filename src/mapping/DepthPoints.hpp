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

#ifndef _EDS_MAPPING_DEPTH_POINTS_HPP_
#define _EDS_MAPPING_DEPTH_POINTS_HPP_

#include <eds/utils/Utils.hpp>
#include <eds/mapping/Config.hpp>
#include <eds/bundles/Config.hpp>
#include <eds/mapping/Types.hpp>


namespace eds { namespace mapping {

enum DEPTH_FILTER{VOGIATZIS, GAUSS};

class DepthPoints
{
public:
    static constexpr double px_noise = 3.0;
    typedef typename Eigen::Vector4d data_type;
    typedef typename std::vector<data_type> vector_type;
    typedef typename vector_type::iterator iterator;
    typedef typename vector_type::const_iterator const_iterator;

private:
    /** Intrinsic camera matrix **/
    cv::Mat K_; double fx_, fy_, cx_, cy_;
    /** Depth mean range **/
    double mu_range, seed_mu_range;
    /** Converge sigma2 threshold **/
    double convergence_sigma2_thresh;
    /** Pixel error angle **/
    double px_error_angle;
    /* Mu mean inv depth **/
    vector_type data; //vector of 4x1 elements [mu, sigma2, a, b]

public:
    /** @brief Default constructor */
    DepthPoints();

    /** @brief Default constructor */
    DepthPoints(const cv::Mat &K, const uint32_t &num_points, const double &min_depth, const double &max_depth, const double &threshold=100,
                const double &init_a=2.0, const double &init_b=5.0);

    /** @brief Default constructor */
    DepthPoints(const cv::Mat &K, const std::vector<double> &inv_depth, const double &min_depth, const double &max_depth, const double &threshold=100,
                const double &init_a=2.0, const double &init_b=5.0);

    /** Initialization **/
    void init(const cv::Mat &K, const uint32_t &num_points, const double &min_depth, const double &max_depth, const double &threshold=100,
                const double &init_a=2.0, const double &init_b=5.0);

    /** Initialization **/
    void init(const cv::Mat &K, const std::vector<double> &inv_depth, const double &min_depth, const double &max_depth, const double &threshold=100,
                const double &init_a=2.0, const double &init_b=5.0);

    /** Update method given event frame coordinates **/
    void update(const ::base::Transform3d &T_kf_ef, const std::vector<cv::Point2d> &kf_coord, const std::vector<cv::Point2d> &ef_coord,
                const eds::mapping::DEPTH_FILTER &filter = VOGIATZIS);

    /** Update method given points tracks **/
    void update(const ::base::Transform3d &T_kf_ef, const std::vector<cv::Point2d> &kf_coord, const std::vector<Eigen::Vector2d> &tracks,
                const eds::mapping::DEPTH_FILTER &filter = VOGIATZIS);

    /** Vogiatzis filter **/
    bool filterVogiatzis(const double &z, const double &tau2, const double &mu_range,  data_type &state);
 
    /** Get mu inverse depth **/
    void getIDepth(std::vector<double> &x);

    /** Return inverse depth **/
    std::vector<double> getIDepth();

    /** Median and Q_3 (third quantile) inverse depth **/
    void meanIDepth(double &mean, double &st_dev);

    /** Median and Q_3 (third quantile) inverse depth **/
    void medianIDepth(double &median, double &third_q);

    double depthRange(){return this->mu_range;};
    cv::Mat &K(){return this->K_;};
    double &fx(){return this->fx_;};
    double &fy(){return this->fy_;};
    double &cx(){return this->cx_;};
    double &cy(){return this->cy_;};

    /** Overloading [] operator to access elements in array style **/
    data_type& operator[](size_t);

    /** Size with number of points **/
    size_t size(){return this->data.size();};
    bool empty(){return this->data.empty();};
    inline iterator begin() noexcept {return this->data.begin();};
    inline const_iterator cbegin() const noexcept {return this->data.cbegin();};
    inline iterator end() noexcept {return this->data.end();};
    inline const_iterator cend() const noexcept {return this->data.cend();};
    inline iterator erase(const int &idx){return this->data.erase(this->data.begin()+idx);};
    inline iterator erase(DepthPoints::iterator &it) {return this->data.erase(it);};

    /** Visualize the points sigma in a given image **/
    cv::Mat sigmaViz(const cv::Mat &img, const std::vector<cv::Point2d> &coord, double &min_sigma, double &max_sigma);

    /** Visualize the points convergence in a given image **/
    cv::Mat convergenceViz(const cv::Mat &img, const std::vector<cv::Point2d> &coord);

private:
    /*
    *  \param[in] pox1 3x1 matrix: homogeneous coords of the 2D point in image 1 (Key Frame)
    *  \param[in] pox2 3x1 matrix: homogeneous coords of the 2D point in image 2 (Event Frame)
    *  \param[in] poP1 3x4 projection matrix for image 1 (Key Frame). Euclidean (finite) camera!
    *  \param[in] poP2 3x4 projection matrix for image 2 (Event Frame). Euclidean (finite) camera!
    *  \param[out] poX3d 4x1 matrix: homogeneous coordinates of 3D point
    *
    *  \warning It only works for Euclidean camera matrices
    *  \see http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO4/tutorial.html#x1-15002r24
    */
    void linTriangTwoPointsEucl(const cv::Mat& ox1, const cv::Mat& ox2,
                                const cv::Mat& oP1, const cv::Mat& oP2, cv::Mat& oX3d);

    /*
    *  \param[in] pox1 3x1 matrix: homogeneous coords of the 2D point in image 1 (Key Frame)
    *  \param[in] pox2 3x1 matrix: homogeneous coords of the 2D point in image 2 (Event Frame)
    *  \param[in] poP1 3x4 projection matrix for image 1 (Key Frame). Euclidean (finite) camera!
    *  \param[in] poP2 3x4 projection matrix for image 2 (Event Frame). Euclidean (finite) camera!
    *  \param[out] inv_depth double inverse depth of the 3D point in camera 1 (Key Frame)
    *
    *  \warning It only works for Euclidean camera matrices
    *  \see http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO4/tutorial.html#x1-15002r24
    */
    void invDepthTwoPointsEucl(const cv::Mat& ox1, const cv::Mat& ox2,
                                const cv::Mat& oP1, const cv::Mat& oP2, double& inv_depth);

    double getAngleError(double img_err) const
    {
       return std::atan(img_err/(2.0*this->fx_)) + std::atan(img_err/(2.0*this->fy_));
    };


    double computeTau( const ::base::Transform3d& T_ref_cur,
        const Eigen::Vector2d& x_norm,
        const double z,
        const double px_error_angle)
    {
        const Eigen::Vector3d& t = T_ref_cur.translation();
        Eigen::Vector3d x_bearing (x_norm[0], x_norm[1], 1.0);
        x_bearing /= x_bearing.norm(); //unit norm vector
        const Eigen::Vector3d a = x_bearing*z-t;
        double t_norm = t.norm();
        double a_norm = a.norm();
        double alpha = std::acos(x_bearing.dot(t)/t_norm); // dot product
        double beta = std::acos(a.dot(-t)/(t_norm*a_norm)); // dot product
        double beta_plus = beta + px_error_angle;
        double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
        double z_plus = t_norm*std::sin(beta_plus)/std::sin(gamma_plus); // law of sines
        return (z_plus - z); // tau
    };

    inline double getSigma2FromDepthSigma(const double &depth, const double &depth_sigma)
    {
        const double sigma = 0.5 * (1.0 / std::max(0.000000000001, depth - depth_sigma)
                                - 1.0 / (depth + depth_sigma));
        return sigma * sigma;
    };

    inline bool isConverged(const Eigen::Ref<const data_type>& mu_sigma2_a_b,
                            double mu_range,
                            double sigma2_convergence_threshold)
    {
        /** If initial uncertainty was reduced by factor sigma2_convergence_threshold
        we accept the seed as converged. **/
        const double thresh = mu_range / sigma2_convergence_threshold;
        return (mu_sigma2_a_b(1) < thresh * thresh);
    };
};

inline double mu(const Eigen::Ref<const DepthPoints::data_type>& mu_sigma2_a_b)
{
return mu_sigma2_a_b(0);
}

inline double sigma2(const Eigen::Ref<const DepthPoints::data_type>& mu_sigma2_a_b)
{
return mu_sigma2_a_b(1);
}

inline double a(const Eigen::Ref<const DepthPoints::data_type>& mu_sigma2_a_b)
{
return mu_sigma2_a_b(2);
}

inline double b(const Eigen::Ref<const DepthPoints::data_type>& mu_sigma2_a_b)
{
return mu_sigma2_a_b(3);
}

inline double& mu(DepthPoints::data_type& mu_sigma2_a_b)
{
return mu_sigma2_a_b[0];
}

inline double& sigma2(DepthPoints::data_type& mu_sigma2_a_b)
{
return mu_sigma2_a_b(1);
}

inline double& a(DepthPoints::data_type& mu_sigma2_a_b)
{
return mu_sigma2_a_b(2);
}

inline double& b(DepthPoints::data_type& mu_sigma2_a_b)
{
return mu_sigma2_a_b(3);
}

inline double getMeanRangeFromDepthMinMax(double depth_min, double depth_max)
{
return 1.0 / depth_min;
}


} // mapping namespace
} // end namespace
#endif // _EDS_MAPPING_DEPTH_POINTS_HPP_

