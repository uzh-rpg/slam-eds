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

#ifndef _EDS_MAPPING_OPTIMIZER_HPP_
#define _EDS_MAPPING_OPTIMIZER_HPP_

#include <eds/utils/Utils.hpp>
#include <eds/mapping/Types.hpp>
#include <eds/mapping/GlobalMap.hpp>
#include <chrono>

#include <eds/bundles/Config.hpp>

namespace eds { namespace bundles {
 
enum LOSS_PARAM_METHOD{CONSTANT, MAD, STD};

struct ResidualInfo
{
    uint64_t kf_id;
    ::eds::mapping::Point2d *coord;
    double *residual;
    ResidualInfo(const uint64_t &id_, ::eds::mapping::Point2d* coord_, double *residual_)
                :kf_id(id_), coord(coord_), residual(residual_){};
};

class BundleAdjustment
{
    private:
        /** Configuration **/
        ::eds::bundles::Config config;

        /**Get the min and max point depth
        It is used as constraint in the BA **/
        double idp_lower_bound, idp_upper_bound;

        /** Keyframes euclidian score dictionary (for marginalization criteria) **/
        std::map<uint64_t, double> score_dict;

        /** Visible point (%) in KF id from other KF in the window **/
        std::map<uint64_t, double > visible_points;

        /** Status Information **/
        eds::bundles::PBAInfo info;

    public:
        /** Residuals {vector of residuals} **/
        std::vector< std::vector<double> > residuals;

    public:
        /** @brief Default constructor */
        BundleAdjustment(const eds::bundles::Config &config, double &min_depth, double &max_depth);

        bool optimize(std::shared_ptr<eds::mapping::GlobalMap> &global_map, uint64_t &kf_to_marginalize,
                                eds::bundles::LOSS_PARAM_METHOD loss_param_method = eds::bundles::LOSS_PARAM_METHOD::MAD,
                                const bool &is_initialized = true);

        uint64_t selecKFToMarginalize(const ::eds::mapping::CameraKFs &camera_kfs, const ::eds::mapping::PointKFs &point_kfs,
                                const uint64_t &last_kf_id, const uint64_t &prev_last_kf_id, const int &num_points_per_kf,
                                const float &percent = 20.0);

        void euclideanScore(const ::eds::mapping::CameraKFs &kfs_dict,
                            std::map<uint64_t, double> &score_dict,
                            const uint64_t &last_kf_id, const uint64_t &prev_last_kf_id,
                            const double epsilon=0.01);

        std::vector<double> getLossParams(std::vector<double> &residuals, eds::bundles::LOSS_PARAM_METHOD method=CONSTANT);

        ::eds::bundles::PBAInfo getInfo();
};

} // mapping namespace
} // end namespace
#endif // _EDS_MAPPING_OPTIMIZER_HPP_
