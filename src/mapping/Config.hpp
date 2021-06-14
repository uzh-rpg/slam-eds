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

#ifndef _EDS_MAPPING_CONFIG_HPP_
#define _EDS_MAPPING_CONFIG_HPP_

#include <yaml-cpp/yaml.h>
#include <stdint.h>

namespace eds { namespace mapping{

    struct Config
    {
        double min_depth;
        double max_depth;
        double convergence_sigma2_thresh;
        bool sor_active;
        uint16_t sor_nb_points;
        double sor_radius;
        int num_desired_points;
        float points_rel_baseline;
    };

    inline ::eds::mapping::Config readMappingConfig(YAML::Node config)
    {
        ::eds::mapping::Config mapping_config;

        mapping_config.min_depth = config["min_depth"].as<double>();
        if (mapping_config.min_depth < 0) mapping_config.min_depth= 1e0-6;
        mapping_config.max_depth = config["max_depth"].as<double>();
        if (mapping_config.max_depth < 0) mapping_config.max_depth= 1e0-6;
        mapping_config.convergence_sigma2_thresh = config["convergence_sigma2_thresh"].as<double>();
        if (mapping_config.convergence_sigma2_thresh < 0) mapping_config.convergence_sigma2_thresh= 10.0;
        YAML::Node sor_config = config["sor"];
        mapping_config.sor_active = sor_config["active"].as<bool>();
        mapping_config.sor_nb_points = sor_config["nb_points"].as<uint16_t>();
        mapping_config.sor_radius = sor_config["radius"].as<double>();
        if (config["num_desired_points"]) mapping_config.num_desired_points = config["num_desired_points"].as<int>();
        else mapping_config.num_desired_points = 2000;
        if (config["points_rel_baseline"]) mapping_config.points_rel_baseline = config["points_rel_baseline"].as<float>();
        else mapping_config.points_rel_baseline = 0.1;

        return mapping_config;
    };

} //mapping namespace
} // end namespace

#endif