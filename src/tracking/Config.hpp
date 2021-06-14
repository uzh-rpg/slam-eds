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

#ifndef _EDS_TRACKING_CONFIG_HPP_
#define _EDS_TRACKING_CONFIG_HPP_

#include <yaml-cpp/yaml.h>
#include <base/Time.hpp>

#include <vector>
#include <string>
#include <numeric>
#include <stdint.h>
#include <iostream>
#include <iomanip>  // std::setprecision()
#include <fstream>

namespace eds { namespace tracking{

    enum LOSS_FUNCTION{NONE, HUBER, CAUCHY};
    enum LINEAR_SOLVER_TYPE{DENSE_QR, DENSE_SCHUR, SPARSE_SCHUR, SPARSE_NORMAL_CHOLESKY};
    enum BOOTSTRAP_TYPE{EIGHT_POINTS, MiDAS};

    struct SolverOptions
    {
        LINEAR_SOLVER_TYPE linear_solver_type;
        int num_threads;
        std::vector<int> max_num_iterations;
        double function_tolerance;
        bool minimizer_progress_to_stdout;
    };

    struct Config
    {
        double percent_points;
        std::string type;
        LOSS_FUNCTION loss_type;
        std::vector<double> loss_params;
        SolverOptions options;
        BOOTSTRAP_TYPE bootstrap; 
    };

    struct TrackerInfo
    {
        base::Time time;
        double meas_time_us;
        uint32_t num_points;
        int num_iterations;
        double time_seconds;
        uint8_t success;
    };

    inline ::eds::tracking::LOSS_FUNCTION selectLoss(const std::string &loss_name)
    {
        if (loss_name.compare("Huber") == 0)
            return eds::tracking::HUBER;
        else if (loss_name.compare("Cauchy") == 0)
            return eds::tracking::CAUCHY;
        else
            return eds::tracking::NONE;
    };

    inline ::eds::tracking::LINEAR_SOLVER_TYPE selectSolver(const std::string &solver_name)
    {
        if (solver_name.compare("DENSE_QR") == 0)
            return eds::tracking::DENSE_QR;
        else if (solver_name.compare("DENSE_SCHUR") == 0)
            return eds::tracking::DENSE_SCHUR;
        else if (solver_name.compare("SPARSE_SCHUR") == 0)
            return eds::tracking::SPARSE_SCHUR;
        else
            return eds::tracking::SPARSE_NORMAL_CHOLESKY;
    };

    inline ::eds::tracking::Config readTrackingConfig(YAML::Node config)
    {
        ::eds::tracking::Config tracker_config;

        /** Number of points per frame **/
        tracker_config.percent_points = config["percent_points"].as<double>();
        /** Config for tracker type (only ceres) **/
        tracker_config.type = config["type"].as<std::string>();

        if (config["bootstrapping"])
        {
            std::string bootstrapping = config["bootstrapping"].as<std::string>();
            if (bootstrapping.compare("EIGHT_POINTS") == 0)
                tracker_config.bootstrap = eds::tracking::EIGHT_POINTS;
            else if (bootstrapping.compare("MiDAS") == 0)
                tracker_config.bootstrap = eds::tracking::MiDAS;
            else
                tracker_config.bootstrap = eds::tracking::EIGHT_POINTS;
        }
        else
            tracker_config.bootstrap = eds::tracking::EIGHT_POINTS;

        /** Config the loss **/
        YAML::Node tracker_loss = config["loss_function"];
        std::string loss_name = tracker_loss["type"].as<std::string>();
        tracker_config.loss_type = eds::tracking::selectLoss(tracker_loss["type"].as<std::string>());
        tracker_config.loss_params = tracker_loss["param"].as< std::vector<double> >();
    
        /** Config for ceres options **/
        YAML::Node tracker_options = config["options"];
        tracker_config.options.linear_solver_type = eds::tracking::selectSolver(tracker_options["solver_type"].as<std::string>());
        tracker_config.options.num_threads = tracker_options["num_threads"].as<int>();
        tracker_config.options.max_num_iterations = tracker_options["max_num_iterations"].as< std::vector<int> > ();
        tracker_config.options.function_tolerance = tracker_options["function_tolerance"].as<double>();
        tracker_config.options.minimizer_progress_to_stdout = tracker_options["minimizer_progress_to_stdout"].as<bool>();

        return tracker_config;
    };

} // tracking namespace
} // end namespace

#endif