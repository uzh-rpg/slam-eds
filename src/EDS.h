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

#ifndef EDS_EDS_H
#define EDS_EDS_H

/** Base types **/
#include <base/Eigen.hpp>

/** Utils **/
#include "utils/Utils.hpp"
#include "utils/Calib.hpp"

/** Tracking **/
#include "tracking/Config.hpp"
#include "tracking/EventFrame.hpp"
#include "tracking/KeyFrame.hpp"
#include "tracking/PhotometricError.hpp"
#include "tracking/PhotometricErrorNC.hpp"
#include "tracking/Tracker.hpp"

/** Mapping **/
#include "mapping/Types.hpp"
#include "mapping/Config.hpp"
#include "mapping/GlobalMap.hpp"
#include "mapping/DepthPoints.hpp"

/** Bundles **/
#include "bundles/Config.hpp"
#include "bundles/PhotometricBAError.hpp"
#include "bundles/BundleAdjustment.hpp"

#endif //EDS_CORE_H