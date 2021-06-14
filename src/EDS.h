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

/** Settings **/
#include <eds/utils/settings.h>

/** Utils **/
#include <eds/utils/Utils.hpp>
#include <eds/utils/Calib.hpp>
#include <eds/utils/NumType.h>
#include <eds/utils/globalFuncs.h>
#include <eds/utils/globalCalib.h>
#include <eds/utils/Undistort.h>
#include <eds/utils/ImageAndExposure.h>
#include <eds/utils/FrameShell.h>
#include <eds/utils/IndexThreadReduce.h>

/** I/O **/
#include <eds/io/ImageRW.h>
#include <eds/io/ImageConvert.h>
#include <eds/io/OutputMaps.h>

/** Initialization **/
#include <eds/init/CoarseInitializer.h>

/** Event Tracker (EDS) **/
#include <eds/tracking/EventFrame.hpp>
#include <eds/tracking/KeyFrame.hpp>
#include <eds/tracking/Tracker.hpp>

/** Frame Tracker (DSO) **/
#include <eds/tracking/Residuals.h>
#include <eds/tracking/HessianBlocks.h>
#include <eds/tracking/ImmaturePoint.h>
#include <eds/tracking/CoarseTracker.h>

/** Mapping **/
#include <eds/mapping/Config.hpp>
#include <eds/mapping/Types.hpp>
#include <eds/mapping/PixelSelector.h>

/** Bundles **/
#include <eds/bundles/Config.hpp>
#include <eds/bundles/EnergyFunctional.h>
#include <eds/bundles/EnergyFunctionalStructs.h>
#include <eds/bundles/MatrixAccumulators.h>

#endif //EDS_EDS_H