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

#pragma once
#include <vector>
#include <base/Time.hpp>
#include <base/samples/Frame.hpp>
#include <eds/utils/NumType.h>
#include <eds/utils/MinimalImage.h>


namespace eds
{

namespace io
{

dso::MinimalImageB3* toMinimalImageB3(const Eigen::Vector3f *fd, const int &w, const int &h);
void MinimalImageB3ToFrame(const dso::MinimalImageB3 *input, const ::base::Time &timestamp, ::base::samples::frame::Frame &frame);

}
}
