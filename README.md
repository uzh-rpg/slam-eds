
Event-aided Direct Sparse Odometry (EDS)
=============
Direct visual odometry approach using events and frames (CVPR 2022)

License
-------
See LICENSE file


Installation
------------
The easiest way to build and install this package is to use Rock's build system.
See [this page](http://rock-robotics.org/documentation/installation.html)
on how to install Rock. You can use the [EDS builconf](https://github.com/uzh-rpg/eds-buildconf)
which tells you how to install and use this library from scratch. There is also the option
to build and use this library in a Docker container.

However, if you feel that it's too heavy for your needs, Rock aims at having
most of its "library" packages (such as this one) to follow best practices. See
[this page](http://rock-robotics.org/documentation/packages/outside_of_rock.html)
for installation instructions outside of Rock. It means this library and
its dependencies are standard C++ libraries. Therefore you can build your own
EDS system based on this code which is out of the Rock ecosystem.

Dependencies
-----------------
This is an standard C++ library which generates a shared library by default.
Dependencies are listed in the manifest file, those are:

* [base/cmake](https://github.com/rock-core/base-cmake): the CMake pure function to build this library
* [base/types](https://github.com/rock-core/base-types): C++ basic types (types depends on std C++ and Eigen)
* [base/logging](https://github.com/rock-core/base-logging): C++ library for logging (similar to Google glog)
* [slam/ceres_solver](https://github.com/ceres-solver/ceres-solver): the Ceres solver library
* [slam/pcl](https://pointclouds.org): the point cloud library
* [opencv](https://github.com/opencv/opencv/tree/4.2.0): Open source Computer Vision library. Ubuntu 20.04 uses 4.2.0
* [yaml-cpp](https://github.com/jbeder/yaml-cpp): YAML config parser for the configuration parameters


Rock CMake Macros
-----------------
This package uses a set of CMake helper shipped as the Rock CMake macros.
Documentations is available on [this page](http://rock-robotics.org/documentation/packages/cmake_macros.html).
These macros are pure CMake functions, so they are totally independent of the Rock
ecosystem.

Library Standard Layout
--------------------
This directory structure follows some simple rules, to allow for generic build
processes and simplify reuse of this project. Following these rules ensures that
the Rock CMake macros automatically handle the project's build process and
install setup properly.

### EDS Folder Structure

| directory         |       purpose                                                        |
| ----------------- | ------------------------------------------------------               |
| src/              | Contains all header (*.h/*.hpp) and source files                     |
| src/bundles       | This is the backend optimation from DSO                              |
| src/init          | This is the DSO initializer class                                    |
| src/io            | I/O wrapper from images and maps different types                     |
| src/mapping       | This is the pixel selector based on DSO                              |
| src/sophus        | The Sophus template header-only library based on Eigen               |
| src/tracking      | This is the EDS tracker to perform Events to Model alignment         |
| src/utils         | Some utils functions                                                 |
| build/ *          | The target directory for the build process, temporary content        |
| test/             | Boost Unit test to check basic class functionalities                 |
