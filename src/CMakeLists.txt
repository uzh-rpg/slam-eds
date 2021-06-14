find_package(OpenCV REQUIRED core imgproc calib3d highgui)
find_package(Ceres REQUIRED)
find_package(PCL 1.7 REQUIRED COMPONENTS common filters)

rock_library(eds
    SOURCES
        tracking/EventFrame.cpp
        tracking/KeyFrame.cpp
        tracking/Tracker.cpp
        mapping/GlobalMap.cpp
        mapping/DepthPoints.cpp
        bundles/BundleAdjustment.cpp
        utils/Utils.cpp
        utils/Calib.cpp
        utils/Colormap.cpp
    HEADERS
        tracking/Types.hpp
        tracking/Config.hpp
        tracking/KeyFrame.hpp
        tracking/EventFrame.hpp
        tracking/PhotometricError.hpp
        tracking/PhotometricErrorNC.hpp
        tracking/Tracker.hpp
        mapping/Config.hpp
        mapping/Types.hpp
        mapping/GlobalMap.hpp
        mapping/DepthPoints.hpp
        bundles/Config.hpp
        bundles/PhotometricBAError.hpp
        bundles/BundleAdjustment.hpp
        utils/Colormap.hpp
        utils/KDTree.hpp
        utils/Utils.hpp
        utils/Calib.hpp
        utils/Config.hpp
        utils/Transforms.hpp
        sophus/rxso3.hpp
        sophus/se2.hpp
        sophus/se3.hpp
        sophus/sim3.hpp
        sophus/so2.hpp
        sophus/so3.hpp
        sophus/sophus.hpp
        EDS.h
    DEPS_TARGET
        ceres
    DEPS_PKGCONFIG
        base-types
        base-logging
        yaml-cpp
        pcl_common-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
        pcl_filters-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
        ${OPENCV_PACKAGE}
        )

