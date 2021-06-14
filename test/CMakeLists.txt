find_package(yaml-cpp REQUIRED)

rock_testsuite(test_initialization
    SOURCES suite.cpp test_initialization.cpp
    DEPS eds yaml-cpp)

rock_testsuite(test_bundles
    SOURCES suite.cpp test_bundles.cpp
    DEPS eds yaml-cpp)

rock_testsuite(test_patches
    SOURCES suite.cpp test_patches.cpp
    DEPS eds yaml-cpp)

rock_testsuite(test_tracking
    SOURCES suite.cpp test_tracking.cpp
    DEPS eds yaml-cpp)

rock_testsuite(test_colormap
    SOURCES suite.cpp test_colormap.cpp
    DEPS eds)

rock_testsuite(test_mapping
    SOURCES suite.cpp test_mapping.cpp
    DEPS eds yaml-cpp)

rock_testsuite(test_yaml
    SOURCES suite.cpp test_yaml.cpp
    DEPS eds yaml-cpp)

rock_testsuite(test_calib
    SOURCES suite.cpp test_calib.cpp
    DEPS eds yaml-cpp)

rock_testsuite(test_sophus
    SOURCES suite.cpp test_sophus.cpp
    DEPS eds yaml-cpp)

rock_testsuite(test_umeyama
    SOURCES suite.cpp test_umeyama.cpp
    DEPS eds yaml-cpp)
