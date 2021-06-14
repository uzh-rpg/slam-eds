# CMakeLists.txt has to be located in the project folder and cmake has to be
# executed from 'project/build' with 'cmake ../'.
cmake_minimum_required(VERSION 3.9)
project(eds
        VERSION 0.1
        DESCRIPTION "Event-based Sparse Odometry")
find_package(Rock)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ROCK_TEST_ENABLED OFF)

if(COVERAGE)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        add_definitions(-fprofile-arcs -ftest-coverage)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -lgcov")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -lgcov")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -lgcov")

        add_custom_target(coverage
          COMMAND lcov --directory . --capture --output-file cov.info
          COMMAND lcov --remove cov.info 'tests/*' '/usr/*' '*/install/include/*' --output-file cov.info.cleaned
          COMMAND genhtml -o ./cov cov.info.cleaned
          COMMAND cmake -E remove cov.info cov.info.cleaned
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    else()
        message(FATAL_ERROR "Code coverage only works in Debug versions" )
    endif()
endif()

rock_init()
rock_opencv_autodetect(OPENCV_PACKAGE)
rock_standard_layout()
