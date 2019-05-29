# Name of project and executable
project(Project)

# set cmake version
cmake_minimum_required(VERSION 3.1.0)

# activate latest c++ compiler version
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# set flags to configure the warning settings
# Note: warnings are compiler specific
if( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)

# include opencv
include_directories ("/home/guy/opencv-3.4.3/build")
find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})

#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# set build type to Debug/Release
set(CMAKE_BUILD_TYPE "Debug")

# Find the QtWidgets library
find_package(Qt5Widgets CONFIG REQUIRED)
find_package(Qt5PrintSupport REQUIRED)

# include opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# copy needed files
FILE(COPY data DESTINATION ${PROJECT_BINARY_DIR})


# Populate a CMake variable with the sources
set(APP_SOURCES
    src/main.cpp
    src/opencvworker.cpp
    src/qcvwidget.cpp
    src/qcvwidget.ui
    src/yolo.cpp
)
aux_source_directory(. SRC_LIST)

set(APP_INCLUDE_DIRS
    src/
)
INCLUDE_DIRECTORIES(${APP_INCLUDE_DIRS})

# Add an executable to the project and sources
add_executable(${PROJECT_NAME} src/main.cpp ${APP_SOURCES})
# Use the Widgets module from Qt 5
target_link_libraries(${PROJECT_NAME} Qt5::Widgets Qt5::PrintSupport)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
