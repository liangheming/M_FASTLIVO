cmake_minimum_required(VERSION 3.0.2)
project(fastlivo)

add_compile_options(-std=c++14)
add_compile_definitions(SOPHUS_USE_BASIC_LOGGING)

set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

add_definitions(-DMP_EN)
add_definitions(-DMP_PROC_NUM=2)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  tf2_ros
  livox_ros_driver2
)

find_package(OpenMP QUIET)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Sophus REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

#
catkin_package(

)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
set(SRC_LIST src/map_builder/commons.cpp
             src/map_builder/ieskf.cpp
             src/map_builder/ikd_Tree.cpp
             src/map_builder/imu_processor.cpp
             src/map_builder/map_builder.cpp
             src/utils.cpp
             )
add_executable(temp_node src/temp_node.cpp ${SRC_LIST})
target_link_libraries(temp_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)


