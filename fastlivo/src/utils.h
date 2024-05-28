#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/core.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/PointCloud2.h>
// #include <livox_ros_driver2/CustomMsg.h>
#include <livox_ros_driver/CustomMsg.h>

#include <pcl_conversions/pcl_conversions.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>

// void livox2pcl(const livox_ros_driver2::CustomMsg::ConstPtr &msg, pcl::PointCloud<pcl::PointXYZINormal>::Ptr out, int filter_num, double blind);
cv::Mat msg2cv(const sensor_msgs::ImageConstPtr &img_msg);

void livox2pcl(const livox_ros_driver::CustomMsg::ConstPtr &msg, pcl::PointCloud<pcl::PointXYZINormal>::Ptr out, int filter_num, double blind, double max_range = 30.0);

void livoxAvia2pcl(const livox_ros_driver::CustomMsg::ConstPtr &msg, pcl::PointCloud<pcl::PointXYZINormal>::Ptr out, int filter_num, double blind, double max_range = 30);

sensor_msgs::PointCloud2 pcl2msg(pcl::PointCloud<pcl::PointXYZINormal>::Ptr inp, const std::string &frame_id, const double &timestamp);

geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);