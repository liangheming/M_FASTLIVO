#pragma once
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

using M2D = Eigen::Matrix2d;
using M3D = Eigen::Matrix3d;
using V2D = Eigen::Vector2d;
using V3D = Eigen::Vector3d;
using V4D = Eigen::Vector4d;

using PointType = pcl::PointXYZINormal;
using CloudType = pcl::PointCloud<PointType>;
using PointVec = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

template <typename T>
using Vec = std::vector<T>;

bool esti_plane(PointVec &points, const double &thresh, V4D &out);

float sq_dist(const PointType &p1, const PointType &p2);

struct Config
{
    double scan_resolution = 0.25;
    double map_resolution = 0.5;

    double cube_len = 300;
    double det_range = 60;
    double move_thresh = 1.5;

    double na = 0.01;
    double ng = 0.01;
    double nba = 0.0001;
    double nbg = 0.0001;
    int imu_init_num = 20;
    int near_search_num = 5;
    bool gravity_align = true;
    M3D r_il = M3D::Identity();
    V3D t_il = V3D::Zero();
    M3D r_cl = M3D::Identity();
    V3D t_cl = V3D::Zero();

    bool esti_li = false;
    bool esti_ci = false;

    double cam_width = 640;
    double cam_height = 512;
    double cam_fx = 431.795259219;
    double cam_fy = 431.550090267;
    double cam_cx = 310.833037316;
    double cam_cy = 266.985989326;
    Vec<double> cam_d{-0.0944205499243979, 0.0946727677776504, -0.00807970960613932, 8.07461209775283e-05, 0.0};
    double patch_size = 8;
    double grid_size = 32;
    double selector_scan_resolution = 0.2;
    double selector_voxel_size = 0.5;
};

struct IMUData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    V3D acc;
    V3D gyro;
    double timestamp;

    IMUData() = default;
    IMUData(const V3D &a, const V3D &g, double &d) : acc(a), gyro(g), timestamp(d) {}
};

struct Pose
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    V3D acc;
    V3D gyro;
    M3D rot;
    V3D pos;
    V3D vel;
    double offset;
    Pose();
    Pose(double t, V3D a, V3D g, V3D v, V3D p, M3D r) : offset(t), acc(a), gyro(g), vel(v), pos(p), rot(r) {}
};

struct SyncPackage
{
    Vec<IMUData> imus;
    CloudType::Ptr cloud;
    cv::Mat image;
    double image_time = 0.0;
    double cloud_start_time = 0.0;
    double cloud_end_time = 0.0;
    bool lidar_end = false;
};
