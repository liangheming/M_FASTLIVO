#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/core.hpp>

namespace livo
{
    typedef pcl::PointXYZINormal PointType;
    typedef pcl::PointCloud<PointType> CloudType;
    typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVec;

    bool esti_plane(PointVec &points, const double &thresh, Eigen::Vector4d &out);
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
        Eigen::Matrix3d r_il = Eigen::Matrix3d::Identity();
        Eigen::Vector3d p_il = Eigen::Vector3d::Zero();
        Eigen::Matrix3d r_cl = Eigen::Matrix3d::Identity();
        Eigen::Vector3d p_cl = Eigen::Vector3d::Zero();

        bool esti_li = false;
        bool esti_ci = false;
    };

    struct IMUData
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d acc;
        Eigen::Vector3d gyro;
        double timestamp;

        IMUData() = default;
        IMUData(const Eigen::Vector3d &a, const Eigen::Vector3d &g, double &d) : acc(a), gyro(g), timestamp(d) {}
    };

    struct Pose
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d acc;
        Eigen::Vector3d gyro;
        Eigen::Matrix3d rot;
        Eigen::Vector3d pos;
        Eigen::Vector3d vel;
        double offset;
        Pose();
        Pose(double t, Eigen::Vector3d a, Eigen::Vector3d g, Eigen::Vector3d v, Eigen::Vector3d p, Eigen::Matrix3d r) : offset(t), acc(a), gyro(g), vel(v), pos(p), rot(r) {}
    };

    struct SyncPackage
    {
        std::vector<IMUData> imus;
        CloudType::Ptr cloud;
        cv::Mat image;
        double image_time = 0.0;
        double cloud_start_time = 0.0;
        double cloud_end_time = 0.0;
        bool lidar_end = false;
    };
} // namespace livo
