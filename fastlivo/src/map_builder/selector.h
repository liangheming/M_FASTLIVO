#pragma once
#include <list>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include "ieskf.h"
#include "camera.h"
#include "commons.h"
#include <pcl/filters/voxel_grid.h>

#define HASH_P 116101
#define MAX_N 10000000000
namespace livo
{
    class VoxelKey
    {
    public:
        int64_t x, y, z;

        VoxelKey(int64_t _x = 0, int64_t _y = 0, int64_t _z = 0) : x(_x), y(_y), z(_z) {}

        bool operator==(const VoxelKey &other) const
        {
            return (x == other.x && y == other.y && z == other.z);
        }

        static VoxelKey index(double x, double y, double z, double resolution, double bias = 0.0);

        struct Hasher
        {
            int64_t operator()(const VoxelKey &k) const
            {
                return ((((k.z) * HASH_P) % MAX_N + (k.y)) * HASH_P) % MAX_N + (k.x);
            }
        };
    };

    class Feature
    {
    public:
        Feature(const Eigen::Vector2d &_px, const Eigen::Vector3d &_fp, const Eigen::Matrix3d &_r_fw, const Eigen::Vector3d _p_fw, double _score, int _level)
            : px(_px), fp(_fp), r_fw(_r_fw), p_fw(_p_fw), score(_score), level(_level) {}

    public:
        cv::Mat frame;
        Eigen::Vector2d px;
        Eigen::Vector3d fp;
        Eigen::Matrix3d r_fw;
        Eigen::Vector3d p_fw;
        float score;
        int level;
    };

    class Point
    {
    public:
        Point(const Eigen::Vector3d &pos);

        void addObs(std::shared_ptr<Feature> ftr);

    public:
        Eigen::Vector3d pos;
        std::list<Feature> obs;
        size_t n_obs;
    };

    struct ReferencePoint
    {
        std::shared_ptr<Point> point;
        double error;
        cv::Mat patch;
        int level;
    };

    class LidarSelector
    {
    public:
        LidarSelector(std::shared_ptr<kf::IESKF> kf,
                      std::shared_ptr<PinholeCamera> camera,
                      int patch_size,
                      int grid_size,
                      double scan_res,
                      double voxel_size);

        bool getReferencePoints(CloudType::Ptr cloud, std::vector<ReferencePoint> &reference_points);

        Eigen::Vector3d w2f(const Eigen::Vector3d &pw);

        Eigen::Vector2d f2c(const Eigen::Vector3d &pf);

        void updateFrameState();

    private:
        int m_grid_size;
        int m_patch_size;
        int m_patch_size_half;
        int m_patch_n_pixels;
        int m_grid_n_width;
        int m_grid_n_height;
        int m_grid_flat_length;
        double m_scan_resolution = 0.2;
        double m_voxel_size = 0.5;
        std::shared_ptr<kf::IESKF> m_kf;
        std::shared_ptr<PinholeCamera> m_camera;
        std::unordered_map<VoxelKey, std::vector<std::shared_ptr<Point>>, VoxelKey::Hasher> m_feat_map;
        pcl::VoxelGrid<PointType> m_scan_filter;
        Eigen::Matrix3d m_r_fw;
        Eigen::Vector3d m_p_fw;
        Eigen::Matrix3d m_r_wf;
        Eigen::Vector3d m_p_wf;

        std::vector<double> temp_depth;
        std::vector<bool> temp_grid_flag;
        std::vector<std::shared_ptr<Point>> temp_grid_points;
    };

} // namespace livo
