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
            : px(_px), fp(_fp), r_fw(_r_fw), p_fw(_p_fw), score(_score), level(_level)
        {
            patches.resize(3);
        }

        Eigen::Matrix3d r_wf() { return r_fw.transpose(); }

        Eigen::Vector3d p_wf() { return -r_fw.transpose() * p_fw; }

    public:
        size_t frame_id;
        cv::Mat frame;
        Eigen::Vector2d px;
        Eigen::Vector3d fp;
        Eigen::Matrix3d r_fw;
        Eigen::Vector3d p_fw;
        std::vector<cv::Mat> patches;
        float score;
        int level;
    };

    class Point
    {
    public:
        Point(const Eigen::Vector3d &_pos);

        void addObs(std::shared_ptr<Feature> ftr);

        bool getCloseViewObs(const Eigen::Vector3d &cam_pos, std::shared_ptr<Feature> &out, double thresh = 0.5);

        bool getFurthestViewObs(const Eigen::Vector3d &cam_pos, std::shared_ptr<Feature> &out);

        void deleteFeatureRef(std::shared_ptr<Feature> feat);

    public:
        Eigen::Vector3d pos;
        std::list<std::shared_ptr<Feature>> obs;
        size_t n_obs;
        double value;
    };

    struct ReferencePoint
    {
        Point *point;
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

        bool getReferencePoints(cv::Mat gray_img, CloudType::Ptr cloud);

        Eigen::Vector3d w2f(const Eigen::Vector3d &pw);

        Eigen::Vector2d f2c(const Eigen::Vector3d &pf);

        void updateFrameState();

        void resetCache();

        int gridIndex(const Eigen::Vector2d &px);

        int getBestSearchLevel(const Eigen::Matrix2d &A_cur_ref, const int max_level);

        void wrapAffine(const Eigen::Matrix2d &affine, const Eigen::Vector2d &px_ref, const cv::Mat &img_ref, const int level_cur, cv::Mat &patch);

        void getPatch(cv::Mat img, const Eigen::Vector2d px, cv::Mat &patch, int level);

        Eigen::Matrix2d getWarpMatrixAffine(const Eigen::Vector2d &px_ref, const Eigen::Vector3d &fp_ref, const double depth_ref, const Eigen::Matrix3d &r_cr, const Eigen::Vector3d &t_cr);

        void addPoint(std::shared_ptr<Point> point);

        void addObservations(cv::Mat img);

        int incrVisualMap(cv::Mat img, CloudType::Ptr cloud);

        void process(cv::Mat img, CloudType::Ptr cloud, bool is_new_cloud);

        Eigen::Vector3d getPixelBRG(cv::Mat img_bgr, const Eigen::Vector2d &px);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr getCurentCloudRGB();

        void updateState();

        void updateOneLevel();

        void dpi(Eigen::Vector3d p, Eigen::Matrix<double, 2, 3> &J);

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
        CloudType::Ptr m_cloud;
        cv::Mat m_img_bgr;
        cv::Mat m_img_gray;
        std::vector<double> cache_depth;
        std::vector<bool> cache_grid_flag;
        std::vector<Point *> cache_grid_points;
        std::vector<double> cache_grid_dist;
        std::vector<double> cache_grid_cur;
        std::vector<Eigen::Vector3d> cache_grid_add_points;
        std::vector<ReferencePoint> cache_reference_points;
    };

} // namespace livo
