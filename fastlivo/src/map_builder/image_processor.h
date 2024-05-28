#pragma once
#include <list>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include <Eigen/Eigen>
#include "ieskf.h"
#include "commons.h"
#include "pinhole_camera.h"
#include <opencv2/opencv.hpp>
#include <pcl/filters/voxel_grid.h>

#define HASH_P 116101
#define MAX_N 10000000000

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

class Point;

class Feature
{
public:
    Feature(const V2D &_px, const V3D &_pc, const M3D &_r_cw, const V3D _t_cw, double _score, int _level)
        : px(_px), pc(_pc), r_cw(_r_cw), t_cw(_t_cw), score(_score), level(_level)
    {
        patches.resize(3);
    }

    M3D r_wc() { return r_cw.transpose(); }

    V3D t_wc() { return -r_cw.transpose() * t_cw; }

public:
    size_t frame_id;
    cv::Mat frame;
    V2D px;
    V3D pc;
    M3D r_cw;
    V3D t_cw;
    Vec<cv::Mat> patches;
    float score;
    int level;
    std::weak_ptr<Point> point;
};

class Point
{
public:
    Point(const V3D &_pos);

    void addObs(std::shared_ptr<Feature> ftr);

    bool getCloseViewObs(const V3D &cam_pos, std::shared_ptr<Feature> &out, double thresh = 0.5);

    bool getFurthestViewObs(const V3D &cam_pos, std::shared_ptr<Feature> &out);

    void deleteFeatureRef(std::shared_ptr<Feature> feat);

public:
    V3D pos;
    std::list<std::shared_ptr<Feature>> obs;
    double value;
};

using FeatMap = std::unordered_map<VoxelKey, Vec<std::shared_ptr<Point>>, VoxelKey::Hasher>;

struct ReferencePoint
{
    cv::Mat patch;
    int search_level;
    double error;
    std::shared_ptr<Feature> feat_ptr;
};

class ImageProcessor
{
public:
    ImageProcessor(Config &config, std::shared_ptr<IESKF> kf);

    int gridIndex(const V2D &px);

    void selectReference(CloudType::Ptr cloud);

    M2D getCRAffine2d(std::shared_ptr<Feature> ref_ptr);

    int getBestSearchLevel(const M2D &affine_cr, const int max_level);

    void getRefAffinePatch(const M2D &affine_rc, const V2D &px_ref, const cv::Mat &img_ref, const int search_level, cv::Mat &patch);

    void process(cv::Mat &img, CloudType::Ptr cloud, bool is_new_cloud);
    
    void addObservations();

    void addPoint(std::shared_ptr<Point> point_ptr);

    int incrVisualMap();
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getLastestColoredCloud();
    
    void computeLevelJacc(State &state,SharedState &share_data,int level);

    

    M3D r_cw();
    V3D t_cw();
    M3D r_ci();
    V3D t_ci();
    M3D r_wc();
    V3D t_wc();

private:
    Config m_config;
    int m_grid_num;
    int m_grid_width;
    int m_grid_height;
    int m_patch_size;
    std::shared_ptr<IESKF> m_kf;
    std::shared_ptr<PinholeCamera> m_camera;

    Vec<Point *> cache_points;
    Vec<bool> cache_flag;
    Vec<double> cache_grid_depth;
    Vec<double> cache_pixel_depth;
    Vec<double> cache_score;
    Vec<V3D> cache_points_to_add;
    Vec<ReferencePoint> cache_reference;
    cv::Mat m_cur_img_color;
    cv::Mat m_cur_img_gray;
    CloudType::Ptr m_cur_cloud;
    pcl::VoxelGrid<PointType> m_cloud_filter;
    FeatMap m_featmap;
    u_int64_t m_frame_count;
};