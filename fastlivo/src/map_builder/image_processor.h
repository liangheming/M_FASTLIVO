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

class Feature
{
public:
    Feature(const V2D &_px, const V3D &_pf, const M3D &_r_fw, const V3D _t_fw, double _score, int _level)
        : px(_px), pf(_pf), r_fw(_r_fw), t_fw(_t_fw), score(_score), level(_level)
    {
        patches.resize(3);
    }

    M3D r_wf() { return r_fw.transpose(); }

    V3D t_wf() { return -r_fw.transpose() * t_fw; }

public:
    size_t frame_id;
    cv::Mat frame;
    V2D px;
    V3D pf;
    M3D r_fw;
    V3D t_fw;
    Vec<cv::Mat> patches;
    float score;
    int level;
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

class ImageProcessor
{
public:
    ImageProcessor(std::shared_ptr<IESKF> kf, std::shared_ptr<PinholeCamera> camera);

    void process(cv::Mat &img, CloudType::Ptr cloud, bool is_new_cloud);

private:
    std::shared_ptr<IESKF> m_kf;
    std::shared_ptr<PinholeCamera> m_camera;
};