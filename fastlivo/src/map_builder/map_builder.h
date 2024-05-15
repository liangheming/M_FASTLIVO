#pragma once
#include <memory>
#include "ieskf.h"
#include "ikd_Tree.h"
#include "imu_processor.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

namespace livo
{
    enum Status
    {
        IMU_INIT,
        MAP_INIT,
        MAPPING
    };

    struct LocalMap
    {
        bool is_initialed = false;
        BoxPointType local_map_corner;
        std::vector<BoxPointType> cub_to_rm;
    };

    class MapBuilder
    {
    public:
        MapBuilder(Config &config);

        void updateLidarLossFunc(kf::State &state, kf::SharedState &share_data);

        void process(SyncPackage &package);

        void trimMap();

        void incrMap();

        CloudType::Ptr lidar2World(CloudType::Ptr inp);

        CloudType::Ptr lidar2Body(CloudType::Ptr inp);

    private:
        Config m_config;
        Status m_status;
        LocalMap m_local_map;
        std::shared_ptr<kf::IESKF> m_kf;
        std::shared_ptr<KD_TREE<PointType>> m_ikdtree;
        std::shared_ptr<IMUProcessor> m_imu_processor;

        CloudType::Ptr m_cloud_lidar;
        CloudType::Ptr m_cloud_down_lidar;
        CloudType::Ptr m_cloud_down_world;
        std::vector<bool> m_point_selected_flag;
        CloudType::Ptr m_norm_vec;
        CloudType::Ptr m_effect_cloud_lidar;
        CloudType::Ptr m_effect_norm_vec;
        std::vector<PointVec> m_nearest_points;

        pcl::VoxelGrid<PointType> m_scan_filter;
    };
}
