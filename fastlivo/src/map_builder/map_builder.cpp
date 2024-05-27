#include "map_builder.h"

MapBuilder::MapBuilder(Config &config, std::shared_ptr<IESKF> kf) : m_config(config), m_kf(kf)
{
    m_imu_processor = std::make_shared<IMUProcessor>(config, kf);
    m_lidar_processor = std::make_shared<LidarProcessor>(config, kf);
    m_image_processor = std::make_shared<ImageProcessor>(config, kf);
    m_lastest_cloud = nullptr;
    m_is_new_cloud = false;
    m_status = BuilderStatus::IMU_INIT;
}

void MapBuilder::process(SyncPackage &package)
{
    if (m_status == BuilderStatus::IMU_INIT)
    {
        if (m_imu_processor->initialize(package))
            m_status = BuilderStatus::MAP_INIT;
        std::cout << "[LIO]: IMU INITIALIZED!" << std::endl;
        return;
    }

    m_imu_processor->undistort(package);

    if (m_status == BuilderStatus::MAP_INIT)
    {
        if (package.lidar_end)
        {

            CloudType::Ptr cloud_world = LidarProcessor::transformCloud(package.cloud,
                                                                        m_kf->x().r_wi * m_kf->x().r_il,
                                                                        m_kf->x().r_wi * m_kf->x().t_il + m_kf->x().t_wi);
            m_lidar_processor->initCloudMap(cloud_world->points);
            m_lastest_cloud = cloud_world;
            m_is_new_cloud = true;
            m_status = BuilderStatus::MAPPING;
        }
        std::cout << "[LIO]: CLOUD MAP INITIALIZED!" << std::endl;
        return;
    }

    if (package.lidar_end)
    {
        m_lidar_processor->process(package);
        M3D r_wl = m_kf->x().r_wi * m_kf->x().r_il;
        V3D t_wl = m_kf->x().r_wi * m_kf->x().t_il + m_kf->x().t_wi;
        m_lastest_cloud = LidarProcessor::transformCloud(package.cloud, r_wl, t_wl);
        m_is_new_cloud = true;
    }
    else
    {
        if (m_lastest_cloud != nullptr)
        {
            // std::cout << "[VIO]: PROCESS START!" << std::endl;
            m_image_processor->process(package.image, m_lastest_cloud, m_is_new_cloud);
            m_is_new_cloud = false;
        }
    }
}