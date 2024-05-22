#include "map_builder.h"

MapBuilder::MapBuilder(Config &config, std::shared_ptr<IESKF> kf) : m_config(config), m_kf(kf)
{
    m_imu_processor = std::make_shared<IMUProcessor>(config, kf);
    m_lidar_processor = std::make_shared<LidarProcessor>(config, kf);
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
            m_status = BuilderStatus::MAPPING;
        }
        std::cout << "[LIO]: CLOUD MAP INITIALIZED!" << std::endl;
        return;
    }

    if (package.lidar_end)
    {
        m_lidar_processor->process(package);
        // std::cout << m_kf->x() << std::endl;
        std::cout << "[LIO]: PROCESS!" << std::endl;
    }
    else
    {
        std::cout << "[VIO]: PROCESS!" << std::endl;
    }
}