#pragma once
#include "ieskf.h"
#include "commons.h"

namespace livo
{
    class IMUProcessor
    {
    public:
        IMUProcessor(Config &config, std::shared_ptr<kf::IESKF> kf);

        bool initialize(std::vector<IMUData> &imus);

        void undistort(SyncPackage &sync);

        void process(SyncPackage &sync);

        bool isInitialized() { return is_initialized; }

    private:
        Config m_config;
        std::shared_ptr<kf::IESKF> m_kf;
        std::vector<IMUData> m_imu_cache;
        IMUData m_last_imu;
        std::vector<Pose> m_imu_poses_cache;

        bool is_initialized;
        double m_last_end_time;
        Eigen::Vector3d m_last_acc;
        Eigen::Vector3d m_last_gyro;
        kf::Matrix12d m_Q;
    };
} // namespace livo
