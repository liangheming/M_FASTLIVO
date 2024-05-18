#include "imu_processor.h"

namespace livo
{
    IMUProcessor::IMUProcessor(Config &config, std::shared_ptr<kf::IESKF> kf) : m_config(config), m_kf(kf)
    {
        m_Q.setIdentity();
        m_Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * m_config.ng;
        m_Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * m_config.na;
        m_Q.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * m_config.nbg;
        m_Q.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * m_config.nba;
        m_last_acc.setZero();
        m_last_gyro.setZero();
        m_imu_cache.clear();
        m_imu_poses_cache.clear();
        push_head_pose = true;
    }

    bool IMUProcessor::initialize(SyncPackage &package)
    {
        m_imu_cache.insert(m_imu_cache.end(), package.imus.begin(), package.imus.end());
        // std::cout << "INIT: " << m_imu_cache.size() << " | " << m_config.imu_init_num << " | " << package.lidar_end << std::endl;
        if (m_imu_cache.size() < m_config.imu_init_num || !package.lidar_end)
            return false;
        Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
        for (const auto &imu : m_imu_cache)
        {
            acc_mean += imu.acc;
            gyro_mean += imu.gyro;
        }
        acc_mean /= static_cast<double>(m_imu_cache.size());
        gyro_mean /= static_cast<double>(m_imu_cache.size());
        m_kf->x().r_il = m_config.r_il;
        m_kf->x().p_il = m_config.p_il;
        m_kf->x().r_cl = m_config.r_cl;
        m_kf->x().p_cl = m_config.p_cl;
        m_kf->x().bg = gyro_mean;

        if (m_config.gravity_align)
        {
            m_kf->x().rot = (Eigen::Quaterniond::FromTwoVectors((-acc_mean).normalized(), Eigen::Vector3d(0.0, 0.0, -1.0)).matrix());
            m_kf->x().initG(Eigen::Vector3d(0, 0, -1.0));
        }
        else
            m_kf->x().initG(-acc_mean);
        m_kf->P().setIdentity();
        m_kf->P().block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 0.00001;
        m_kf->P().block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 0.00001;
        m_kf->P().block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 0.00001;
        m_kf->P().block<3, 3>(15, 15) = Eigen::Matrix3d::Identity() * 0.00001;
        m_kf->P().block<3, 3>(21, 21) = Eigen::Matrix3d::Identity() * 0.0001;
        m_kf->P().block<3, 3>(24, 24) = Eigen::Matrix3d::Identity() * 0.0001;
        m_kf->P().block<2, 2>(27, 27) = Eigen::Matrix2d::Identity() * 0.00001;

        m_last_imu = m_imu_cache.back();
        m_last_propagate_end_time = package.cloud_end_time;
        return true;
    }

    void IMUProcessor::undistort(SyncPackage &package)
    {

        m_imu_cache.clear();
        m_imu_cache.push_back(m_last_imu);
        m_imu_cache.insert(m_imu_cache.end(), package.imus.begin(), package.imus.end());

        const double imu_time_begin = m_imu_cache.front().timestamp;
        const double imu_time_end = m_imu_cache.back().timestamp;

        const double cloud_time_begin = package.cloud_start_time;
        const double propagate_time_end = package.lidar_end ? package.cloud_end_time : package.image_time;

        if (push_head_pose)
        {
            m_imu_poses_cache.clear();
            m_imu_poses_cache.emplace_back(0.0, m_last_acc, m_last_gyro, m_kf->x().vel, m_kf->x().pos, m_kf->x().rot);
            push_head_pose = false;
        }

        Eigen::Vector3d acc_val, gyro_val;
        double dt = 0.0;
        kf::Input inp;
        inp.acc = m_imu_cache.back().acc;
        inp.gyro = m_imu_cache.back().gyro;
        for (auto it_imu = m_imu_cache.begin(); it_imu < (m_imu_cache.end() - 1); it_imu++)
        {
            IMUData &head = *it_imu;
            IMUData &tail = *(it_imu + 1);
            if (tail.timestamp < m_last_propagate_end_time)
                continue;
            gyro_val = 0.5 * (head.gyro + tail.gyro);
            acc_val = 0.5 * (head.acc + tail.acc);

            if (head.timestamp < m_last_propagate_end_time)
                dt = tail.timestamp - m_last_propagate_end_time;
            else
                dt = tail.timestamp - head.timestamp;

            inp.acc = acc_val;
            inp.gyro = gyro_val;
            m_kf->predict(inp, dt, m_Q);

            m_last_gyro = gyro_val - m_kf->x().bg;
            m_last_acc = m_kf->x().rot * (acc_val - m_kf->x().ba) + m_kf->x().g;
            double offset = tail.timestamp - cloud_time_begin;
            m_imu_poses_cache.emplace_back(offset, m_last_acc, m_last_gyro, m_kf->x().vel, m_kf->x().pos, m_kf->x().rot);
        }

        dt = propagate_time_end - imu_time_end;
        m_kf->predict(inp, dt, m_Q);
        m_last_imu = m_imu_cache.back();
        m_last_propagate_end_time = propagate_time_end;

        if (package.lidar_end)
        {
            Eigen::Matrix3d cur_rot = m_kf->x().rot;
            Eigen::Vector3d cur_pos = m_kf->x().pos;
            Eigen::Matrix3d cur_r_il = m_kf->x().r_il;
            Eigen::Vector3d cur_p_il = m_kf->x().p_il;
            auto it_pcl = package.cloud->points.end() - 1;

            for (auto it_kp = m_imu_poses_cache.end() - 1; it_kp != m_imu_poses_cache.begin(); it_kp--)
            {
                auto head = it_kp - 1;
                auto tail = it_kp;

                Eigen::Matrix3d imu_rot = head->rot;
                Eigen::Vector3d imu_pos = head->pos;
                Eigen::Vector3d imu_vel = head->vel;
                Eigen::Vector3d imu_acc = tail->acc;
                Eigen::Vector3d imu_gyro = tail->gyro;

                for (; it_pcl->curvature / double(1000) > head->offset; it_pcl--)
                {
                    dt = it_pcl->curvature / double(1000) - head->offset;
                    Eigen::Vector3d point(it_pcl->x, it_pcl->y, it_pcl->z);
                    Eigen::Matrix3d point_rot = imu_rot * Sophus::SO3d::exp(imu_gyro * dt).matrix();
                    Eigen::Vector3d point_pos = imu_pos + imu_vel * dt + 0.5 * imu_acc * dt * dt;
                    Eigen::Vector3d p_compensate = cur_r_il.transpose() * (cur_rot.transpose() * (point_rot * (cur_r_il * point + cur_p_il) + point_pos - cur_pos) - cur_p_il);
                    it_pcl->x = p_compensate(0);
                    it_pcl->y = p_compensate(1);
                    it_pcl->z = p_compensate(2);
                    if (it_pcl == package.cloud->points.begin())
                        break;
                }
            }
        }
        push_head_pose = true;
    }
} // namespace livo
