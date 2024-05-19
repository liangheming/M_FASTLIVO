#include "selector.h"

namespace livo
{
    VoxelKey VoxelKey::index(double x, double y, double z, double resolution, double bias)
    {
        Eigen::Vector3d point(x, y, z);
        Eigen::Vector3d idx = (point / resolution + Eigen::Vector3d(bias, bias, bias)).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }

    LidarSelector::LidarSelector(std::shared_ptr<kf::IESKF> kf,
                                 std::shared_ptr<PinholeCamera> camera,
                                 int patch_size,
                                 int grid_size,
                                 double scan_res,
                                 double voxel_size)
        : m_kf(kf), m_camera(camera), m_patch_size(patch_size), m_grid_size(grid_size), m_scan_resolution(scan_res), m_voxel_size(voxel_size)
    {
        m_patch_size_half = patch_size / 2;
        m_patch_n_pixels = m_patch_size * m_patch_size;
        m_grid_n_width = static_cast<int>(camera->width() / m_grid_size);
        m_grid_n_height = static_cast<int>(camera->height() / m_grid_size);
        m_grid_flat_length = m_grid_n_height * m_grid_n_height;
        m_feat_map.clear();
        m_scan_filter.setLeafSize(m_scan_resolution, m_scan_resolution, m_scan_resolution);
        m_r_fw.setIdentity();
        m_p_fw.setZero();
    }

    void LidarSelector::updateFrameState()
    {
        kf::State &x = m_kf->x();
        // Eigen::Matrix3d r_l_w = x.r_il.transpose() * x.rot.transpose();
        // Eigen::Vector3d p_l_w = -x.r_il.transpose() * (x.rot.transpose() * x.pos + x.p_il);

        m_r_fw = x.r_cl * x.r_il.transpose() * x.rot.transpose();

        m_p_fw = -x.r_cl * x.r_il.transpose() * (x.rot.transpose() * x.pos + x.p_il) + x.p_cl;

        m_r_wf = m_r_fw.transpose();

        m_p_wf = -m_r_wf * m_p_fw;
    }

    Eigen::Vector3d LidarSelector::w2f(const Eigen::Vector3d &pw)
    {
        return m_r_fw * pw + m_p_fw;
    }

    Eigen::Vector2d LidarSelector::f2c(const Eigen::Vector3d &pf)
    {
        return Eigen::Vector2d(m_camera->fx() * pf.x() / pf.z() + m_camera->cx(), m_camera->fy() * pf.y() / pf.z() + m_camera->cy());
    }

    bool LidarSelector::getReferencePoints(CloudType::Ptr cloud, std::vector<ReferencePoint> &reference_points)
    {
        if (m_feat_map.size() <= 0)
            return false;
        updateFrameState();

        CloudType::Ptr cloud_ds(new CloudType);
        m_scan_filter.setInputCloud(cloud);
        m_scan_filter.filter(*cloud_ds);

        std::vector<double> depths;
        depths.resize(m_camera->width() * m_camera->height(), 0.0);
        std::unordered_set<VoxelKey, VoxelKey::Hasher> selected_voxels;
        for (auto p : cloud_ds->points)
        {
            Eigen::Vector3d pw(p.x, p.y, p.z);

            VoxelKey k = VoxelKey::index(pw.x(), pw.y(), pw.z(), m_voxel_size, 0.0);

            if (selected_voxels.find(k) == selected_voxels.end())
                selected_voxels.insert(k);

            Eigen::Vector3d pc = w2f(pw);
            if (pc(2) <= 0)
                continue;
            Eigen::Vector2i px = f2c(pc).cast<int>();
            if (!m_camera->isInFrame(px, (m_patch_size_half + 1) * 8))
                continue;

            depths[m_camera->width() * px(1) + px(0)] = pc(2);
        }

        for (auto k : selected_voxels)
        {
            auto it = m_feat_map.find(k);
            if (it == m_feat_map.end())
                continue;
            std::vector<std::shared_ptr<Point>> &voxel_points = it->second;
            for (auto p : voxel_points)
            {
                Eigen::Vector3d pc = w2f(p->pos);
                if (pc(2) <= 0)
                    continue;
                Eigen::Vector2i px = f2c(pc);
                if (!m_camera->isInFrame(px, (m_patch_size_half + 1) * 8))
                    continue;
            }
        }
    }

} // namespace livo
