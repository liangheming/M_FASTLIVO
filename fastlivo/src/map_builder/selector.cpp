#include "selector.h"

namespace livo
{
    VoxelKey VoxelKey::index(double x, double y, double z, double resolution, double bias)
    {
        Eigen::Vector3d point(x, y, z);
        Eigen::Vector3d idx = (point / resolution + Eigen::Vector3d(bias, bias, bias)).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }

    bool Point::getCloseViewObs(const Eigen::Vector3d &cam_pos, std::shared_ptr<Feature> &out, double thresh)
    {
        if (obs.size() <= 0)
            return false;
        Eigen::Vector3d obs_dir(cam_pos - pos);
        obs_dir.normalize();
        auto min_it = obs.begin();
        double min_cos_angle = 0;
        for (auto it = obs.begin(); it != obs.end(); ++it)
        {
            Eigen::Vector3d dir((*it)->p_wf() - pos);
            dir.normalize();
            double cos_angle = obs_dir.dot(dir);
            if (cos_angle > min_cos_angle)
            {
                min_cos_angle = cos_angle;
                min_it = it;
            }
        }
        out = *min_it;
        if (min_cos_angle < thresh)
            return false;
        return true;
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

        cache_depth.reserve(camera->width() * camera->height());
        cache_grid_flag.reserve(m_grid_flat_length);
        cache_grid_points.reserve(m_grid_flat_length);
        cache_grid_dist.reserve(m_grid_flat_length);
        cache_grid_cur.reserve(m_grid_flat_length);
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

    void LidarSelector::resetCache()
    {
        cache_depth.resize(m_camera->width() * m_camera->height(), 0.0);
        cache_grid_flag.resize(m_grid_flat_length, false);
        cache_grid_points.resize(m_grid_flat_length, nullptr);
        cache_grid_dist.resize(m_grid_flat_length, 10000.0);
        cache_grid_cur.resize(m_grid_flat_length, 0.0);
    }

    int LidarSelector::gridIndex(const Eigen::Vector2d &px)
    {
        return static_cast<int>(px[1] / m_grid_size) * m_grid_n_width + static_cast<int>(px[0] / m_grid_size);
    }

    Eigen::Matrix2d LidarSelector::getWarpMatrixAffine(const Eigen::Vector2d &px_ref, const Eigen::Vector3d &fp_ref, const double depth_ref, const Eigen::Matrix3d &r_cr, const Eigen::Vector3d &t_cr)
    {
        const Eigen::Vector3d xyz_ref(fp_ref * depth_ref);
        Eigen::Vector3d xyz_du_ref(m_camera->cam2world(px_ref + Eigen::Vector2d(m_patch_size_half, 0)));
        Eigen::Vector3d xyz_dv_ref(m_camera->cam2world(px_ref + Eigen::Vector2d(0, m_patch_size_half)));
        xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
        xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];

        Eigen::Vector3d pc_cur = r_cr * xyz_ref + t_cr;
        Eigen::Vector3d pc_du_cur = r_cr * xyz_du_ref + t_cr;
        Eigen::Vector3d pc_dv_cur = r_cr * xyz_dv_ref + t_cr;
        const Eigen::Vector2d px_cur(m_camera->world2cam(pc_cur));
        const Eigen::Vector2d px_du(m_camera->world2cam(pc_du_cur));
        const Eigen::Vector2d px_dv(m_camera->world2cam(pc_dv_cur));
        Eigen::Matrix2d affine_mat;
        affine_mat.col(0) = (px_du - px_cur) / m_patch_size_half;
        affine_mat.col(1) = (px_dv - px_cur) / m_patch_size_half;
        return affine_mat;
    }

    int LidarSelector::getBestSearchLevel(const Eigen::Matrix2d &A_cur_ref, const int max_level)
    {
        int search_level = 0;
        double D = A_cur_ref.determinant();

        while (D > 3.0 && search_level < max_level)
        {
            search_level += 1;
            D *= 0.25;
        }
        return search_level;
    }

    void LidarSelector::wrapAffine(const Eigen::Matrix2d &affine, const Eigen::Vector2d &px_ref, const cv::Mat &img_ref, const int level_cur, cv::Mat &patch)
    {
        for (int y = 0; y < m_patch_size; ++y)
        {
            for (int x = 0; x < m_patch_size; ++x)
            {
                Eigen::Vector2d px_patch(x - m_patch_size_half, y - m_patch_size_half);
                px_patch *= (1 << level_cur);
                const Eigen::Vector2d px(affine * px_patch + px_ref);

                if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
                    patch.ptr<float>(y)[x] = 0.0;
                else
                    patch.ptr<float>(y)[x] = PinholeCamera::interpolateMat_8u(img_ref, px(0), px(1));
            }
        }
    }

    void LidarSelector::getPatch(cv::Mat img, const Eigen::Vector2d px, cv::Mat &patch, int level)
    {
        const float u_ref = px[0];
        const float v_ref = px[1];
        const int scale = (1 << level);
        const int u_ref_i = floorf(px[0] / scale) * scale;
        const int v_ref_i = floorf(px[1] / scale) * scale;
        const float subpix_u_ref = (u_ref - u_ref_i) / scale;
        const float subpix_v_ref = (v_ref - v_ref_i) / scale;
        const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
        const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
        const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
        const float w_ref_br = subpix_u_ref * subpix_v_ref;

        for (int y = 0; y < m_patch_size; y++)
        {
            for (int x = 0; x < m_patch_size; ++x)
            {
                Eigen::Vector2d px_patch(x - m_patch_size_half, y - m_patch_size_half);
                px_patch *= (1 << level);
                int tl_x = u_ref_i + px_patch(0), tl_y = v_ref_i + px_patch(1);
                uint8_t tl = img.ptr<uint8_t>(tl_y)[tl_x];
                uint8_t tr = img.ptr<uint8_t>(tl_y)[tl_x + 1];
                uint8_t bl = img.ptr<uint8_t>(tl_y + 1)[tl_x];
                uint8_t br = img.ptr<uint8_t>(tl_y + 1)[tl_x + 1];
                patch.ptr<float>(y)[x] = w_ref_tl * tl + w_ref_tr * tr + w_ref_bl * bl + w_ref_br * br;
            }
        }
    }

    bool LidarSelector::getReferencePoints(cv::Mat gray_img, CloudType::Ptr cloud, std::vector<ReferencePoint> &reference_points)
    {
        if (m_feat_map.size() <= 0)
            return false;
        updateFrameState();

        resetCache();

        CloudType::Ptr cloud_ds(new CloudType);
        m_scan_filter.setInputCloud(cloud);
        m_scan_filter.filter(*cloud_ds);

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

            cache_depth[m_camera->width() * px(1) + px(0)] = pc(2);
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
                Eigen::Vector2d px = f2c(pc);
                if (!m_camera->isInFrame(px.cast<int>(), (m_patch_size_half + 1) * 8))
                    continue;

                int index = gridIndex(px);
                cache_grid_flag[index] = true;
                Eigen::Vector3d obs_vec(m_p_wf - p->pos);

                double cur_dist = obs_vec.norm();
                float cur_value = p->value;

                if (cur_dist <= cache_grid_dist[index])
                {
                    cache_grid_dist[index] = cur_dist;
                    cache_grid_points[index] = p;
                }

                if (cur_value >= cache_grid_cur[index])
                    cache_grid_cur[index] = cur_value;
            }
        }

        std::vector<ReferencePoint>().swap(cache_reference_points);

        for (int i = 0; i < m_grid_flat_length; i++)
        {
            if (!cache_grid_flag[i])
                continue;
            std::shared_ptr<Point> p = cache_grid_points[i];
            Eigen::Vector3d pc = w2f(p->pos);
            Eigen::Vector2d px = f2c(pc);
            bool depth_continous = false;
            for (int u = -m_patch_size_half; u <= m_patch_size_half; u++)
            {
                for (int v = -m_patch_size_half; v <= m_patch_size_half; v++)
                {
                    if (u == 0 && v == 0)
                        continue;
                    double depth = cache_depth[m_camera->width() * (v + int(px(1))) + u + int(px(0))];
                    if (depth == 0.0)
                        continue;
                    double delta_dist = abs(pc(2) - depth);
                    if (delta_dist > 1.5)
                    {
                        depth_continous = true;
                        break;
                    }
                }
                if (depth_continous)
                    break;
            }
            if (depth_continous)
                continue;
            std::shared_ptr<Feature> feat;

            if (!p->getCloseViewObs(m_p_wf, feat, 0.5))
                continue;
            Eigen::Matrix3d r_cr = m_r_fw * feat->r_wf();
            Eigen::Vector3d p_cr = m_r_fw * feat->p_wf() + m_p_fw;
            Eigen::Matrix2d affine_cr = getWarpMatrixAffine(feat->px, feat->fp, (feat->p_wf() - m_p_wf).norm(), r_cr, p_cr);
            int search_level = getBestSearchLevel(affine_cr, 2);
            Eigen::Matrix2d affine_rc = affine_cr.inverse();

            if (affine_rc.hasNaN())
            {
                std::cout << "AFFINE HAS NAN" << std::endl;
                continue;
            }
            cv::Mat ref_patch(m_patch_size, m_patch_size, CV_32FC1);
            wrapAffine(affine_rc, feat->px, feat->frame, 0, ref_patch);
            cv::Mat cur_patch(m_patch_size, m_patch_size, CV_32FC1);
            getPatch(gray_img, px, cur_patch, search_level);

            double sq_dist = cv::sum((ref_patch - cur_patch) * (ref_patch - cur_patch))[0];

            if (sq_dist > 100 * m_patch_n_pixels)
                continue;

            ReferencePoint ref_point;
            ref_point.error = sq_dist;
            ref_point.level = search_level;
            ref_point.patch = ref_patch;
            ref_point.point = p;
            cache_reference_points.push_back(ref_point);
        }
    }

} // namespace livo
