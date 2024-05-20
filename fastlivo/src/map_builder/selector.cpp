#include "selector.h"

namespace livo
{
    VoxelKey VoxelKey::index(double x, double y, double z, double resolution, double bias)
    {
        Eigen::Vector3d point(x, y, z);
        Eigen::Vector3d idx = (point / resolution + Eigen::Vector3d(bias, bias, bias)).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }
    Point::Point(const Eigen::Vector3d &_pos) : pos(_pos) {}
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

    void Point::addObs(std::shared_ptr<Feature> ftr)
    {
        obs.push_front(ftr);
        n_obs++;
    }

    bool Point::getFurthestViewObs(const Eigen::Vector3d &cam_pos, std::shared_ptr<Feature> &out)
    {
        if (obs.size() <= 0)
            return false;

        auto max_it = obs.begin();
        double maxdist = 0.0;
        for (auto it = obs.begin(); it != obs.end(); it++)
        {
            double dist = ((*it)->p_wf() - cam_pos).norm();
            if (dist > maxdist)
            {
                maxdist = dist;
                max_it = it;
            }
        }
        out = *max_it;
        return true;
    }

    void Point::deleteFeatureRef(std::shared_ptr<Feature> feat)
    {
        for (auto it = obs.begin(); it != obs.end(); ++it)
        {
            if ((*it) == feat)
            {
                obs.erase(it);
                return;
            }
        }
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

        cache_depth.resize(camera->width() * camera->height());
        cache_grid_flag.resize(m_grid_flat_length, false);
        cache_grid_points.resize(m_grid_flat_length, nullptr);
        cache_grid_dist.resize(m_grid_flat_length, 10000.0);
        cache_grid_cur.resize(m_grid_flat_length, 0.0);
        cache_grid_add_points.resize(m_grid_flat_length, Eigen::Vector3d::Zero());
    }

    void LidarSelector::updateFrameState()
    {
        kf::State &x = m_kf->x();

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
        cache_depth.assign(m_camera->width() * m_camera->height(), 0.0);
        cache_grid_flag.assign(m_grid_flat_length, false);
        cache_grid_points.assign(m_grid_flat_length, nullptr);
        cache_grid_dist.assign(m_grid_flat_length, 10000.0);
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

    void LidarSelector::addPoint(std::shared_ptr<Point> point)
    {
        Eigen::Vector3d pt_w(point->pos[0], point->pos[1], point->pos[2]);
        VoxelKey k = VoxelKey::index(pt_w.x(), pt_w.y(), pt_w.z(), m_voxel_size, 0.0);

        auto iter = m_feat_map.find(k);
        if (iter == m_feat_map.end())
        {
            m_feat_map[k] = std::vector<std::shared_ptr<Point>>();
        }
        m_feat_map[k].push_back(point);
    }

    void LidarSelector::addObservations(cv::Mat img)
    {
        int total_points = cache_reference_points.size();
        if (total_points <= 0)
            return;
        for (int i = 0; i < total_points; i++)
        {
            ReferencePoint &rp = cache_reference_points[i];
            Point *p_ptr = rp.point;
            Eigen::Vector2d pc = f2c(w2f(p_ptr->pos));

            std::shared_ptr<Feature> last_feature = p_ptr->obs.back();
            Eigen::Matrix3d last_r_wf = last_feature->r_wf();
            Eigen::Vector3d last_p_wf = last_feature->p_wf();
            double delta_p = (m_p_wf - last_p_wf).norm();
            Eigen::Matrix3d delta_r = last_r_wf.transpose() * m_r_wf;
            double delta_theta = (delta_r.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_r.trace() - 1));

            if (delta_p <= 0.5 && delta_theta <= 10)
                continue;
            Eigen::Vector2d last_px = last_feature->px;
            double pixel_dist = (pc - last_px).norm();
            if (pixel_dist <= 40)
                continue;

            if (p_ptr->obs.size() >= 20)
            {
                std::shared_ptr<Feature> feat_to_delete;
                if (p_ptr->getFurthestViewObs(m_p_wf, feat_to_delete))
                    p_ptr->deleteFeatureRef(feat_to_delete);
            }

            p_ptr->value = PinholeCamera::shiTomasiScore(img, static_cast<int>(pc[0]), static_cast<int>(pc[1]));
            Eigen::Vector3d pf = m_camera->cam2world(pc);

            cv::Mat patch0(m_patch_size, m_patch_size, CV_32FC1), patch1(m_patch_size, m_patch_size, CV_32FC1), patch2(m_patch_size, m_patch_size, CV_32FC1);
            getPatch(img, pc, patch0, 0);
            getPatch(img, pc, patch1, 1);
            getPatch(img, pc, patch2, 2);
            std::shared_ptr<Feature> feat = std::make_shared<Feature>(pc, pf, m_r_fw, m_p_fw, p_ptr->value, 0);
            feat->frame = img;
            feat->patches[0] = patch0;
            feat->patches[1] = patch1;
            feat->patches[2] = patch2;
            p_ptr->addObs(feat);
        }
    }

    int LidarSelector::incrVisualMap(cv::Mat img, CloudType::Ptr cloud)
    {
        cache_grid_flag.assign(m_grid_flat_length, false);
        for (int i = 0; i < cloud->size(); i++)
        {
            Eigen::Vector3d pw(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
            Eigen::Vector3d pf = w2f(pw);
            if (pf(2) <= 0)
                continue;
            Eigen::Vector2d pc = f2c(pf);
            if (!m_camera->isInFrame(pc.cast<int>(), (m_patch_size_half + 1) * 8))
                continue;

            int index = gridIndex(pc);
            double cur_value = static_cast<double>(PinholeCamera::shiTomasiScore(img, static_cast<int>(pc[0]), static_cast<int>(pc[1])));
            if (cur_value > cache_grid_cur[index])
            {

                cache_grid_cur[index] = cur_value;
                cache_grid_add_points[index] = pw;
                cache_grid_flag[index] = true;
            }
        }

        int count = 0;
        for (int i = 0; i < m_grid_flat_length; i++)
        {
            if (!cache_grid_flag[i])
                continue;
            Eigen::Vector3d pw = cache_grid_add_points[i];
            Eigen::Vector2d pc = f2c(w2f(pw));
            cv::Mat patch0(m_patch_size, m_patch_size, CV_32FC1), patch1(m_patch_size, m_patch_size, CV_32FC1), patch2(m_patch_size, m_patch_size, CV_32FC1);
            getPatch(img, pc, patch0, 0);
            getPatch(img, pc, patch1, 1);
            getPatch(img, pc, patch2, 2);
            std::shared_ptr<Point> p_add = std::make_shared<Point>(pw);
            Eigen::Vector3d pf = m_camera->cam2world(pc);
            std::shared_ptr<Feature> feat = std::make_shared<Feature>(pc, pf, m_r_fw, m_p_fw, cache_grid_cur[i], 0);
            feat->patches[0] = patch0;
            feat->patches[1] = patch1;
            feat->patches[2] = patch2;
            feat->frame = img;
            p_add->addObs(feat);
            p_add->value = cache_grid_cur[i];
            addPoint(p_add);
            count++;
        }
        return count;
    }

    bool LidarSelector::getReferencePoints(cv::Mat gray_img, CloudType::Ptr cloud)
    {
        if (m_feat_map.size() <= 0)
            return false;
        resetCache();
        CloudType::Ptr cloud_ds(new CloudType);
        m_scan_filter.setInputCloud(cloud);
        m_scan_filter.filter(*cloud_ds);

        std::unordered_set<VoxelKey, VoxelKey::Hasher> selected_voxels;
        std::cout << "111111111111111, SELECT VOXELS" << std::endl;
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

        cache_grid_cur.assign(m_grid_flat_length, 0.0);
        std::cout << "22222222222222, SELECT POINTS" << std::endl;
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
                double cur_value = p->value;
                if (cur_dist <= cache_grid_dist[index])
                {
                    cache_grid_dist[index] = cur_dist;
                    cache_grid_points[index] = p.get();
                }
                if (cur_value >= cache_grid_cur[index])
                    cache_grid_cur[index] = cur_value;
            }
        }

        std::vector<ReferencePoint>().swap(cache_reference_points);

        std::cout << "33333333333333, SELECT REFERENCE POINTS" << std::endl;
        for (int i = 0; i < m_grid_flat_length; i++)
        {
            if (!cache_grid_flag[i])
                continue;
            Point *p = cache_grid_points[i];
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
        std::cout << "FINDED LOSS PAIR ===================: " << cache_reference_points.size() << std::endl;

        return true;
    }

    void LidarSelector::process(cv::Mat img, CloudType::Ptr cloud, bool is_new_cloud)
    {

        if (img.cols != m_camera->width() || img.rows != m_camera->height())
        {
            cv::resize(img, img, cv::Size2i(m_camera->width(), m_camera->height()), 0, 0, cv::INTER_LINEAR);
        }
        m_img_bgr = img.clone();
        cv::cvtColor(img, m_img_gray, cv::COLOR_BGR2GRAY);
        std::cout << "is new cloud: " << is_new_cloud << " width: " << img.cols << " height: " << img.rows << std::endl;
        m_cloud = cloud;

        updateFrameState();

        std::cout << "GET REFERENCE POINTS" << std::endl;

        getReferencePoints(m_img_gray, cloud);

        std::cout << "INCRVISUALMAP" << std::endl;

        if (is_new_cloud)
        {
            int add_count = incrVisualMap(m_img_gray, cloud);
            std::cout << "FEAT_MAP_SIZE: " << m_feat_map.size() << std::endl;
            std::cout << "ADD POINT COUNT: " << add_count << std::endl;
        }

        std::cout << "FUNCTION FINISHED!" << std::endl;
        // 找到匹配特征

        // 更新状态

        updateFrameState();

        // 如果是新的地图点，需要更新视觉地图

        // 根据新的状态更新观测
    }

    Eigen::Vector3d LidarSelector::getPixelBRG(cv::Mat img_bgr, const Eigen::Vector2d &px)
    {
        const double u_ref = px[0];
        const double v_ref = px[1];
        const int u_ref_i = floorf(px[0]);
        const int v_ref_i = floorf(px[1]);
        const double subpix_u_ref = (u_ref - u_ref_i);
        const double subpix_v_ref = (v_ref - v_ref_i);
        const double w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
        const double w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
        const double w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
        const double w_ref_br = subpix_u_ref * subpix_v_ref;
        cv::Vec3b &tl = img_bgr.ptr<cv::Vec3b>(v_ref_i)[u_ref_i];
        cv::Vec3b &tr = img_bgr.ptr<cv::Vec3b>(v_ref_i)[u_ref_i + 1];
        cv::Vec3b &bl = img_bgr.ptr<cv::Vec3b>(v_ref_i + 1)[u_ref_i];
        cv::Vec3b &br = img_bgr.ptr<cv::Vec3b>(v_ref_i + 1)[u_ref_i + 1];

        Eigen::Vector3d tl_v(tl[0], tl[1], tl[2]);
        Eigen::Vector3d tr_v(tr[0], tr[1], tr[2]);
        Eigen::Vector3d bl_v(bl[0], bl[1], bl[2]);
        Eigen::Vector3d br_v(br[0], br[1], br[2]);

        return w_ref_tl * tl_v + w_ref_tr * tr_v + w_ref_bl * bl_v + w_ref_br * br_v;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr LidarSelector::getCurentCloudRGB()
    {

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ret(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (m_cloud == nullptr || m_cloud->size() <= 0)
            return ret;
        ret->reserve(m_cloud->size());
        for (auto p : m_cloud->points)
        {
            Eigen::Vector3d pw = Eigen::Vector3d(p.x, p.y, p.z);
            Eigen::Vector3d pf = w2f(pw);
            if (pf(2) <= 0)
                continue;
            Eigen::Vector2d pc = m_camera->world2cam(pf);
            if (!m_camera->isInFrame(pc.cast<int>(), 1))
                continue;
            Eigen::Vector3f bgr = getPixelBRG(m_img_bgr, pc).cast<float>();
            pcl::PointXYZRGB p_color;
            p_color.x = p.x;
            p_color.y = p.y;
            p_color.z = p.z;
            p_color.b = bgr(0);
            p_color.g = bgr(1);
            p_color.r = bgr(2);
            ret->push_back(p_color);
        }
        return ret;
    }
} // namespace livo
