#include "image_processor.h"

void Point::addObs(std::shared_ptr<Feature> ftr)
{
    obs.push_front(ftr);
}

bool Point::getCloseViewObs(const V3D &cam_pos, std::shared_ptr<Feature> &out, double thresh)
{
    if (obs.size() <= 0)
        return false;
    V3D obs_dir(cam_pos - pos);
    obs_dir.normalize();
    auto min_it = obs.begin();
    double min_cos_angle = 0;
    for (auto it = obs.begin(); it != obs.end(); ++it)
    {
        Eigen::Vector3d dir((*it)->t_wc() - pos);
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

bool Point::getFurthestViewObs(const V3D &cam_pos, std::shared_ptr<Feature> &out)
{
    if (obs.size() <= 0)
        return false;

    auto max_it = obs.begin();
    double maxdist = 0.0;
    for (auto it = obs.begin(); it != obs.end(); it++)
    {
        double dist = ((*it)->t_wc() - cam_pos).norm();
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

ImageProcessor::ImageProcessor(Config &config, std::shared_ptr<IESKF> kf)
    : m_kf(kf), m_config(config)
{
    m_camera = std::make_shared<PinholeCamera>(m_config.cam_width, m_config.cam_height,
                                               m_config.cam_fx, m_config.cam_fy,
                                               m_config.cam_cx, m_config.cam_cy,
                                               m_config.cam_d[0], m_config.cam_d[1],
                                               m_config.cam_d[2], m_config.cam_d[3],
                                               m_config.cam_d[4]);
    m_grid_width = static_cast<int>(std::ceil(m_camera->width() / static_cast<double>(m_config.grid_size)));
    m_grid_height = static_cast<int>(std::ceil(m_camera->height() / static_cast<double>(m_config.grid_size)));
    m_grid_num = m_grid_height * m_grid_height;
    m_patch_size = m_config.half_patch_size * 2 + 1;
    cache_pixel_depth.resize(m_camera->width() * m_camera->height(), 0.0);

    cache_flag.resize(m_grid_num, false);
    cache_points.resize(m_grid_num, nullptr);
    cache_score.resize(m_grid_num, 0.0);
    cache_grid_depth.resize(m_grid_num, 10000.0);
    m_cloud_filter.setLeafSize(m_config.selector_scan_resolution, m_config.selector_scan_resolution, m_config.selector_scan_resolution);
}

int ImageProcessor::gridIndex(const V2D &px)
{
    return static_cast<int>(px(1) / m_config.grid_size) * m_grid_width + static_cast<int>(px(0) / m_config.grid_size);
}
/**
 * 以中心点为参考点，将参考帧上的像素点转换到当前帧的像素
 */
M2D ImageProcessor::getCRAffine2d(std::shared_ptr<Feature> ref_ptr)
{
    double depth = (ref_ptr->t_wc() - ref_ptr->point.lock()->pos).norm();
    V3D xyz_ref = (ref_ptr->pc * depth);

    V3D xyz_du_ref = m_camera->img2Cam(ref_ptr->px + V2D(m_config.half_patch_size, 0.0));
    V3D xyz_dv_ref = m_camera->img2Cam(ref_ptr->px + V2D(0.0, m_config.half_patch_size));

    xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
    xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];

    M3D r_cr = r_cw() * ref_ptr->r_wc();
    V3D t_cr = r_cw() * (ref_ptr->t_wc() - t_wc());

    V3D pc_cur = r_cr * xyz_ref + t_cr;
    V3D pc_du = r_cr * xyz_du_ref + t_cr;
    V3D pc_dv = r_cr * xyz_dv_ref + t_cr;

    V2D px_cur = m_camera->cam2Img(pc_cur);
    V2D px_du = m_camera->cam2Img(pc_du);
    V2D px_dv = m_camera->cam2Img(pc_dv);

    M2D ret;
    ret.col(0) = (px_du - px_cur) / m_config.half_patch_size;
    ret.col(1) = (px_dv - px_cur) / m_config.half_patch_size;
    return ret;
}

// 参考帧的原始图像对应当前帧最好的金字塔层级
int ImageProcessor::getBestSearchLevel(const M2D &affine_cr, const int max_level)
{
    int search_level = 0;
    double D = affine_cr.determinant();

    while (D > 3.0 && search_level < max_level)
    {
        search_level += 1;
        D *= 0.25;
    }
    return search_level;
}

// 获得参考帧上图像patch
void ImageProcessor::getRefAffinePatch(const M2D &affine_rc, const V2D &px_ref, const cv::Mat &img_ref, const int search_level, cv::Mat &patch)
{
    assert(img_ref.type() == CV_8U);
    assert(patch.type() == CV_32FC1);
    assert(patch.rows == m_patch_size && patch.cols == m_patch_size);
    for (int y = 0; y < m_patch_size; ++y)
    {
        for (int x = 0; x < m_patch_size; ++x)
        {
            Eigen::Vector2d px_patch(x - m_config.half_patch_size, y - m_config.half_patch_size);
            px_patch *= (1 << search_level);
            Eigen::Vector2d px_ref = affine_rc * px_patch + px_ref;
            if (px_ref(0) < 0 || px_ref(1) < 0 || px_ref(0) >= img_ref.cols - 1 || px_ref(1) >= img_ref.rows - 1)
                patch.ptr<float>(y)[x] = 0.0;
            else
                patch.ptr<float>(y)[x] = CVUtils::interpolateMat_8u(img_ref, static_cast<float>(px_ref(0)), static_cast<float>(px_ref(1)));
        }
    }
}

void ImageProcessor::selectReference(CloudType::Ptr cloud)
{
    if (m_featmap.size() <= 0)
        return;
    CloudType::Ptr cloud_down(new CloudType);
    m_cloud_filter.setInputCloud(cloud);
    m_cloud_filter.filter(*cloud_down);

    // 记录最近点云所关联的Voxel，并记录点云到相机的深度
    cache_pixel_depth.assign(cache_pixel_depth.size(), 0.0);
    std::unordered_set<VoxelKey, VoxelKey::Hasher> selected_voxels;
    for (int i = 0; i < cloud_down->size(); i++)
    {
        V3D pw(cloud_down->points[i].x, cloud_down->points[i].y, cloud_down->points[i].z);
        VoxelKey position = VoxelKey::index(pw(0), pw(1), pw(2), m_config.selector_voxel_size, 0.0);
        if (selected_voxels.find(position) == selected_voxels.end())
            selected_voxels.insert(position);
        V3D pc = r_cw() * pw + t_cw();
        if (pc(2) <= 0)
            continue;
        V2D px = m_camera->cam2Img(pc);
        if (!m_camera->isInImg(px.cast<int>(), (m_config.half_patch_size + 1) * 8))
            continue;
        double depth = pc(2);
        int col = static_cast<int>(px(0));
        int row = static_cast<int>(px(1));
        cache_pixel_depth[m_camera->width() * row + col] = depth;
    }

    // 以grid为单位，找到每个grid中观测最近的Point;
    cache_flag.assign(cache_flag.size(), false);
    cache_score.assign(cache_score.size(), 0.0);
    cache_points.assign(cache_points.size(), nullptr);
    cache_grid_depth.assign(cache_grid_depth.size(), 10000.0);
    for (const VoxelKey &position : selected_voxels)
    {
        if (m_featmap.find(position) == m_featmap.end())
            continue;
        Vec<std::shared_ptr<Point>> &points = m_featmap[position];
        for (int i = 0; i < points.size(); i++)
        {
            std::shared_ptr<Point> point_ptr = points[i];
            V3D pc = r_cw() * point_ptr->pos + t_cw();
            if (pc(2) <= 0)
                continue;
            V2D px = m_camera->cam2Img(pc);
            if (!m_camera->isInImg(px.cast<int>(), (m_config.half_patch_size + 1) * 8))
                continue;

            int index = gridIndex(px);
            cache_flag[index] = true;
            V3D obs_vec(t_wc() - point_ptr->pos);
            float cur_dist = obs_vec.norm();
            float cur_value = point_ptr->value;
            if (cur_dist <= cache_grid_depth[index])
            {
                cache_points[index] = point_ptr.get();
                cache_grid_depth[index] = cur_dist;
            }
            if (cur_value >= cache_score[index])
            {
                cache_score[index] = cur_value;
            }
        }
    }

    // 获得相应的参考特征，计算损失
    for (int i = 0; i < m_grid_num; i++)
    {
        if (!cache_flag[i])
            continue;
        Point *p_ptr = cache_points[i];
        V3D pc = r_cw() * p_ptr->pos + t_cw();
        V3D px = m_camera->cam2Img(pc);

        // 检查深度是否连续
        bool depth_continous = false;
        for (int u = -m_config.half_patch_size; u <= m_config.half_patch_size; u++)
        {
            for (int v = -m_config.half_patch_size; v <= m_config.half_patch_size; v++)
            {
                if (u == 0 && v == 0)
                    continue;
                double depth = cache_pixel_depth[m_camera->width() * (v + static_cast<int>(px(1))) + u + static_cast<int>(px(0))];
                if (depth == 0.)
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

        std::shared_ptr<Feature> feat_ptr;
        if (!p_ptr->getCloseViewObs(t_wc(), feat_ptr))
            continue;
        M2D affine_cr = getCRAffine2d(feat_ptr);
        int best_search_level = getBestSearchLevel(affine_cr, 2);

        M2D affine_rc = affine_cr.inverse();
        if (affine_rc.hasNaN())
            continue;

        cv::Mat ref_patch(m_patch_size, m_patch_size, CV_32FC1);
        getRefAffinePatch(affine_rc, feat_ptr->px, feat_ptr->frame, best_search_level, ref_patch);
        cv::Mat cur_patch(m_patch_size, m_patch_size, CV_32FC1);
        CVUtils::getPatch(m_cur_img_gray, px, cur_patch, m_config.half_patch_size, best_search_level);

        double sq_dist = cv::sum((ref_patch - cur_patch) * (ref_patch - cur_patch))[0];

        if (sq_dist > m_config.pixel_sq_dist_thresh * m_patch_size * m_patch_size)
            continue;
        
    }
}

void ImageProcessor::process(cv::Mat &img, CloudType::Ptr cloud, bool is_new_cloud)
{
    if (img.cols != m_camera->width() || img.rows != m_camera->height())
        cv::resize(img, m_cur_img_color, cv::Size2i(m_camera->width(), m_camera->height()));
    else
        m_cur_img_color = img;
    cv::cvtColor(m_cur_img_color, m_cur_img_gray, cv::COLOR_BGR2GRAY);
}

M3D ImageProcessor::r_cw()
{
    const State &s = m_kf->x();
    return s.r_cl * s.r_il.transpose() * s.r_wi.transpose();
}

V3D ImageProcessor::t_cw()
{
    const State &s = m_kf->x();
    return -s.r_cl * s.r_il.transpose() * (s.r_wi.transpose() * s.t_wi + s.t_il) + s.t_cl;
}

M3D ImageProcessor::r_ci()
{
    const State &s = m_kf->x();
    return s.r_cl * s.r_il.transpose();
}

V3D ImageProcessor::t_ci()
{
    const State &s = m_kf->x();

    return -s.r_cl * s.r_il.transpose() * s.t_il + s.t_cl;
}

M3D ImageProcessor::r_wc()
{
    return r_cw().transpose();
}

V3D ImageProcessor::t_wc()
{
    return -r_cw().transpose() * t_cw();
}

void ImageProcessor::incrVisualMap()
{
}