#include "camera.h"

namespace livo
{
    PinholeCamera::PinholeCamera(double width, double height,
                                 double fx, double fy,
                                 double cx, double cy,
                                 double d0, double d1, double d2, double d3, double d4)
        : m_width(width), m_height(height),
          m_fx(fx), m_fy(fy), m_cx(cx), m_cy(cy),
          m_distort(std::abs(d0) > 0.0000001),
          m_undist_map_x(height, width, CV_16SC2),
          m_undist_map_y(height, width, CV_16SC2)
    {
        m_d[0] = d0;
        m_d[1] = d1;
        m_d[2] = d2;
        m_d[3] = d3;
        m_d[4] = d4;
        m_cvK = (cv::Mat_<float>(3, 3) << m_fx, 0.0, m_cx, 0.0, m_fy, m_cy, 0.0, 0.0, 1.0);
        m_cvD = (cv::Mat_<float>(1, 5) << m_d[0], m_d[1], m_d[2], m_d[3], m_d[4]);
        cv::initUndistortRectifyMap(m_cvK, m_cvD, cv::Mat_<double>::eye(3, 3), m_cvK,
                                    cv::Size(m_width, m_height), CV_16SC2, m_undist_map_x, m_undist_map_y);
        m_K << m_fx, 0.0, m_cx, 0.0, m_fy, m_cy, 0.0, 0.0, 1.0;
        m_K_inv = m_K.inverse();
    }

    Eigen::Vector3d PinholeCamera::cam2world(const double &u, const double &v) const
    {
        Eigen::Vector3d xyz;
        if (!m_distort)
        {
            xyz[0] = (u - m_cx) / m_fx;
            xyz[1] = (v - m_cy) / m_fy;
            xyz[2] = 1.0;
        }
        else
        {
            cv::Point2f uv(u, v), px;
            const cv::Mat src_pt(1, 1, CV_32FC2, &uv.x);
            cv::Mat dst_pt(1, 1, CV_32FC2, &px.x);
            cv::undistortPoints(src_pt, dst_pt, m_cvK, m_cvD);
            xyz[0] = px.x;
            xyz[1] = px.y;
            xyz[2] = 1.0;
        }
        return xyz.normalized();
    }

    Eigen::Vector3d PinholeCamera::cam2world(const Eigen::Vector2d &uv) const
    {
        return cam2world(uv(0), uv(1));
    }

    Eigen::Vector2d PinholeCamera::world2cam(const Eigen::Vector2d &uv) const
    {
        Eigen::Vector2d px;
        if (!m_distort)
        {
            px[0] = m_fx * uv[0] + m_cx;
            px[1] = m_fy * uv[1] + m_cy;
        }
        else
        {
            double x, y, r2, r4, r6, a1, a2, a3, cdist, xd, yd;
            x = uv[0];
            y = uv[1];
            r2 = x * x + y * y;
            r4 = r2 * r2;
            r6 = r4 * r2;
            a1 = 2 * x * y;
            a2 = r2 + 2 * x * x;
            a3 = r2 + 2 * y * y;
            cdist = 1 + m_d[0] * r2 + m_d[1] * r4 + m_d[4] * r6;
            xd = x * cdist + m_d[2] * a1 + m_d[3] * a2;
            yd = y * cdist + m_d[2] * a3 + m_d[3] * a1;
            px[0] = xd * m_fx + m_cx;
            px[1] = yd * m_fy + m_cy;
        }
        return px;
    }

    Eigen::Vector2d PinholeCamera::world2cam(const Eigen::Vector3d &xyz) const
    {
        Eigen::Vector2d uv = xyz.head<2>() / xyz[2];
        return world2cam(uv);
    }

    void PinholeCamera::undistortImage(const cv::Mat &raw, cv::Mat &rectified)
    {
        if (m_distort)
            cv::remap(raw, rectified, m_undist_map_x, m_undist_map_y, cv::INTER_LINEAR);
        else
            rectified = raw.clone();
    }

    bool PinholeCamera::isInFrame(const Eigen::Vector2i &obs, int boundary) const
    {
        if (obs(0) >= boundary && obs(0) < width() - boundary && obs[1] >= boundary && obs[1] < height() - boundary)
            return true;
        return false;
    }

    float PinholeCamera::shiTomasiScore(const cv::Mat &img, int u, int v)
    {
        assert(img.type() == CV_8UC1);

        float dXX = 0.0;
        float dYY = 0.0;
        float dXY = 0.0;
        const int halfbox_size = 4;
        const int box_size = 2 * halfbox_size;
        const int box_area = box_size * box_size;
        const int x_min = u - halfbox_size;
        const int x_max = u + halfbox_size;
        const int y_min = v - halfbox_size;
        const int y_max = v + halfbox_size;

        if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1)
            return 0.0; // patch is too close to the boundary

        const int stride = img.step.p[0];
        for (int y = y_min; y < y_max; ++y)
        {
            const uint8_t *ptr_left = img.data + stride * y + x_min - 1;
            const uint8_t *ptr_right = img.data + stride * y + x_min + 1;
            const uint8_t *ptr_top = img.data + stride * (y - 1) + x_min;
            const uint8_t *ptr_bottom = img.data + stride * (y + 1) + x_min;
            for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
            {
                float dx = *ptr_right - *ptr_left;
                float dy = *ptr_bottom - *ptr_top;
                dXX += dx * dx;
                dYY += dy * dy;
                dXY += dx * dy;
            }
        }

        // Find and return smaller eigenvalue:
        dXX = dXX / (2.0 * box_area);
        dYY = dYY / (2.0 * box_area);
        dXY = dXY / (2.0 * box_area);
        return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
    }

    float PinholeCamera::interpolateMat_8u(const cv::Mat &mat, float u, float v)
    {
        assert(mat.type() == CV_8U);
        int x = floor(u);
        int y = floor(v);
        float subpix_x = u - x;
        float subpix_y = v - y;

        float w00 = (1.0f - subpix_x) * (1.0f - subpix_y);
        float w01 = (1.0f - subpix_x) * subpix_y;
        float w10 = subpix_x * (1.0f - subpix_y);
        float w11 = 1.0f - w00 - w01 - w10;

        const int stride = mat.step.p[0];
        unsigned char *ptr = mat.data + y * stride + x;
        return w00 * ptr[0] + w01 * ptr[stride] + w10 * ptr[1] + w11 * ptr[stride + 1];

    }
    
} // namespace livo
