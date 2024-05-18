#pragma once
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace livo
{
    class PinholeCamera
    {
    public:
        PinholeCamera(double width, double height,
                      double fx, double fy,
                      double cx, double cy,
                      double d0, double d1, double d2, double d3, double d4);

        Eigen::Vector3d cam2world(const double &u, const double &v) const;

        Eigen::Vector3d cam2world(const Eigen::Vector2d &uv) const;

        Eigen::Vector2d world2cam(const Eigen::Vector2d &uv) const;

        Eigen::Vector2d world2cam(const Eigen::Vector3d &xyz) const;

        void undistortImage(const cv::Mat &raw, cv::Mat &rectified);

        int width() const { return static_cast<int>(m_width); }

        int height() const { return static_cast<int>(m_height); }

        bool isInFrame(const Eigen::Vector2i &obs, int boundary = 0) const;

        static float shiTomasiScore(const cv::Mat& img, int u, int v);

    private:
        double m_width;
        double m_height;
        double m_fx;
        double m_fy;
        double m_cx;
        double m_cy;
        double m_d[5];
        bool m_distort;
        cv::Mat m_cvK;
        cv::Mat m_cvD;
        cv::Mat m_undist_map_x;
        cv::Mat m_undist_map_y;
        Eigen::Matrix3d m_K;
        Eigen::Matrix3d m_K_inv;
    };
} // namespace livo
