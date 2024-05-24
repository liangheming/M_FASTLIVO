#pragma once
#include "commons.h"

class CVUtils
{
    static float interpolateMat_8u(const cv::Mat &mat, float u, float v);

    static float shiTomasiScore(const cv::Mat &img, int u, int v);

    static Eigen::Vector3f interpolateMat_color(const cv::Mat &mat, float u, float v);

    bool getPatch(cv::Mat img, const V2D px, cv::Mat &patch, int half_path, int level = 0);
};

class PinholeCamera
{
public:
    PinholeCamera(double width, double height,
                  double fx, double fy,
                  double cx, double cy,
                  double d0, double d1, double d2, double d3, double d4);

    V3D img2Cam(const double &u, const double &v) const;

    V3D img2Cam(const V2D &uv) const;

    V2D cam2Img(const V2D &uv) const;

    V2D cam2Img(const V3D &xyz) const;

    inline V2D project2d(const V3D &v) const { return v.head<2>() / v[2]; }

    inline int width() const { return static_cast<int>(m_width); }

    inline int height() const { return static_cast<int>(m_height); }

    inline double fx() const { return m_fx; }

    inline double fy() const { return m_fy; }

    inline double cx() const { return m_cx; }

    inline double cy() const { return m_cy; }

    bool isInImg(const Eigen::Vector2i &obs, int boundary = 0) const;

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
};