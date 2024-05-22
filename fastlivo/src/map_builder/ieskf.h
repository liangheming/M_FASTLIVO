#pragma once
#include "commons.h"
#include <sophus/so3.hpp>

const double GRAVITY = 9.81;
using V18D = Eigen::Matrix<double, 18, 1>;
using V27D = Eigen::Matrix<double, 27, 1>;
using M12D = Eigen::Matrix<double, 12, 12>;
using M18D = Eigen::Matrix<double,18,18>;
using M27D = Eigen::Matrix<double, 27, 27>;
using M27x12D = Eigen::Matrix<double, 27, 12>;

M3D rightJacobian(const V3D &inp);

struct SharedState
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    M18D H;
    V18D b;
    double res = 1e10;
    bool valid = false;
    size_t iter_num = 0;
};

struct State
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    M3D r_wi = M3D::Identity();
    V3D t_wi = V3D::Zero();
    M3D r_il = M3D::Identity();
    V3D t_il = V3D::Zero();
    M3D r_cl = M3D::Identity();
    V3D t_cl = V3D::Zero();
    V3D v = V3D::Zero();
    V3D bg = V3D::Zero();
    V3D ba = V3D::Zero();
    V3D g = V3D(0.0, 0.0, -GRAVITY);

    void initGWithDir(const V3D &gravity_dir) { g = gravity_dir.normalized() * GRAVITY; }
    
    void initG(const V3D &gravity) { g = gravity; }

    void operator+=(const V27D &delta);

    V27D operator-(const State &other);

    friend std::ostream &operator<<(std::ostream &os, const State &state);

};

struct Input
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    V3D acc;
    V3D gyro;
    Input() = default;
    Input(V3D &a, V3D &g) : acc(a), gyro(g) {}
    Input(double a1, double a2, double a3, double g1, double g2, double g3) : acc(a1, a2, a3), gyro(g1, g2, g3) {}
};

using measure_func = std::function<void(State &, SharedState &)>;

class IESKF
{
public:
    IESKF();

    IESKF(size_t max_iter) : m_max_iter(max_iter) {}

    State &x() { return m_x; }

    void change_x(const State &x) { m_x = x; }

    M27D &P() { return m_P; }

    void set_share_function(measure_func func) { m_func = func; }

    void change_P(const M27D &P) { m_P = P; }

    void predict(const Input &inp, double dt, const M12D &Q);
    
    void update();

private:
    size_t m_max_iter = 5;
    double m_eps = 0.001;
    State m_x;
    M27D m_P;
    measure_func m_func;
    M27D m_F;
    M27x12D m_G;
};
