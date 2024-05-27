#include "ieskf.h"

M3D rightJacobian(const V3D &inp)
{
    return Sophus::SO3d::leftJacobian(inp).transpose();
}

M3D jrInv(const V3D &inp)
{
    return Sophus::SO3d::leftJacobianInverse(inp).transpose();
}

void State::operator+=(const V27D &delta)
{
    r_wi *= Sophus::SO3d::exp(delta.segment<3>(0)).matrix();
    t_wi += delta.segment<3>(3);
    r_il *= Sophus::SO3d::exp(delta.segment<3>(6)).matrix();
    t_il += delta.segment<3>(9);
    r_cl *= Sophus::SO3d::exp(delta.segment<3>(12)).matrix();
    t_cl += delta.segment<3>(15);
    v += delta.segment<3>(18);
    bg += delta.segment<3>(21);
    ba += delta.segment<3>(24);
}

V27D State::operator-(const State &other)
{
    V27D delta = V27D::Zero();
    delta.segment<3>(0) = Sophus::SO3d(other.r_wi.transpose() * r_wi).log();
    delta.segment<3>(3) = t_wi - other.t_wi;
    delta.segment<3>(6) = Sophus::SO3d(other.r_il.transpose() * r_il).log();
    delta.segment<3>(9) = t_il - other.t_il;
    delta.segment<3>(12) = Sophus::SO3d(other.r_cl.transpose() * r_cl).log();
    delta.segment<3>(15) = t_cl - other.t_cl;
    delta.segment<3>(18) = v - other.v;
    delta.segment<3>(21) = bg - other.bg;
    delta.segment<3>(24) = ba - other.ba;
    return delta;
}

std::ostream &operator<<(std::ostream &os, const State &state)
{
    os << "t_wi: " << state.t_wi.transpose() << std::endl
       << "v: " << state.v.transpose() << std::endl
       << "ba: " << state.ba.transpose() << std::endl
       << "bg: " << state.bg.transpose() << std::endl
       << "t_il" << state.t_il.transpose() << std::endl
       << "t_cl" << state.t_cl.transpose() << std::endl
       << "g" << state.g.transpose() << std::endl;
    return os;
}

IESKF::IESKF() = default;

void IESKF::predict(const Input &inp, double dt, const M12D &Q)
{
    V27D delta = V27D::Zero();
    delta.segment<3>(0) = (inp.gyro - m_x.bg) * dt;
    delta.segment<3>(3) = m_x.v * dt;
    delta.segment<3>(18) = (m_x.r_wi * (inp.acc - m_x.ba) + m_x.g) * dt;
    m_F.setIdentity();
    m_F.block<3, 3>(0, 0) = Sophus::SO3d::exp(-(inp.gyro - m_x.bg) * dt).matrix();
    m_F.block<3, 3>(0, 21) = -rightJacobian((inp.gyro - m_x.bg) * dt) * dt;
    m_F.block<3, 3>(3, 18) = Eigen::Matrix3d::Identity() * dt;
    m_F.block<3, 3>(18, 0) = -m_x.r_wi * Sophus::SO3d::hat(inp.acc - m_x.ba) * dt;
    m_F.block<3, 3>(18, 24) = -m_x.r_wi * dt;

    m_G.setZero();
    m_G.block<3, 3>(0, 0) = -rightJacobian((inp.gyro - m_x.bg) * dt) * dt;
    m_G.block<3, 3>(18, 3) = -m_x.r_wi * dt;
    m_G.block<3, 3>(21, 6) = Eigen::Matrix3d::Identity() * dt;
    m_G.block<3, 3>(24, 9) = Eigen::Matrix3d::Identity() * dt;
    m_x += delta;
    m_P = m_F * m_P * m_F.transpose() + m_G * Q * m_G.transpose();
}

void IESKF::update()
{
    State predict_x = m_x;
    SharedState shared_data;
    shared_data.iter_num = 0;
    shared_data.res = 1e10;
    V27D delta = V27D::Zero();
    M27D H = M27D::Identity();
    V27D b;
    for (size_t i = 0; i < m_max_iter; i++)
    {
        m_loss_func(m_x, shared_data);
        if (!shared_data.valid)
            break;
        H.setZero();
        b.setZero();
        delta = m_x - predict_x;
        M27D J = M27D::Identity();

        J.block<3, 3>(0, 0) = jrInv(delta.segment<3>(0));
        J.block<3, 3>(6, 6) = jrInv(delta.segment<3>(6));
        J.block<3, 3>(12, 12) = jrInv(delta.segment<3>(12));
        b += (J.transpose() * m_P.inverse() * delta);
        H += (J.transpose() * m_P.inverse() * J);
        H.block<18, 18>(0, 0) += shared_data.H;
        b.block<18, 1>(0, 0) += shared_data.b;
        delta = -H.inverse() * b;
        m_x += delta;
        shared_data.iter_num += 1;
        // if (delta.maxCoeff() < m_eps)
        //     break;
        if (m_stop_func(delta))
            break;
    }
    M27D L = M27D::Identity();
    L.block<3, 3>(0, 0) = jrInv(delta.segment<3>(0));
    L.block<3, 3>(6, 6) = jrInv(delta.segment<3>(6));
    L.block<3, 3>(12, 12) = jrInv(delta.segment<3>(12));
    m_P = L * H.inverse() * L.transpose();
    // m_P = L.transpose() * H.inverse() * L;
}
