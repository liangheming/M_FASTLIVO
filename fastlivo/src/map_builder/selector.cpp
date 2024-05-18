#include "selector.h"

namespace livo
{
    VoxelKey VoxelKey::index(double x, double y, double z, double resolution, double bias)
    {
        Eigen::Vector3d point(x, y, z);
        Eigen::Vector3d idx = (point / resolution + Eigen::Vector3d(bias, bias, bias)).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }
} // namespace livo
