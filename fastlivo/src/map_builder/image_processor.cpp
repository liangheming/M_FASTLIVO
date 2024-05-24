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
        Eigen::Vector3d dir((*it)->t_wf() - pos);
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
        double dist = ((*it)->t_wf() - cam_pos).norm();
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

