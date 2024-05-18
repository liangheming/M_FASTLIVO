#include "utils.h"

// void livox2pcl(const livox_ros_driver2::CustomMsg::ConstPtr &msg, pcl::PointCloud<pcl::PointXYZINormal>::Ptr out, int filter_num, double blind)
// {
//     int point_num = msg->point_num;
//     out->clear();
//     out->reserve(point_num / filter_num + 1);

//     uint valid_num = 0;
//     for (uint i = 0; i < point_num; i++)
//     {
//         if ((msg->points[i].line < 4) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
//         {
//             if ((valid_num++) % filter_num != 0)
//                 continue;
//             pcl::PointXYZINormal p;
//             p.x = msg->points[i].x;
//             p.y = msg->points[i].y;
//             p.z = msg->points[i].z;
//             p.intensity = msg->points[i].reflectivity;
//             p.curvature = msg->points[i].offset_time / float(1000000); // 纳秒->毫秒
//             if ((p.x * p.x + p.y * p.y + p.z * p.z > (blind * blind)))
//             {
//                 out->push_back(p);
//             }
//         }
//     }
// }

void livox2pcl(const livox_ros_driver::CustomMsg::ConstPtr &msg, pcl::PointCloud<pcl::PointXYZINormal>::Ptr out, int filter_num, double blind, double max_range)
{
    int point_num = msg->point_num;
    out->clear();
    out->reserve(point_num / filter_num + 1);

    uint valid_num = 0;
    for (uint i = 0; i < point_num; i++)
    {
        if ((msg->points[i].line <= 6) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
        {
            if ((valid_num++) % filter_num != 0)
                continue;
            pcl::PointXYZINormal p;
            p.x = msg->points[i].x;
            p.y = msg->points[i].y;
            p.z = msg->points[i].z;
            p.intensity = msg->points[i].reflectivity;
            p.curvature = msg->points[i].offset_time / float(1000000); // 纳秒->毫秒
            double sq_range = p.x * p.x + p.y * p.y + p.z * p.z;
            if (sq_range > (blind * blind) && sq_range < (max_range * max_range))
            {
                out->push_back(p);
            }
        }
    }
}

cv::Mat msg2cv(const sensor_msgs::ImageConstPtr &img_msg)
{
    return cv_bridge::toCvCopy(img_msg, "bgr8")->image;
}

sensor_msgs::PointCloud2 pcl2msg(pcl::PointCloud<pcl::PointXYZINormal>::Ptr inp, const std::string &frame_id, const double &timestamp)
{
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*inp, msg);
    if (timestamp < 0)
        msg.header.stamp = ros::Time().now();
    else
        msg.header.stamp = ros::Time().fromSec(timestamp);
    msg.header.frame_id = frame_id;
    return msg;
}

geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp)
{
    geometry_msgs::TransformStamped transform;
    transform.header.frame_id = frame_id;
    transform.header.stamp = ros::Time().fromSec(timestamp);
    transform.child_frame_id = child_frame_id;
    transform.transform.translation.x = pos(0);
    transform.transform.translation.y = pos(1);
    transform.transform.translation.z = pos(2);
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);

    transform.transform.rotation.w = q.w();
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    return transform;
}