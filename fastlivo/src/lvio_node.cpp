#include <ros/ros.h>
#include <queue>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <tf2_ros/transform_broadcaster.h>
#include <opencv2/core.hpp>
#include "utils.h"
#include "map_builder/map_builder.h"

struct Config
{
    std::string imu_topic = "/livox/imu";
    std::string lidar_topic = "/livox/lidar";
    std::string image_topic = "/left_camera/image";
    std::string map_frame = "lidar";
    std::string body_frame = "body";
};
struct DataGroup
{
    bool lidar_pushed = false;

    std::mutex imu_mutex;
    std::mutex lidar_mutex;
    std::mutex image_mutex;

    double last_imu_time;
    double last_lidar_time;
    double last_image_time;

    std::deque<livo::IMUData> imu_buffer;
    std::deque<std::pair<double, livo::CloudType::Ptr>> lidar_buffer;
    std::deque<std::pair<double, cv::Mat>> image_buffer;
};

class LIVONode
{
public:
    LIVONode() : m_nh("~")
    {
        loadCofig();
        initSubsriber();
        initPublisher();
        m_builder = std::make_shared<livo::MapBuilder>(m_builder_config);
        main_loop = m_nh.createTimer(ros::Duration(0.05), &LIVONode::mainCB, this);
    }

    void loadCofig()
    {
    }

    void initSubsriber()
    {
        m_lidar_sub = m_nh.subscribe(m_node_config.lidar_topic, 10000, &LIVONode::lidarCB, this);
        m_imu_sub = m_nh.subscribe(m_node_config.imu_topic, 10000, &LIVONode::imuCB, this);
        m_image_sub = m_nh.subscribe(m_node_config.image_topic, 10000, &LIVONode::imageCB, this);
    }

    void initPublisher()
    {
        m_body_cloud_pub = m_nh.advertise<sensor_msgs::PointCloud2>("body_cloud", 1000);
        m_world_cloud_pub = m_nh.advertise<sensor_msgs::PointCloud2>("world_cloud", 1000);
    }

    void imuCB(const sensor_msgs::Imu::ConstPtr msg)
    {
        std::lock_guard<std::mutex> lock(m_group_data.imu_mutex);
        double timestamp = msg->header.stamp.toSec();
        if (timestamp < m_group_data.last_imu_time)
        {
            ROS_WARN("IMU TIME SYNC ERROR");
            m_group_data.imu_buffer.clear();
        }

        m_group_data.last_imu_time = timestamp;
        m_group_data.imu_buffer.emplace_back(Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z) * 10.0,
                                             Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                                             timestamp);
    }

    void lidarCB(const livox_ros_driver::CustomMsg::ConstPtr msg)
    {
        livo::CloudType::Ptr cloud(new livo::CloudType);
        livox2pcl(msg, cloud, 3, 0.5);
        std::lock_guard<std::mutex> lock(m_group_data.lidar_mutex);
        double timestamp = msg->header.stamp.toSec();
        if (timestamp < m_group_data.last_lidar_time)
        {
            ROS_WARN("LIDAR TIME SYNC ERROR");
            m_group_data.lidar_buffer.clear();
        }
        m_group_data.last_lidar_time = timestamp;
        m_group_data.lidar_buffer.emplace_back(timestamp, cloud);
    }

    void imageCB(const sensor_msgs::Image::ConstPtr msg)
    {
        std::lock_guard<std::mutex> lock(m_group_data.image_mutex);
        double timestamp = msg->header.stamp.toSec();

        if (timestamp < m_group_data.last_image_time)
        {
            ROS_WARN("IMAGE TIME SYNC ERROR");
            m_group_data.image_buffer.clear();
        }
        m_group_data.image_buffer.emplace_back(timestamp, msg2cv(msg));
        m_group_data.last_image_time = timestamp;
    }

    bool syncPackage()
    {
        // 1.0 no image no lidar
        if (m_group_data.lidar_buffer.empty())
            return false;
        if (!m_group_data.lidar_pushed)
        {
            m_sync_pack.cloud = m_group_data.lidar_buffer.front().second;
            std::sort(m_sync_pack.cloud->points.begin(), m_sync_pack.cloud->points.end(), [](livo::PointType &p1, livo::PointType &p2)
                      { return p1.curvature < p2.curvature; });
            m_sync_pack.cloud_start_time = m_group_data.lidar_buffer.front().first;
            m_sync_pack.cloud_end_time = m_sync_pack.cloud_start_time + m_sync_pack.cloud->points.back().curvature / double(1000.0);
            m_sync_pack.lidar_end = false;
            m_group_data.lidar_pushed = true;
        }
        // 如果没有图片帧，或者图片帧已经晚于当前雷达帧的最后时间
        if (m_group_data.image_buffer.empty() || m_group_data.image_buffer.front().first > m_sync_pack.cloud_end_time)
        {
            // no enough imu
            if (m_group_data.last_imu_time < m_sync_pack.cloud_end_time)
                return false;
            double imu_time = m_group_data.imu_buffer.front().timestamp;
            std::vector<livo::IMUData>().swap(m_sync_pack.imus);
            {
                std::lock_guard<std::mutex> lock(m_group_data.imu_mutex);
                while ((!m_group_data.imu_buffer.empty() && (imu_time < m_sync_pack.cloud_end_time)))
                {
                    m_sync_pack.imus.push_back(m_group_data.imu_buffer.front());
                    m_group_data.imu_buffer.pop_front();
                    imu_time = m_group_data.imu_buffer.front().timestamp;
                }
            }
            {
                std::lock_guard<std::mutex> lock(m_group_data.lidar_mutex);
                m_group_data.lidar_buffer.pop_front();
            }
            m_sync_pack.lidar_end = true;
            m_group_data.lidar_pushed = false;
            return true;
        }
        // 添加图像帧
        double image_start_time = m_group_data.image_buffer.front().first;
        if (image_start_time < m_sync_pack.cloud_start_time)
        {
            std::lock_guard<std::mutex> lock(m_group_data.image_mutex);
            m_group_data.image_buffer.pop_front();
            return false;
        }

        if (m_group_data.last_imu_time < image_start_time)
            return false;

        double imu_time = m_group_data.imu_buffer.front().timestamp;

        std::vector<livo::IMUData>().swap(m_sync_pack.imus);

        m_sync_pack.image_time = image_start_time;
        m_sync_pack.image = m_group_data.image_buffer.front().second;

        {
            std::lock_guard<std::mutex> lock(m_group_data.imu_mutex);
            while ((!m_group_data.imu_buffer.empty() && (imu_time < image_start_time)))
            {
                m_sync_pack.imus.push_back(m_group_data.imu_buffer.front());
                m_group_data.imu_buffer.pop_front();
                imu_time = m_group_data.imu_buffer.front().timestamp;
            }
        }
        {
            std::lock_guard<std::mutex> lock(m_group_data.image_mutex);
            m_group_data.image_buffer.pop_front();
        }
        m_sync_pack.lidar_end = false;
        return true;
    }

    void publishCloud(ros::Publisher &cloud_pub, livo::CloudType::Ptr cloud, std::string &frame_id, double &sec)
    {
        if (cloud_pub.getNumSubscribers() < 1)
            return;
        cloud_pub.publish(pcl2msg(cloud, frame_id, sec));
    }

    void mainCB(const ros::TimerEvent &e)
    {
        if (!syncPackage())
            return;

        // double start_imu_time = 0.0, end_imu_time = 0.0;
        // if (m_sync_pack.imus.size() > 0)
        // {
        //     start_imu_time = m_sync_pack.imus.front().timestamp;
        //     end_imu_time = m_sync_pack.imus.back().timestamp;
        // }
        // ROS_WARN("package type: %d, lidar begin: %.4f, imu begin:%.4f, image off: %.4f, imu end: %.4f, lidar end: %4f, imu_size: %lu",
        //          m_sync_pack.lidar_end,
        //          m_sync_pack.cloud_start_time,
        //          start_imu_time,
        //          m_sync_pack.image_time,
        //          end_imu_time,
        //          m_sync_pack.cloud_end_time,
        //          m_sync_pack.imus.size());
    }

private:
    ros::NodeHandle m_nh;
    ros::Subscriber m_lidar_sub;
    ros::Subscriber m_imu_sub;
    ros::Subscriber m_image_sub;

    ros::Publisher m_body_cloud_pub;
    ros::Publisher m_world_cloud_pub;

    ros::Timer main_loop;
    DataGroup m_group_data;
    Config m_node_config;
    livo::Config m_builder_config;
    livo::SyncPackage m_sync_pack;
    std::shared_ptr<livo::MapBuilder> m_builder;
    tf2_ros::TransformBroadcaster m_br;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "temp_node");
    ROS_INFO("hello test");
    LIVONode node;
    ros::spin();
    return 0;
}