#include <ros/ros.h>
#include <queue>
#include <sensor_msgs/Imu.h>
#include <tf2_ros/transform_broadcaster.h>
#include "utils.h"
#include "map_builder/map_builder.h"

struct NodeConfig
{
    std::string imu_topic = "/livox/imu";
    std::string lidar_topic = "/livox/lidar";
    std::string map_frame = "lidar";
    std::string body_frame = "body";
};
struct DataGroup
{
    bool lidar_pushed = false;
    std::mutex imu_mutex;
    std::mutex lidar_mutex;
    double last_imu_time;
    std::deque<IMUData> imu_buffer;
    std::deque<std::pair<double, CloudType::Ptr>> lidar_buffer;
    double last_lidar_time;
};

class LIVONode
{
public:
    LIVONode() : m_nh("~")
    {
        loadCofig();
        initSubsriber();
        initPublisher();

        m_kf = std::make_shared<IESKF>();
        m_builder = std::make_shared<MapBuilder>(m_builder_config, m_kf);
        main_loop = m_nh.createTimer(ros::Duration(0.05), &LIVONode::mainCB, this);
    }
    
    void loadCofig()
    {
    }
    
    void initSubsriber()
    {
        m_lidar_sub = m_nh.subscribe(m_node_config.lidar_topic, 10000, &LIVONode::lidarCB, this);
        m_imu_sub = m_nh.subscribe(m_node_config.imu_topic, 10000, &LIVONode::imuCB, this);
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

    // void lidarCB(const livox_ros_driver2::CustomMsg::ConstPtr msg)
    void lidarCB(const livox_ros_driver::CustomMsg::ConstPtr msg)
    {
        CloudType::Ptr cloud(new CloudType);
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
    
    bool syncPackage()
    {
        if (m_group_data.imu_buffer.empty() || m_group_data.lidar_buffer.empty())
            return false;
        // 同步点云数据
        if (!m_group_data.lidar_pushed)
        {
            m_sync_pack.cloud = m_group_data.lidar_buffer.front().second;
            std::sort(m_sync_pack.cloud->points.begin(), m_sync_pack.cloud->points.end(), [](PointType &p1, PointType &p2)
                      { return p1.curvature < p2.curvature; });
            m_sync_pack.cloud_start_time = m_group_data.lidar_buffer.front().first;
            m_sync_pack.cloud_end_time = m_sync_pack.cloud_start_time + m_sync_pack.cloud->points.back().curvature / double(1000.0);
            m_group_data.lidar_pushed = true;
        }
        // 等待IMU的数据
        if (m_group_data.last_imu_time < m_sync_pack.cloud_end_time)
            return false;

        m_sync_pack.imus.clear();

        // 同步IMU的数据
        // IMU的最后一帧数据的时间小于点云最后一个点的时间
        while (!m_group_data.imu_buffer.empty() && (m_group_data.imu_buffer.front().timestamp < m_sync_pack.cloud_end_time))
        {
            m_sync_pack.imus.push_back(m_group_data.imu_buffer.front());
            m_group_data.imu_buffer.pop_front();
        }
        m_group_data.lidar_buffer.pop_front();
        m_group_data.lidar_pushed = false;
        m_sync_pack.lidar_end = true;
        return true;
    }

    void publishCloud(ros::Publisher &cloud_pub, CloudType::Ptr cloud, std::string &frame_id, double &sec)
    {
        if (cloud_pub.getNumSubscribers() < 1)
            return;
        cloud_pub.publish(pcl2msg(cloud, frame_id, sec));
    }
    
    void mainCB(const ros::TimerEvent &e)
    {
        if (!syncPackage())
            return;

        m_builder->process(m_sync_pack);

        if (m_builder->status() != BuilderStatus::MAPPING)
            return;
        if (m_sync_pack.lidar_end)
        {
            m_br.sendTransform(eigen2Transform(m_kf->x().r_wi, m_kf->x().t_wi, m_node_config.map_frame, m_node_config.body_frame, m_sync_pack.cloud_end_time));
            CloudType::Ptr cloud_body = LidarProcessor::transformCloud(m_sync_pack.cloud, m_kf->x().r_il, m_kf->x().t_il);
            publishCloud(m_body_cloud_pub, cloud_body, m_node_config.body_frame, m_sync_pack.cloud_end_time);
            M3D r_wl = m_kf->x().r_wi * m_kf->x().r_il;
            V3D t_wl = m_kf->x().r_wi * m_kf->x().t_il + m_kf->x().t_wi;
            CloudType::Ptr cloud_world = LidarProcessor::transformCloud(m_sync_pack.cloud, r_wl, t_wl);
            publishCloud(m_world_cloud_pub, cloud_world, m_node_config.map_frame, m_sync_pack.cloud_end_time);
        }
    }

private:
    ros::NodeHandle m_nh;
    ros::Subscriber m_lidar_sub;
    ros::Subscriber m_imu_sub;

    ros::Publisher m_body_cloud_pub;
    ros::Publisher m_world_cloud_pub;

    ros::Timer main_loop;

    std::shared_ptr<IESKF> m_kf;
    DataGroup m_group_data;
    NodeConfig m_node_config;
    Config m_builder_config;
    SyncPackage m_sync_pack;
    std::shared_ptr<MapBuilder> m_builder;
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