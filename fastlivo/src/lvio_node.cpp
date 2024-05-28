#include <ros/ros.h>
#include <queue>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <tf2_ros/transform_broadcaster.h>

#include "utils.h"
#include "map_builder/commons.h"
#include "map_builder/map_builder.h"

struct NodeConfig
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

    std::deque<IMUData> imu_buffer;
    std::deque<std::pair<double, CloudType::Ptr>> lidar_buffer;
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
        m_kf = std::make_shared<IESKF>(m_builder_config.ieskf_max_iter);
        m_builder = std::make_shared<MapBuilder>(m_builder_config, m_kf);
        main_loop = m_nh.createTimer(ros::Duration(0.05), &LIVONode::mainCB, this);
    }

    void loadCofig()
    {
        m_nh.param<std::string>("imu_topic", m_node_config.imu_topic, "/livox/imu");
        m_nh.param<std::string>("lidar_topic", m_node_config.lidar_topic, "/livox/lidar");
        m_nh.param<std::string>("image_topic", m_node_config.image_topic, "/left_camera/image");
        m_nh.param<std::string>("map_frame", m_node_config.map_frame, "lidar");
        m_nh.param<std::string>("body_frame", m_node_config.body_frame, "body");

        m_nh.param<int>("lidar_filter_num", m_builder_config.lidar_filter_num, 2);
        m_nh.param<double>("lidar_blind", m_builder_config.lidar_blind, 5.0);
        m_nh.param<double>("scan_resolution", m_builder_config.scan_resolution, 0.15);
        m_nh.param<double>("map_resolution", m_builder_config.map_resolution, 0.3);

        m_nh.param<double>("cube_len", m_builder_config.cube_len, 300);
        m_nh.param<double>("det_range", m_builder_config.det_range, 60);
        m_nh.param<double>("move_thresh", m_builder_config.move_thresh, 1.5);

        m_nh.param<bool>("image_enable", m_builder_config.image_enable, true);

        m_nh.param<double>("na", m_builder_config.na, 0.01);
        m_nh.param<double>("ng", m_builder_config.ng, 0.01);
        m_nh.param<double>("nba", m_builder_config.nba, 0.0001);
        m_nh.param<double>("nbg", m_builder_config.nbg, 0.0001);

        m_nh.param<int>("imu_init_num", m_builder_config.imu_init_num, 20);
        m_nh.param<int>("near_search_num", m_builder_config.near_search_num, 5);
        m_nh.param<int>("ieskf_max_iter", m_builder_config.ieskf_max_iter, 10);

        m_nh.param<bool>("gravity_align", m_builder_config.gravity_align, true);
        m_nh.param<bool>("esti_li", m_builder_config.esti_li, false);
        m_nh.param<bool>("esti_ci", m_builder_config.esti_ci, false);

        m_nh.param<double>("cam_width", m_builder_config.cam_width, 640);
        m_nh.param<double>("cam_height", m_builder_config.cam_height, 512);
        m_nh.param<double>("cam_fx", m_builder_config.cam_fx, 431.795259219);
        m_nh.param<double>("cam_fy", m_builder_config.cam_fy, 431.550090267);
        m_nh.param<double>("cam_cx", m_builder_config.cam_cx, 310.833037316);
        m_nh.param<double>("cam_cy", m_builder_config.cam_cy, 266.985989326);

        Vec<double> cam_d;
        m_nh.param<Vec<double>>("cam_d", m_builder_config.cam_d, {-0.0944205499243979, 0.0946727677776504, -0.00807970960613932, 8.07461209775283e-05, 0.0});
        m_nh.param<int>("half_patch_size", m_builder_config.half_patch_size, 4);
        m_nh.param<int>("grid_size", m_builder_config.grid_size, 32);
        m_nh.param<int>("skip_first_image_num", m_builder_config.skip_first_image_num, 5);
        m_nh.param<double>("selector_scan_resolution", m_builder_config.selector_scan_resolution, 0.2);
        m_nh.param<double>("selector_voxel_size", m_builder_config.selector_voxel_size, 0.5);
        m_nh.param<double>("pixel_sq_dist_thresh", m_builder_config.pixel_sq_dist_thresh, 300.0);
        m_nh.param<double>("lidar_cov_inv", m_builder_config.lidar_cov_inv, 1000.0);
        m_nh.param<double>("image_cov_inv", m_builder_config.image_cov_inv, 0.01);

        Vec<double> r_il, t_il, r_cl, t_cl;
        m_nh.param<Vec<double>>("r_il", r_il, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        m_nh.param<Vec<double>>("t_il", t_il, {0.0, 0.0, 0.0});
        m_nh.param<Vec<double>>("r_cl", r_cl, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        m_nh.param<Vec<double>>("t_cl", t_cl, {0.0, 0.0, 0.0});

        m_builder_config.r_il << r_il[0], r_il[1], r_il[2], r_il[3], r_il[4], r_il[5], r_il[6], r_il[7], r_il[8];
        m_builder_config.t_il << t_il[0], t_il[1], t_il[2];
        m_builder_config.r_cl << r_cl[0], r_cl[1], r_cl[2], r_cl[3], r_cl[4], r_cl[5], r_cl[6], r_cl[7], r_cl[8];
        m_builder_config.t_cl << t_cl[0], t_cl[1], t_cl[2];
        m_builder_config.r_cl = Eigen::Quaterniond(m_builder_config.r_cl).normalized().toRotationMatrix();
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

        m_world_cloud_color_pub = m_nh.advertise<sensor_msgs::PointCloud2>("world_cloud_color", 1000);
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
        m_group_data.imu_buffer.emplace_back(V3D(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z) * 10.0,
                                             V3D(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                                             timestamp);
    }

    void lidarCB(const livox_ros_driver::CustomMsg::ConstPtr msg)
    {
        CloudType::Ptr cloud(new CloudType);
        livoxAvia2pcl(msg, cloud, m_builder_config.lidar_filter_num, m_builder_config.lidar_blind);
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
            std::sort(m_sync_pack.cloud->points.begin(), m_sync_pack.cloud->points.end(), [](PointType &p1, PointType &p2)
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
            Vec<IMUData>().swap(m_sync_pack.imus);
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

        Vec<IMUData>().swap(m_sync_pack.imus);

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

    void publishCloud(ros::Publisher &cloud_pub, CloudType::Ptr cloud, std::string &frame_id, double &sec)
    {
        if (cloud_pub.getNumSubscribers() < 1)
            return;
        cloud_pub.publish(pcl2msg(cloud, frame_id, sec));
    }

    void publishCloudColor(ros::Publisher &cloud_pub, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, double &sec)
    {
        if (cloud_pub.getNumSubscribers() < 1)
            return;
        if (cloud->size() <= 0)
            return;
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        if (sec < 0)
            msg.header.stamp = ros::Time().now();
        else
            msg.header.stamp = ros::Time().fromSec(sec);
        msg.header.frame_id = m_node_config.map_frame;
        cloud_pub.publish(msg);
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
        else
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_color = m_builder->image_processor()->getLastestColoredCloud();
            publishCloudColor(m_world_cloud_color_pub, cloud_color, m_sync_pack.image_time);
        }
    }

private:
    ros::NodeHandle m_nh;
    ros::Subscriber m_lidar_sub;
    ros::Subscriber m_imu_sub;
    ros::Subscriber m_image_sub;

    ros::Publisher m_body_cloud_pub;
    ros::Publisher m_world_cloud_pub;
    ros::Publisher m_world_cloud_color_pub;

    ros::Timer main_loop;
    DataGroup m_group_data;
    NodeConfig m_node_config;
    Config m_builder_config;
    SyncPackage m_sync_pack;
    std::shared_ptr<IESKF> m_kf;
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