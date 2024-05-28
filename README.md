# FASTLIVO
代码重构版本

## 主要工作
1. 重构了[FAST-LIVO](https://github.com/hku-mars/FAST-LIVO) 的代码，剔除了原始REPO中不必要的环境依赖；
2. 暂时支持官方数据集的传感器(livox avia),感兴趣的同学可以继续扩展;

## 环境说明
```text
系统版本: ubuntu20.04
机器人操作系统: ros1-noetic
```
## 编译依赖
1. livox_ros_driver
2. pcl (1.10)
3. sophus (1.22.10)
4. eigen
5. opencv (4.2)

### 1.安装 LIVOX-SDK
```shell
git clone https://github.com/Livox-SDK/Livox-SDK.git
cd Livox-SDK
cd build && cmake ..
make
sudo make install
```

### 2.安装 livox_ros_driver
```shell
mkdir -p ws_livox/src
git clone https://github.com/Livox-SDK/livox_ros_driver.git ws_livox/src
cd ws_livox
catkin_make
```
### 3. 安装Sophus
```
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout 1.22.10
mkdir build && cd build
cmake .. -DSOPHUS_USE_BASIC_LOGGING=ON
make
sudo make install
```
**新的Sophus依赖fmt，可以在CMakeLists.txt中添加add_compile_definitions(SOPHUS_USE_BASIC_LOGGING)去除，否则会报错**

**opencv和pcl使用ros完整版中自带的即可，无需额外安装**

## DEMO 数据
[dataset](https://connecthkuhk-my.sharepoint.com/personal/zhengcr_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzhengcr%5Fconnect%5Fhku%5Fhk%2FDocuments%2FFAST%2DLIVO%2DDatasets&ga=1)

## 启动脚本
```shell
roslaunch fastlivo livo.launch
```

## 特别感谢
[FAST-LIVO](https://github.com/hku-mars/FAST-LIVO)

