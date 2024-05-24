#include "map_builder/pinhole_camera.h"

int main(int argc, char **argv)
{
    cv::Mat img = cv::imread("/home/zhouzhou/a.jpg");
    cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    std::cout << img.cols << "|" << img.rows << std::endl;
    cv::Mat patch(51, 51, CV_32F);
    V2D px(200.4, 200.4);
    CVUtils::getPatch(img, px, patch, 25, 1);
    std::cout << patch.cols << " | " << patch.rows <<std::endl;
    cv::Mat img_show;
    patch.convertTo(img_show,CV_8U);
    cv::imshow("image", img_show);
    cv::waitKey();
    return 0;
}