#ifndef _ARMORDETECTOR_HPP_
#define _ARMORDETECTOR_HPP_

#include "Classifier.hpp"

class ArmorDetector
{
private:
    bool isLight(const Light & possible_light);
    bool containLight(
        const Light & light_1, const Light & light_2, const std::vector<Light> & lights);
    ArmorType isArmor(const Light & light_1, const Light & light_2);

    std::vector<Light> lights_;
    std::vector<Armor> armors_;
    // 卡尔曼滤波器实例
    cv::KalmanFilter kalmanFilter;
    // 初始化卡尔曼滤波器
    void initKalmanFilter();
    // 更新卡尔曼滤波器
    cv::Point2f updateKalmanFilter(const cv::Point2f &measurement);
    cv::Mat prevGrayImg;          // 上一帧的灰度图像
    std::vector<cv::Point2f> prevPoints;  // 上一帧检测到的灯条中心点
public:
    int binary_thres;
    int detect_color;
    LightParams l;
    ArmorParams a;

    std::shared_ptr<NumberClassifier> classifier;
    // Debug
    cv::Mat binary_img;
    void drawResults(cv::Mat & img);

    cv::Mat preprocessImage(const cv::Mat & rbg_img);
    std::vector<Light> findLights(const cv::Mat &rbg_img, const cv::Mat &binary_img);
    std::vector<Armor> matchLights(const std::vector<Light> &lights);

    std::vector<Armor> work(const cv::Mat & input);
    ArmorDetector();
    ArmorDetector(const int & bin_thres, const int & color, const LightParams & l, const ArmorParams & a);
    ~ArmorDetector();
};


#endif // !_ARMORDETECTOR_HPP_
