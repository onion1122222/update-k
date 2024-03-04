#include "ArmorDetector.hpp"

std::vector<Armor> ArmorDetector::work(const cv::Mat &input)
{
    binary_img = preprocessImage(input);
    lights_ = findLights(input, binary_img);
    armors_ = matchLights(lights_);

    if (!armors_.empty()){
        classifier->extractNumbers(input, armors_);
        classifier->classify(armors_);
    }

    return armors_;
}

ArmorDetector::ArmorDetector() {
    // 初始化卡尔曼滤波器
    initKalmanFilter();
}

cv::Mat ArmorDetector::preprocessImage(const cv::Mat & rgb_img)
{
    cv::Mat gray_img;
    cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);

    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);

    return binary_img;
}

std::vector<Light> ArmorDetector::findLights(const cv::Mat &rbg_img, const cv::Mat &binary_img) {
    using std::vector;
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    vector<Light> lights;

    for (const auto &contour : contours) {
        if (contour.size() < 5) continue;

        auto r_rect = cv::minAreaRect(contour);
        auto light = Light(r_rect);

        if (isLight(light)) {
            auto rect = light.boundingRect();
            if (  // Avoid assertion failed
                0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols && 0 <= rect.y &&
                0 <= rect.height && rect.y + rect.height <= rbg_img.rows) {
                // int sum_r = 0, sum_b = 0;
                // auto roi = rbg_img(rect);
                // // Iterate through the ROI
                // for (int i = 0; i < roi.rows; i++) {
                //     for (int j = 0; j < roi.cols; j++) {
                //         if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y), false) >= 0) {
                //             // if point is inside contour bgr
                //             sum_b += roi.at<cv::Vec3b>(i, j)[0];
                //             sum_r += roi.at<cv::Vec3b>(i, j)[2];
                //         }
                //     }
                // }
                // // Sum of red pixels > sum of blue pixels ?
                // light.color = sum_r > sum_b ? RED : BLUE;
                // // cv::Point2f predictedPos = updateKalmanFilter(cv::Point2f(light.center.x, light.center.y));
                // lights.emplace_back(light);
                auto roi = rbg_img(rect);

                // 转换颜色空间到HSV
                cv::Mat hsv_roi;
                cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);

                // 设置HSV颜色阈值，这里可以根据实际情况进行调整
                cv::Scalar lower_red_hsv(0, 150, 150);
                cv::Scalar upper_red_hsv(10, 255, 255);
                cv::Scalar lower_blue_hsv(100, 150, 150);
                cv::Scalar upper_blue_hsv(140, 255, 255);

                // 在HSV空间中判断颜色
                cv::Mat red_mask, blue_mask;
                cv::inRange(hsv_roi, lower_red_hsv, upper_red_hsv, red_mask);
                cv::inRange(hsv_roi, lower_blue_hsv, upper_blue_hsv, blue_mask);

                // 使用形态学操作改善颜色区域
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
                cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, kernel);
                cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_CLOSE, kernel);

                int red_count = cv::countNonZero(red_mask);
                int blue_count = cv::countNonZero(blue_mask);

                // 判断颜色
                light.color = red_count > blue_count ? RED : BLUE;
                lights.emplace_back(light);
            }
        }
    }

    // // 使用光流法估计灯条的运动
    // if (!prevPoints.empty() && !prevGrayImg.empty()) {
    //     cv::Mat currGrayImg;
    //     cv::cvtColor(rbg_img, currGrayImg, cv::COLOR_RGB2GRAY);

    //     std::vector<cv::Point2f> currPoints;
    //     for (const auto &light : lights) {
    //         currPoints.push_back(light.center);
    //     }

    //     std::vector<cv::Point2f> trackedPoints;
    //     std::vector<uchar> status;
    //     std::vector<float> err;
    //     cv::calcOpticalFlowPyrLK(prevGrayImg, currGrayImg, prevPoints, trackedPoints, status, err);

    //     // 更新灯条位置
    //     for (size_t i = 0; i < lights.size(); ++i) {
    //         if (status[i]) {
    //             lights[i].center = trackedPoints[i];
    //         }
    //     }

    //     prevGrayImg = currGrayImg.clone();
    //     prevPoints = currPoints;
    // } else {
    //     // 如果没有上一帧的信息，直接使用当前帧的灰度图像
    //     cv::cvtColor(rbg_img, prevGrayImg, cv::COLOR_RGB2GRAY);
    //     prevPoints.clear();
    //     for (const auto &light : lights) {
    //         prevPoints.push_back(light.center);
    //     }
    // }

    return lights;
}


bool ArmorDetector::isLight(const Light & light)
{
  // The ratio of light (short side / long side)
  float ratio = light.width / light.length;
  bool ratio_ok = l.min_ratio < ratio && ratio < l.max_ratio;

  bool angle_ok = light.tilt_angle < l.max_angle;

  bool is_light = ratio_ok && angle_ok;

  return is_light;
}

std::vector<Armor> ArmorDetector::matchLights(const std::vector<Light> & lights)
{
  std::vector<Armor> armors;

  // Loop all the pairing of lights
  for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
    for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
      if (light_1->color != detect_color || light_2->color != detect_color) continue;

      if (containLight(*light_1, *light_2, lights)) {
        continue;
      }

      auto type = isArmor(*light_1, *light_2);
      if (type != ArmorType::INVALID) {
        auto armor = Armor(*light_1, *light_2);
        armor.type = type;
        armors.emplace_back(armor);
      }
    }
  }

  return armors;
}

// Check if there is another light in the boundingRect formed by the 2 lights
bool ArmorDetector::containLight(
  const Light & light_1, const Light & light_2, const std::vector<Light> & lights)
{
  auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
  auto bounding_rect = cv::boundingRect(points);

  for (const auto & test_light : lights) {
    if (test_light.center == light_1.center || test_light.center == light_2.center) continue;

    if (
      bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
      bounding_rect.contains(test_light.center)) {
      return true;
    }
  }

  return false;
}

ArmorType ArmorDetector::isArmor(const Light & light_1, const Light & light_2)
{
  // Ratio of the length of 2 lights (short side / long side)
  float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                             : light_2.length / light_1.length;
  bool light_ratio_ok = light_length_ratio > a.min_light_ratio;

  // Distance between the center of 2 lights (unit : light length)
  float avg_light_length = (light_1.length + light_2.length) / 2;
  float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
  bool center_distance_ok = (a.min_small_center_distance <= center_distance &&
                             center_distance < a.max_small_center_distance) ||
                            (a.min_large_center_distance <= center_distance &&
                             center_distance < a.max_large_center_distance);

  // Angle of light center connection
  cv::Point2f diff = light_1.center - light_2.center;
  float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
  bool angle_ok = angle < a.max_angle;

  bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

  // Judge armor type
  ArmorType type;
  if (is_armor) {
    type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
  } else {
    type = ArmorType::INVALID;
  }

  return type;
}


void ArmorDetector::drawResults(cv::Mat & img)
{
  // Draw Lights
  for (const auto & light : lights_) {
    cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
    cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
    auto line_color = light.color == RED ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
    cv::line(img, light.top, light.bottom, line_color, 1);
  }

  // Draw armors
  for (const auto & armor : armors_) {
    cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
    cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
  }

  // Show numbers and confidence
  for (const auto & armor : armors_) {
    cv::putText(
      img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(0, 255, 255), 2);
  }
}

ArmorDetector::ArmorDetector(
    const int &bin_thres, const int &color, const LightParams &l, const ArmorParams &a)
    : binary_thres(bin_thres), detect_color(color), l(l), a(a)
{
}

ArmorDetector::~ArmorDetector()
{
}
void ArmorDetector::initKalmanFilter() {
    int stateSize = 4; // [x, y, deltaX, deltaY]
    int measSize = 2;  // [x, y]
    int contrSize = 0; // No control input

    kalmanFilter.init(stateSize, measSize, contrSize, CV_32F);
   
kalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

    cv::setIdentity(kalmanFilter.measurementMatrix);
    cv::setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kalmanFilter.errorCovPost, cv::Scalar::all(1));
}

cv::Point2f ArmorDetector::updateKalmanFilter(const cv::Point2f &measurement) {
  if (kalmanFilter.transitionMatrix.empty()) {
    
    return measurement; // Return the original measurement if the filter is not initialized
}


    cv::Mat prediction = kalmanFilter.predict();
    if (prediction.empty()) {
        std::cerr << "Warning: Kalman filter prediction is empty." << std::endl;
        return measurement;
    }
    cv::Point2f predictPt(prediction.at<float>(0), prediction.at<float>(1));

    cv::Mat measurementMat(2, 1, CV_32F);
    measurementMat.at<float>(0) = measurement.x;
    measurementMat.at<float>(1) = measurement.y;

    kalmanFilter.correct(measurementMat);

    std::cout << "Predicted position: " << predictPt << ", Updated position: " << measurement << std::endl;

    return predictPt;
}
