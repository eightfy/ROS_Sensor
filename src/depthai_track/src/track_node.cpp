#include <iostream>
#include <vector>
#include <string>
#include <chrono>

// ROS Headers
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"

#include <common_msgs/TrackObjs.h>
#include <common_msgs/TrackObj.h>
#include <ros/package.h>
// DepthAI Headers
#include "depthai/depthai.hpp"

// OpenCV Headers
#include "opencv2/opencv.hpp"

#include <image_transport/image_transport.h>
cv::Scalar getHSVColor(int hash) {
    // OpenCV中H范围是[0, 179], S和V是[0, 255]
    // 我们用一个乘数让相邻的hash值在色盘上跳得远一些
    // 黄金比例共轭数 (约0.618) 是一个很好的选择，可以产生均匀分布的颜色
    const float golden_ratio_conjugate = 0.61803398875;
    float hue = std::fmod(hash * golden_ratio_conjugate * 180.0f, 180.0f);

    // 创建一个1x1的HSV图像
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
    
    // 将HSV转换为BGR
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    
    // 从转换后的图像中获取BGR值
    cv::Vec3b bgr_vec = bgr.at<cv::Vec3b>(0,0);
    return cv::Scalar(bgr_vec[0], bgr_vec[1], bgr_vec[2]);
}

// 追踪状态表：0.new, 1.tracked, 2.lost 3.remove
// 预定义的标签映射
static const std::vector<std::string> labelMap = {"background", "aeroplane", "bicycle",     "bird",  "boat",        "bottle", "bus",
                                                  "car",        "cat",       "chair",       "cow",   "diningtable", "dog",    "horse",
                                                  "motorbike",  "person",    "pottedplant", "sheep", "sofa",        "train",  "tvmonitor"};

// 主函数
int main(int argc, char** argv) {
    using namespace std;
    using namespace std::chrono;
    bool fullFrameTracking = true;
    // 初始化 ROS 节点
    ros::init(argc, argv, "track_node");
    ros::NodeHandle nh;

    // ImageTransport实例，用于发布和订阅图像
    image_transport::ImageTransport it(nh);
    image_transport::Publisher image_pub = it.advertise("yolo_image", 1);

    // ROS Publishers
    ros::Publisher detection_pub = nh.advertise<common_msgs::TrackObjs>("TrackObjs", 10);

    std::string nnPath = ros::package::getPath("depthai_track") + "/models/mobilenet-ssd_openvino_2021.4_8shave.blob";
    ROS_INFO("model: %s", nnPath.c_str());

    // 定义相机坐标系名称
    const std::string camera_frame_id = "oak_rgb_camera_frame";

    dai::Pipeline pipeline;
    // Define sources and outputs
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    auto spatialDetectionNetwork = pipeline.create<dai::node::MobileNetSpatialDetectionNetwork>();
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    auto objectTracker = pipeline.create<dai::node::ObjectTracker>();

    auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
    auto trackerOut = pipeline.create<dai::node::XLinkOut>();

    xoutRgb->setStreamName("preview");
    trackerOut->setStreamName("tracklets");

    // Properties
    camRgb->setPreviewSize(300, 300);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setInterleaved(false);
    camRgb->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);

    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoLeft->setCamera("left");
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setCamera("right");

    // setting node configs
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    // Align depth map to the perspective of RGB camera, on which inference is done
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);
    stereo->setOutputSize(monoLeft->getResolutionWidth(), monoLeft->getResolutionHeight());

    spatialDetectionNetwork->setBlobPath(nnPath);
    spatialDetectionNetwork->setConfidenceThreshold(0.5f);
    spatialDetectionNetwork->input.setBlocking(false);
    spatialDetectionNetwork->setBoundingBoxScaleFactor(0.5);
    spatialDetectionNetwork->setDepthLowerThreshold(100);
    spatialDetectionNetwork->setDepthUpperThreshold(10000);

    objectTracker->setDetectionLabelsToTrack({15});  // track only person
    // possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    objectTracker->setTrackerType(dai::TrackerType::ZERO_TERM_COLOR_HISTOGRAM);
    // take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker->setTrackerIdAssignmentPolicy(dai::TrackerIdAssignmentPolicy::SMALLEST_ID);

    // Linking
    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);

    camRgb->preview.link(spatialDetectionNetwork->input);
    objectTracker->passthroughTrackerFrame.link(xoutRgb->input);
    objectTracker->out.link(trackerOut->input);

    if(fullFrameTracking) {
        camRgb->setPreviewKeepAspectRatio(false);
        camRgb->video.link(objectTracker->inputTrackerFrame);
        objectTracker->inputTrackerFrame.setBlocking(false);
        // do not block the pipeline if it's too slow on full frame
        objectTracker->inputTrackerFrame.setQueueSize(2);
    } else {
        spatialDetectionNetwork->passthrough.link(objectTracker->inputTrackerFrame);
    }

    spatialDetectionNetwork->passthrough.link(objectTracker->inputDetectionFrame);
    spatialDetectionNetwork->out.link(objectTracker->inputDetections);
    stereo->depth.link(spatialDetectionNetwork->inputDepth);

    // 7. 连接设备并开始
    dai::Device device(pipeline);

    // 获取输出队列
    auto preview = device.getOutputQueue("preview", 4, false);
    auto tracklets = device.getOutputQueue("tracklets", 4, false);

    ROS_INFO("Node started, press Ctrl+C to exit...");

    // 8. ROS 主循环
    while (ros::ok()) {
        auto imgFrame = preview->get<dai::ImgFrame>();
        auto track = tracklets->get<dai::Tracklets>();
        cv::Mat frame = imgFrame->getCvFrame();


        // 获取当前时间戳
        ros::Time currentTime = ros::Time::now();

        // 处理和发布自定义的3D检测结果
        common_msgs::TrackObjs result_box_msg;
        result_box_msg.header.stamp = currentTime;
        result_box_msg.header.frame_id = camera_frame_id;

        auto trackletsData = track->tracklets;
        result_box_msg.time_t = static_cast<uint16_t>(currentTime.sec);

        for (auto& t : trackletsData) {
            auto roi = t.roi.denormalize(frame.cols, frame.rows);
            int x1 = roi.topLeft().x;
            int y1 = roi.topLeft().y;
            int x2 = roi.bottomRight().x;
            int y2 = roi.bottomRight().y;
            uint32_t labelIndex = t.label;
            std::string labelStr = to_string(labelIndex);
            if(labelIndex < labelMap.size()) {
                labelStr = labelMap[labelIndex];
            }

            cv::putText(frame, labelStr, cv::Point(x1 + 10, y1 + 20), cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);

            std::stringstream idStr;
            idStr << "ID: " << t.id;
            cv::putText(frame, idStr.str(), cv::Point(x1 + 10, y1 + 35), cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
            std::stringstream statusStr;
            statusStr << "Status: " << t.status;
            cv::putText(frame, statusStr.str(), cv::Point(x1 + 10, y1 + 50), cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);

            std::stringstream depthX;
            depthX << "X: " << (int)t.spatialCoordinates.x << " mm";
            cv::putText(frame, depthX.str(), cv::Point(x1 + 10, y1 + 65), cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
            std::stringstream depthY;
            depthY << "Y: " << (int)t.spatialCoordinates.y << " mm";
            cv::putText(frame, depthY.str(), cv::Point(x1 + 10, y1 + 80), cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
            std::stringstream depthZ;
            depthZ << "Z: " << (int)t.spatialCoordinates.z << " mm";
            cv::putText(frame, depthZ.str(), cv::Point(x1 + 10, y1 + 95), cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
            uint32_t hash = t.id*t.id*t.id*t.id*t.id ;
            cv::Scalar color = getHSVColor(hash);
            cv::rectangle(frame, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), color, 3);
            
            // ROS_INFO("t.status: %d",t.status);
            if (true) {
                common_msgs::TrackObj trackObj;
                uint32_t labelIndex = t.label;
                labelStr = labelMap[labelIndex];
                // 填充 TrackObj 消息
                trackObj.id   = t.id;
                trackObj.type = t.label;
                
                // 设置3D坐标 (毫米)
                trackObj.x = t.spatialCoordinates.x;
                trackObj.y = t.spatialCoordinates.y;
                trackObj.z = t.spatialCoordinates.z;
                trackObj.xmin = x1;
                trackObj.ymin = y1;
                trackObj.xmax = x2;
                trackObj.ymax = y2;
                // 追踪状态表：0.new, 1.tracked, 2.lost 3.remove
                trackObj.status = int(t.status);
                // 将填充好的 TrackObj 添加到 ReasultBox 消息的数组中
                result_box_msg.box_msgs.push_back(trackObj);
                result_box_msg.box_msgs.push_back(trackObj);
                ROS_INFO("Detected: %s, rectangle: (%.2f, %.2f), (%.2f, %.2f)",labelStr.c_str(),trackObj.xmin,trackObj.ymin,trackObj.xmax,trackObj.ymax);
            }
        }
        // 将绘制好的OpenCV Mat图像转换回ROS图像消息
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
        msg->header.stamp = currentTime; // 使用与检测结果相同的时间戳
        msg->header.frame_id = camera_frame_id; // 使用相同的坐标系

        // 发布带有识别框的图像
        image_pub.publish(msg);
        // 在筛选后更新物体数量
        result_box_msg.obj_num = result_box_msg.box_msgs.size();

        // if (result_box_msg.obj_num > 0) {
            detection_pub.publish(result_box_msg);
        // }

        ros::spinOnce();
    }

    return 0;
}
