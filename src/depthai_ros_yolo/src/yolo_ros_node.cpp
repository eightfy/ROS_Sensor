#include <iostream>
#include <vector>
#include <string>
#include <chrono>

// ROS Headers
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"

#include <common_msgs/ReasultBox.h>
#include <common_msgs/FusionObj.h>
#include <ros/package.h>
// DepthAI Headers
#include "depthai/depthai.hpp"

// OpenCV Headers
#include "opencv2/opencv.hpp"

#include <image_transport/image_transport.h>

// 预定义的标签映射
static const std::vector<std::string> labelMap = {
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

// 主函数
int main(int argc, char** argv) {
    // 初始化 ROS 节点
    ros::init(argc, argv, "yolo_ros_node");
    ros::NodeHandle nh;

    // 创建一个ImageTransport实例，用于发布和订阅图像
    image_transport::ImageTransport it(nh);
    // 创建一个图像发布者，话题为 "yolo_image"，队列大小为1
    image_transport::Publisher image_pub = it.advertise("yolo_image", 1);

    // 定义 ROS Publishers
    // 将 Publisher 的类型更改为自定义消息类型 ReasultBox
    ros::Publisher detection_pub = nh.advertise<common_msgs::ReasultBox>("detections", 10);

    // 模型路径
    std::string nnPath = ros::package::getPath("depthai_ros_yolo") + "/models/yolov11n_openvino_2021.4_5shave.blob";
    ROS_INFO("model: %s", nnPath.c_str());

    // 定义相机坐标系名称
    const std::string camera_frame_id = "oak_rgb_camera_frame";

    dai::Pipeline pipeline;
    // 定义源和节点
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    auto spatialDetectionNetwork = pipeline.create<dai::node::YoloSpatialDetectionNetwork>();

    // 定义输出
    auto xoutNN = pipeline.create<dai::node::XLinkOut>();
    xoutNN->setStreamName("detections");

    auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
    xoutRgb->setStreamName("rgb");

    // 从模型获取输入尺寸 
    auto blob = dai::OpenVINO::Blob(nnPath);
    auto networkInputs = blob.networkInputs;
    auto firstInput = networkInputs.begin();
    int nnWidth = firstInput->second.dims[0];
    int nnHeight = firstInput->second.dims[1];

    // 配置节点
    camRgb->setPreviewSize(nnWidth, nnHeight);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setInterleaved(false);
    camRgb->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
    camRgb->setFps(30);

    // 黑白相机设置
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_1200_P);
    monoLeft->setCamera("left");
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_1200_P);
    monoRight->setCamera("right");

    // 为左右相机分别创建 ImageManip 节点用于缩放
    // 确保输出的图像是灰度图 (RAW8)
    auto manipLeft = pipeline.create<dai::node::ImageManip>();
    manipLeft->initialConfig.setResize(nnWidth, nnHeight); 
    manipLeft->initialConfig.setFrameType(dai::RawImgFrame::Type::RAW8);
    
    auto manipRight = pipeline.create<dai::node::ImageManip>();
    manipRight->initialConfig.setResize(nnWidth, nnHeight);
    manipRight->initialConfig.setFrameType(dai::RawImgFrame::Type::RAW8);

    // 立体匹配设置
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A); // 将深度图与RGB相机对齐
    stereo->setOutputSize(monoLeft->getResolutionWidth(), monoLeft->getResolutionHeight());
    stereo->setLeftRightCheck(true);
    stereo->setExtendedDisparity(false);
    stereo->setSubpixel(true);

    // Yolo空间检测网络设置
    spatialDetectionNetwork->setBlobPath(nnPath);
    spatialDetectionNetwork->setConfidenceThreshold(0.3f);
    spatialDetectionNetwork->input.setBlocking(false);
    spatialDetectionNetwork->setBoundingBoxScaleFactor(0.5);
    spatialDetectionNetwork->setDepthLowerThreshold(100);   // 10cm
    spatialDetectionNetwork->setDepthUpperThreshold(40000); // 40m

    // YOLO 特定参数 (需要根据你的模型进行调整)
    spatialDetectionNetwork->setNumClasses(10); // COCO 数据集有 80 个类别
    spatialDetectionNetwork->setCoordinateSize(4);
    spatialDetectionNetwork->setIouThreshold(0.5f);

    // 链接节点
    monoLeft->out.link(manipLeft->inputImage);
    monoRight->out.link(manipRight->inputImage);
    manipLeft->out.link(stereo->left);
    manipRight->out.link(stereo->right);
    camRgb->preview.link(xoutRgb->input);

    camRgb->preview.link(spatialDetectionNetwork->input);
    spatialDetectionNetwork->out.link(xoutNN->input);
    stereo->depth.link(spatialDetectionNetwork->inputDepth);

    // 7. 连接设备并开始
    dai::Device device(pipeline);

    // 获取输出队列
    auto detectionNNQueue = device.getOutputQueue("detections", 4, false);
    auto qRgb = device.getOutputQueue("rgb", 4, false);

    ROS_INFO("Node started, press Ctrl+C to exit...");

    // 8. ROS 主循环
    while (ros::ok()) {
        auto inDet = detectionNNQueue->get<dai::SpatialImgDetections>();
        auto inRgb = qRgb->get<dai::ImgFrame>();
        cv::Mat frame = inRgb->getCvFrame();


        // 获取当前时间戳
        ros::Time currentTime = ros::Time::now();

        // 处理和发布自定义的3D检测结果
        common_msgs::ReasultBox result_box_msg;
        result_box_msg.header.stamp = currentTime;
        result_box_msg.header.frame_id = camera_frame_id;

        auto detections = inDet->detections;
        
        // 注意: ReasultBox.msg 中的 time_t 是 uint16，而 ROS 时间戳是 64 位。
        // 这里进行强制类型转换，可能会导致时间戳信息丢失或回绕。
        result_box_msg.time_t = static_cast<uint16_t>(currentTime.sec);

        for (const auto& detection : detections) {
            // 只处理 "person" (标签 0) 和 "car" (标签 2)，并且 z 距离小于等于 10 米
            float z_meters = detection.spatialCoordinates.z / 1000.0;
            int x1 = detection.xmin * frame.cols;
            int y1 = detection.ymin * frame.rows;
            int x2 = detection.xmax * frame.cols;
            int y2 = detection.ymax * frame.rows;
            std::string labelStr = labelMap[detection.label];
            float confidence = detection.confidence;
            std::string text = labelStr + ": " + std::to_string(static_cast<int>(confidence * 100)) + "%, Z: " + cv::format("%.2f", z_meters) + "m";
            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, text, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

            if ((detection.label == 0 || detection.label == 0) && z_meters <= 10.0) {
                common_msgs::FusionObj fusion_obj;
                
                // 填充 FusionObj 消息
                // 假设 id 和 type 都使用检测到的标签索引
                fusion_obj.id   = detection.label;
                fusion_obj.type = detection.label;
                
                // 设置3D坐标 (从毫米转换为米)
                fusion_obj.x = detection.spatialCoordinates.x / 1000.0;
                fusion_obj.y = detection.spatialCoordinates.y / 1000.0;
                fusion_obj.z = z_meters;
                
                // 将填充好的 FusionObj 添加到 ReasultBox 消息的数组中
                result_box_msg.box_msgs.push_back(fusion_obj);
                result_box_msg.box_msgs.push_back(fusion_obj);
                ROS_INFO("Detected: %s, Confidence: %.2f, Coords(m): [x: %.2f, y: %.2f, z: %.2f]",labelMap[detection.label].c_str(),detection.confidence,fusion_obj.x,fusion_obj.y,fusion_obj.z);
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
