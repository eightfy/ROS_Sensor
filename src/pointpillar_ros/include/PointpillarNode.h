// #include <ros/ros.h>

// #include "cuda_runtime.h"

// #include "./params.h"
// #include "./pointpillar.h"

// #include <iostream>
// #include <sstream>
// #include <fstream>
// #define checkCudaErrors(status)                                   \
// {                                                                 \
//   if (status != 0)                                                \
//   {                                                               \
//     std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
//               << " at line " << __LINE__                          \
//               << " in file " << __FILE__                          \
//               << " error status: " << status                      \
//               << std::endl;                                       \
//               abort();                                            \
//     }                                                             \
// }

// class PointpillarNode {
// public:
//     PointpillarNode(ros::NodeHandle& nh);
//     ~PointpillarNode();
//     void run();

// private:
//     // ROS节点句柄
//     ros::NodeHandle nh_;
//     PointPillar pointpillar;
//     void Getinfo();
//     int runModelDetection();
// };