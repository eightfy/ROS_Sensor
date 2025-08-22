#include <iostream>
#include <sstream>
#include <fstream>
#include <ros/ros.h>
#include "cuda_runtime.h"
#include "common.h"
#include <pcl_conversions/pcl_conversions.h>// PCL与ROS数据转换
#include <csignal>
#include <sensor_msgs/PointCloud2.h>
#include "./params.h"
#include "./pointpillar.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}
// shared_ptr<Preprocess> p_pre(new Preprocess());
// PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
std::string Data_File = "/home/jean/jean/percp/src/pointpillar_ros/data/";
std::string Save_Dir = "eval/kitti/object/pred_velo/";
std::string Model_File = "/home/jean/git/Lidar_AI_Solution/CUDA-PointPillars/model/pointpillar.onnx";
ros::Publisher pointcloud_pub_;
cudaEvent_t start, stop;
float elapsedTime = 0.0f;
cudaStream_t stream = NULL;
std::vector<Bndbox> nms_pred;
PointPillar pointpillar;
ros::Publisher pub_cluster_3d;

MeasureGroup Measures;
double lidar_end_time = 0,last_timestamp_lidar = 0;
int need_pointnum;
mutex mtx_buffer;
int   scan_count = 0;
string lid_topic = "/kitti/velo/pointcloud";
const int point_num = 4096;
const int class_num = 13;
bool  flg_exit = false;
nvinfer1::ICudaEngine* engine;
nvinfer1::IExecutionContext* execution_context;
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI pl_surf;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
bool   lidar_pushed = true,flg_first_scan = true;
ros::Publisher pubLaserCloudFull;
ros::Publisher pubLaserCloudPoint;
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
double lidar_mean_scantime = 0.0;
int    scan_num = 0;

namespace pointself_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(pointself_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
)

// void Preprocess::pointself_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
// {
//     pcl::PointCloud<pointself_ros::Point> pl_orig;
//     pcl::fromROSMsg(*msg, pl_orig);
//     int plsize = pl_orig.points.size();
//     if (plsize == 0) return;
//     pl_surf.reserve(plsize);
// }

void Getinfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty()) {
        return false;
    }
    /*** push a lidar scan ***/
    if(!lidar_pushed)/// 标志位确保每帧激光只处理一次
    {
      meas.lidar = lidar_buffer.front();
      lidar_pushed = true;
    }
    lidar_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

// void SaveBoxPred(std::vector<Bndbox> boxes, std::string file_name)
// {
//     std::ofstream ofs;
//     ofs.open(file_name, std::ios::out);
//     if (ofs.is_open()) {
//         for (const auto box : boxes) {
//           ofs << box.x << " ";
//           ofs << box.y << " ";
//           ofs << box.z << " ";
//           ofs << box.w << " ";
//           ofs << box.l << " ";
//           ofs << box.h << " ";
//           ofs << box.rt << " ";
//           ofs << box.id << " ";
//           ofs << box.score << " ";
//           ofs << "\n";
//         }
//     }
//     else {
//       std::cerr << "Output file cannot be opened!" << std::endl;
//     }
//     ofs.close();
//     std::cout << "Saved prediction in: " << file_name << std::endl;
//     return;
// };

void pointself_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pcl::PointCloud<pcl::PointXYZI> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);
	for (int i = 0; i < plsize; i++)
	{
	PointType added_pt;
	added_pt.normal_x = 0;
	added_pt.normal_y = 0;
	added_pt.normal_z = 0;
  added_pt.intensity = pl_orig.points[i].intensity;
	added_pt.x = pl_orig.points[i].x;
	added_pt.y = pl_orig.points[i].y;
	added_pt.z = pl_orig.points[i].z;
	pl_surf.points.push_back(added_pt);
	}
}

void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{//时间单位转换
  pointself_handler(msg);
  *pcl_out = pl_surf;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    process(msg, ptr);
    lidar_buffer.push_back(ptr);
    std::cout<<lidar_buffer.size()<<std::endl;
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
}


void pub_cluster_fun()
{
  visualization_msgs::MarkerArray boxes;
  for (int i=0;i<nms_pred.size();i++){
    if (nms_pred[i].score<0.5){
      continue;
    }
    visualization_msgs::Marker marker;
    marker.header.frame_id = "base_link";  // 参考坐标系
    marker.ns = "boxes";
    marker.id = i;  // 每个marker必须有唯一的ID
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    
    // 设置位置（沿x轴排列）
    marker.pose.position.x = nms_pred[i].x;
    marker.pose.position.y = nms_pred[i].y;
    marker.pose.position.z = nms_pred[i].z;
    marker.pose.orientation.w = nms_pred[i].rt;  // 无旋转
    
    marker.scale.x = nms_pred[i].w;
    marker.scale.y = nms_pred[i].l;
    marker.scale.z = nms_pred[i].h;
    // std::cout<<"xyz:"<<std::endl;
    // std::cout<<obj_list[i].x_max<<","<<obj_list[i].x_min<<std::endl;
    // std::cout<<obj_list[i].y_max<<","<<obj_list[i].y_min<<std::endl;
    // std::cout<<obj_list[i].z_max<<","<<obj_list[i].z_min<<std::endl;
    std::cout<<nms_pred[i].x<<","<<nms_pred[i].y<<","<<nms_pred[i].z<<std::endl;
    
    // 设置不同颜色
    marker.color.r = 0.0 + i * 0.2;
    marker.color.g = 1.0 - i * 0.2;
    marker.color.b = 0.5;
    marker.color.a = 0.5;  // 透明度
    marker.lifetime = ros::Duration(0.1);  // 永久显示
    if(marker.scale.x<10&&marker.scale.y<10&&marker.scale.z<10){
      boxes.markers.push_back(marker);
    }
  }
  pub_cluster_3d.publish(boxes);
  nms_pred.clear();
}

void point_net(){
  unsigned int length = 0;
  void *data = NULL;
  float data_f[feats_undistort->points.size()*4];

  for (int i=0;i<feats_undistort->points.size();i++){
    data_f[i*4] = feats_undistort->points[i].x;
    data_f[i*4+1] = feats_undistort->points[i].y;
    data_f[i*4+2] = feats_undistort->points[i].z;
    data_f[i*4+3] = feats_undistort->points[i].intensity;
  }
  data = (void **)&data_f;
  size_t points_size = feats_undistort->points.size();
  
  float* points = data_f;
  std::cout << "find points num: "<< points_size <<std::endl;

  float *points_data = nullptr;
  unsigned int points_data_size = points_size * 4 * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
  checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
  checkCudaErrors(cudaDeviceSynchronize());
  cudaEventRecord(start, stream);
  pointpillar.doinfer(points_data, points_size, nms_pred);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout<<"TIME: pointpillar: "<< elapsedTime <<" ms." <<std::endl;
  // checkCudaErrors(cudaFree(points_data));
  std::cout<<"Bndbox objs: "<< nms_pred.size()<<std::endl;
  // std::string save_file_name = Save_Dir + index_str + ".txt";
  // SaveBoxPred(nms_pred, save_file_name);
  pub_cluster_fun();
  std::cout << ">>>>>>>>>>>" <<std::endl;

}

void keyboardInterruptHandler(int sig) {
    ROS_INFO("Shutting down...");
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));
    ros::shutdown();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pointpillar_node");
  ros::NodeHandle nh;
  Getinfo();
  signal(SIGINT, keyboardInterruptHandler);
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  ros::Subscriber sync = nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
  pointcloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/percp/radar_objs", 1000);
  pub_cluster_3d = nh.advertise<visualization_msgs::MarkerArray>("/obj_registered_3d", 100000);

  Params params_;
  pointpillar.set(Model_File, stream);
  nms_pred.reserve(100);
  ros::Rate rate(5000);
  bool status = ros::ok();
  while(status){
    ros::spinOnce();
    if(sync_packages(Measures)){
      if (flg_first_scan)
      {
          flg_first_scan = false;
          continue;
      }
      *feats_undistort = *(Measures.lidar);
      if (feats_undistort->empty() || (feats_undistort == NULL))
      {
          ROS_WARN("No point, skip this scan!\n");
          continue;
      }
      point_net();
      // publish_frame_world(pubLaserCloudFull);
    }
    status = ros::ok();
    rate.sleep();
  }
  return 0;
}
