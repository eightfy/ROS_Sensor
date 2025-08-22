cmake_minimum_required(VERSION 3.10)
project(depthai_ros_yolo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  # vision_msgs
  cv_bridge
  image_transport
  common_msgs
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(ONNXRUNTIME_DIR "/usr/local/include/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

find_package(depthai REQUIRED)
message(STATUS "DepthAI include dir: ${depthai_INCLUDE_DIRS}")
message(STATUS "DepthAI libraries: ${depthai_LIBRARIES}")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs
  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(yolo_ros_node src/yolo_ros_node.cpp)
target_link_libraries(yolo_ros_node
 ${OpenCV_LIBS} 
 ${catkin_LIBRARIES}
 depthai::opencv)

install(TARGETS yolo_ros_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

