cmake_minimum_required(VERSION 3.5)
project(pointpillar_ros)
add_compile_options(-std=c++14 -g)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_C_COMPILER /usr/bin/gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
  set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
endif()
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O2")
set( SMS 70 72 75 80 86)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    -Xcompiler -I/usr/aarch64-linux-gnu/include/
    -Xlinker -lsocket
    -Xlinker -rpath=/usr/lib/aarch64-linux-gnu/
    -Xlinker -rpath=/usr/aarch64-linux-gnu/lib/
    -Xlinker -L/usr/lib/aarch64-linux-gnu/
    -Xlinker -L/usr/aarch64-linux-gnu/lib/
)
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu/)
set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu/)

# find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(yaml-cpp REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  sensor_msgs
  message_generation
  std_msgs
  actionlib_msgs
  actionlib
  pluginlib
  tf2_ros
  pcl_ros
  common_msgs
)

catkin_package(
 CATKIN_DEPENDS roscpp rospy std_msgs
)

# ============= cuda ============
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/home/jean/TensorRT-8.5.1.7/include)
link_directories(/home/jean/TensorRT-8.5.1.7/lib)

# ============ opencv ============
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "    include path: ${CMAKE_CURRENT_SOURCE_DIR}")
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
cuda_add_library(yolo_infer SHARED ${SRCS} )
# target_link_libraries(yolo_infer nvinfer nvonnxparser ${OpenCV_LIBS})
#############
## Install ##
#############
# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)
add_executable(pointpillar_ros_node src/main.cpp src/PointpillarNode.cpp)

target_link_libraries(pointpillar_ros_node
    yolo_infer
    ${CUDA_LIBRARIES}
    ${catkin_LIBRARIES}
    nvinfer 
    nvonnxparser 
    ${OpenCV_LIBS}
)

add_dependencies(pointpillar_ros_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_include_directories(pointpillar_ros_node PRIVATE ${PYTHON_INCLUDE_DIRS})