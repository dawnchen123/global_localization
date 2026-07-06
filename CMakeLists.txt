cmake_minimum_required(VERSION 3.0.2)
project(fast_livo2_global_localization)

add_compile_options(-std=c++14 -O2)

find_package(catkin REQUIRED COMPONENTS
  roscpp rospy std_msgs sensor_msgs nav_msgs geometry_msgs
  visualization_msgs
  tf2 tf2_ros tf2_geometry_msgs
  cv_bridge image_transport pcl_ros pcl_conversions
  livox_ros_driver
)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs nav_msgs geometry_msgs visualization_msgs tf2 tf2_ros tf2_geometry_msgs cv_bridge image_transport pcl_ros pcl_conversions livox_ros_driver
  DEPENDS OpenCV PCL Eigen3
)

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

set(RANGENETPP_ROOT "/home/dawn/software/rangenet_ws" CACHE PATH "RangeNet++ rangenet_lib catkin workspace")
set(TENSORRT_ROOT "/usr/local/TensorRT-10.8.0.43" CACHE PATH "TensorRT root used by rangenet_lib")
set(RANGENETPP_INCLUDE_DIR "${RANGENETPP_ROOT}/src/rangenet_lib/include")
set(RANGENETPP_LIB_DIR "${RANGENETPP_ROOT}/devel/lib")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
if(EXISTS "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")
  set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")
else()
  set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")
endif()
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include" CACHE PATH "CUDA include directory used by TensorRT headers")

if(EXISTS "${RANGENETPP_INCLUDE_DIR}/selector.hpp" AND
   EXISTS "${RANGENETPP_LIB_DIR}/librangenet_lib.so" AND
   EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h" AND
   EXISTS "${TENSORRT_LIB_DIR}/libnvinfer.so")
  set(HAVE_REAL_RANGENETPP ON)
  include_directories(${RANGENETPP_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
  link_directories(${RANGENETPP_LIB_DIR} ${TENSORRT_LIB_DIR})
else()
  set(HAVE_REAL_RANGENETPP OFF)
  message(WARNING "Real RangeNet++ rangenet_lib/TensorRT files not found; rangenetpp_ros_node will not be built.")
endif()

add_executable(global_localizer_node src/global_localizer_node.cpp)
target_link_libraries(global_localizer_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})


add_executable(cloud_topic_probe_node src/cloud_topic_probe_node.cpp)
target_link_libraries(cloud_topic_probe_node
  ${catkin_LIBRARIES}
)

add_executable(semantic_voxel_mapper_node src/semantic_voxel_mapper_node.cpp)
target_link_libraries(semantic_voxel_mapper_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(semantic_keyframe_pose_graph_node src/semantic_keyframe_pose_graph_node.cpp)
target_link_libraries(semantic_keyframe_pose_graph_node
  ${catkin_LIBRARIES}
)

add_executable(segformer_mask_projector_node src/segformer_mask_projector_node.cpp)
target_link_libraries(segformer_mask_projector_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(rangenet_geometry_adapter_node src/rangenet_geometry_adapter_node.cpp)
target_link_libraries(rangenet_geometry_adapter_node
  ${catkin_LIBRARIES}
)

add_executable(livox_custom_to_pointcloud2_node src/livox_custom_to_pointcloud2_node.cpp)
target_link_libraries(livox_custom_to_pointcloud2_node
  ${catkin_LIBRARIES}
)

if(HAVE_REAL_RANGENETPP)
  set(TENSORRT_RUNTIME_LIBS
    ${TENSORRT_LIB_DIR}/libnvinfer.so
    ${TENSORRT_LIB_DIR}/libnvonnxparser.so
  )
  if(EXISTS "${TENSORRT_LIB_DIR}/libnvparsers.so")
    list(APPEND TENSORRT_RUNTIME_LIBS ${TENSORRT_LIB_DIR}/libnvparsers.so)
  endif()
  add_executable(rangenetpp_ros_node src/rangenetpp_ros_node.cpp)
  target_link_libraries(rangenetpp_ros_node
    ${catkin_LIBRARIES}
    rangenet_lib
    rangenet_external
    yaml-cpp
    -Wl,--no-as-needed
    ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
    -Wl,--as-needed
    ${TENSORRT_RUNTIME_LIBS}
  )
  set_target_properties(rangenetpp_ros_node PROPERTIES
    BUILD_RPATH "${RANGENETPP_LIB_DIR};${TENSORRT_LIB_DIR}"
    INSTALL_RPATH "${RANGENETPP_LIB_DIR};${TENSORRT_LIB_DIR}"
  )
endif()

add_executable(trajectory_groundtruth_evaluator_node src/trajectory_groundtruth_evaluator_node.cpp)
target_link_libraries(trajectory_groundtruth_evaluator_node
  ${catkin_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/satellite_tile_downloader.py
  scripts/efficient_loftr_bridge.py
  scripts/cloud_topic_probe.py
  scripts/map_match_evaluator.py
  scripts/eloftr_queue_exporter.py
  scripts/eloftr_queue_result_publisher.py
  scripts/eloftr_file_service.py
  scripts/semantic_file_service.py
  scripts/sam3_semantic_match_service.py
  scripts/geo_prior_constraint_match_service.py
  scripts/local_geometric_semantic_mapper.py
  scripts/camera_lidar_queue_exporter.py
  scripts/projected_semantic_bev_mapper.py
  scripts/sam3_image_mask_service.py
  scripts/rangenetpp_inference_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS
  global_localizer_node
  cloud_topic_probe_node
  semantic_voxel_mapper_node
  semantic_keyframe_pose_graph_node
  segformer_mask_projector_node
  rangenet_geometry_adapter_node
  livox_custom_to_pointcloud2_node
  trajectory_groundtruth_evaluator_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
if(HAVE_REAL_RANGENETPP)
  install(TARGETS rangenetpp_ros_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endif()
install(DIRECTORY launch config docs DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
