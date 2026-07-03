cmake_minimum_required(VERSION 3.0.2)
project(fast_livo2_global_localization)

add_compile_options(-std=c++14 -O2)

find_package(catkin REQUIRED COMPONENTS
  roscpp rospy std_msgs sensor_msgs nav_msgs geometry_msgs
  tf2 tf2_ros tf2_geometry_msgs
  cv_bridge image_transport pcl_ros pcl_conversions
)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs cv_bridge image_transport pcl_ros pcl_conversions
  DEPENDS OpenCV PCL Eigen3
)

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

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

add_executable(segformer_mask_projector_node src/segformer_mask_projector_node.cpp)
target_link_libraries(segformer_mask_projector_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(rangenet_geometry_adapter_node src/rangenet_geometry_adapter_node.cpp)
target_link_libraries(rangenet_geometry_adapter_node
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
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS
  global_localizer_node
  cloud_topic_probe_node
  semantic_voxel_mapper_node
  segformer_mask_projector_node
  rangenet_geometry_adapter_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config docs DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
