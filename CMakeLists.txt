cmake_minimum_required(VERSION 3.0.2)
project(fast_livo2_global_localization)

add_compile_options(-std=c++14 -O2)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  diagnostic_msgs
  livox_ros_driver
  tf2
  tf2_ros
)
# The locally installed GTSAM 4.0.2 was built against Ubuntu's Eigen 3.3.7.
# Prefer that matching ABI over a newer /usr/local Eigen installation.
if(EXISTS "/usr/lib/cmake/eigen3/Eigen3Config.cmake")
  set(Eigen3_DIR "/usr/lib/cmake/eigen3" CACHE PATH
      "Eigen version matching the installed GTSAM ABI" FORCE)
endif()
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GTSAM REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs calib3d video features2d)
set(GTSAM_INCLUDE_DIRS ${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBRARIES gtsam)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES hybrid_semantic_pose_graph hybrid_visual_rotation hybrid_sparse_visual_map
            hybrid_visual_loop
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs nav_msgs geometry_msgs
                 visualization_msgs diagnostic_msgs livox_ros_driver tf2 tf2_ros
  DEPENDS GTSAM OpenCV
)

if(NOT EIGEN3_INCLUDE_DIRS AND EIGEN3_INCLUDE_DIR)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(hybrid_semantic_pose_graph
  src/semantic_pose_graph.cpp
)
target_link_libraries(hybrid_semantic_pose_graph gtsam)

add_library(hybrid_visual_rotation
  src/visual_rotation_tracker.cpp
)
# GTSAM exports the Eigen vectorization flags used by the pose-graph consumer
# (notably -march=native on the local 4.0.2 build).  Inheriting that interface
# keeps fixed-size Eigen objects ABI-compatible across this shared-library API.
target_link_libraries(hybrid_visual_rotation ${OpenCV_LIBRARIES} gtsam)

add_library(hybrid_sparse_visual_map
  src/sparse_visual_map.cpp
)
target_link_libraries(hybrid_sparse_visual_map ${OpenCV_LIBRARIES})

add_library(hybrid_visual_loop
  src/visual_loop_detector.cpp
)
target_link_libraries(hybrid_visual_loop ${OpenCV_LIBRARIES} gtsam)

add_executable(pdf_hybrid_localization_node
  src/hybrid_localization_core.cpp
  src/lidar_odometry.cpp
  src/hybrid_localization_node.cpp
)
target_link_libraries(pdf_hybrid_localization_node
  hybrid_sparse_visual_map
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(semantic_gtsam_pose_graph_node
  src/semantic_gtsam_pose_graph_node.cpp
)
target_link_libraries(semantic_gtsam_pose_graph_node
  hybrid_semantic_pose_graph
  hybrid_visual_rotation
  hybrid_visual_loop
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  gtsam
)

add_executable(hybrid_localization_core_smoke_test
  src/hybrid_localization_core.cpp
  src/lidar_odometry.cpp
  src/hybrid_localization_core_smoke_test.cpp
)
if(CATKIN_ENABLE_TESTING)
  add_test(NAME hybrid_localization_core_smoke_test
           COMMAND hybrid_localization_core_smoke_test)
endif()

add_executable(visual_loop_detector_smoke_test
  src/visual_loop_detector_smoke_test.cpp
)
target_link_libraries(visual_loop_detector_smoke_test
  hybrid_visual_loop
  ${OpenCV_LIBRARIES}
  gtsam
)
if(CATKIN_ENABLE_TESTING)
  add_test(NAME visual_loop_detector_smoke_test
           COMMAND visual_loop_detector_smoke_test)
endif()

add_executable(semantic_pose_graph_smoke_test
  src/semantic_pose_graph_smoke_test.cpp
)
target_link_libraries(semantic_pose_graph_smoke_test
  hybrid_semantic_pose_graph
  gtsam
)
if(CATKIN_ENABLE_TESTING)
  add_test(NAME semantic_pose_graph_smoke_test
           COMMAND semantic_pose_graph_smoke_test)
endif()

add_executable(visual_rotation_tracker_smoke_test
  src/visual_rotation_tracker_smoke_test.cpp
)
target_link_libraries(visual_rotation_tracker_smoke_test
  hybrid_visual_rotation
  ${OpenCV_LIBRARIES}
  gtsam
)
if(CATKIN_ENABLE_TESTING)
  add_test(NAME visual_rotation_tracker_smoke_test
           COMMAND visual_rotation_tracker_smoke_test)
endif()

add_executable(sparse_visual_map_smoke_test
  src/sparse_visual_map_smoke_test.cpp
)
target_link_libraries(sparse_visual_map_smoke_test
  hybrid_sparse_visual_map
  ${OpenCV_LIBRARIES}
)
if(CATKIN_ENABLE_TESTING)
  add_test(NAME sparse_visual_map_smoke_test
           COMMAND sparse_visual_map_smoke_test)
endif()

add_executable(livox_custom_to_pointcloud2_node
  src/livox_custom_to_pointcloud2_node.cpp
)
target_link_libraries(livox_custom_to_pointcloud2_node ${catkin_LIBRARIES})

add_executable(fast_livo2_global_hybrid_localization_evaluator_node
  src/hybrid_localization_evaluator_node.cpp
)
set_target_properties(fast_livo2_global_hybrid_localization_evaluator_node PROPERTIES
  OUTPUT_NAME hybrid_localization_evaluator_node
)
target_link_libraries(fast_livo2_global_hybrid_localization_evaluator_node
  ${catkin_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/camera_lidar_queue_exporter.py
  scripts/projected_semantic_bev_mapper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS hybrid_semantic_pose_graph hybrid_visual_rotation
                hybrid_sparse_visual_map hybrid_visual_loop
                pdf_hybrid_localization_node semantic_gtsam_pose_graph_node
                fast_livo2_global_hybrid_localization_evaluator_node
                livox_custom_to_pointcloud2_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY launch config docs rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
