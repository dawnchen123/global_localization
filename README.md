# 独立 LiDAR/IMU + SAM3 语义图定位

本包直接读取原始 LiDAR、IMU 和相机，不订阅 FAST-LIVO2 或其他 SLAM 里程计。默认 launch 会运行独立 ESKF-LIO 前端、GTSAM 后端、SAM3 队列导出器、投影语义 BEV、轨迹/地图保存接口和 RViz；SAM3 模型服务需要在独立 Python 环境中先启动。

```text
IMU/LiDAR/Image 独立入口 -> 时间水位调度 -> 去畸变 ESKF-LIO
LiDAR 深度 + 图像 patch -> 图像时刻直接 ESKF 更新 -> /hybrid/frontend/odometry
KLT/PnP -> 仅发布标定/同步观测，不作为相邻帧图因子
相机 + LiDAR + 前端位姿 -> SAM3 动态掩码/类别权重 -> 单帧投影语义观测
3 个互不重叠观测 -> 多帧语义子图 -> MNN + RANSAC/Huber XY/Z 因子
非相邻 ORB/PnP 闭环 + 语义因子 -> GTSAM iSAM2 -> /hybrid/odometry
```

## 1. 编译

```bash
cd ~/workspace/fast_livo2_global_ws
source /opt/ros/noetic/setup.bash
export ROS_LANG_DISABLE=geneus:genlisp:gennodejs:genpy

catkin_make_isolated \
  --install \
  --only-pkg-with-deps fast_livo2_global_localization \
  --build-space build_hybrid \
  --devel-space devel_hybrid \
  --install-space install_hybrid \
  --make-args -j2 -l2
```

ROS 终端均先执行：

```bash
source /opt/ros/noetic/setup.bash
source ~/workspace/fast_livo2_global_ws/install_hybrid/setup.bash
```

## 2. 启动真实 SAM3

每次实验使用新的队列目录，避免旧结果混入。先在终端 1 启动模型服务，看到 `SAM3 loaded` 和 `image mask service ready` 后再启动 ROS launch：

```bash
mamba activate sam3
python3 ~/workspace/fast_livo2_global_ws/src/fast_livo2_global_localization/scripts/sam3_image_mask_service.py \
  --queue_dir /tmp/sam3_street00 \
  --backend sam3 \
  --sam3_root /home/dawn/software/sam3 \
  --sam3_checkpoint /home/dawn/software/sam3/checkpoints/sam3.pt \
  --device cuda \
  --sam3_dtype bf16 \
  --max_batch 1
```

服务会对每张图只计算一次 image backbone，并在同一 state 上执行文本 prompt。`--backend heuristic` 只用于检查 topic/队列链路，不能用于精度评估。

## 3. Hesai AT128 完整启动

终端 2：

```bash
roslaunch fast_livo2_global_localization hybrid_localization_hesai.launch \
  pointcloud_topic:=/hesai/at128/points \
  imu_topic:=/adi/adis16465/imu \
  camera_topic:=/avt_camera/left/image/compressed \
  subscribe_camera:=true \
  enable_visual_frontend:=true \
  visual_observation_only:=true \
  enable_visual_loop_factors:=true \
  enable_sam3_semantics:=true \
  enable_semantic_observation_factors:=true \
  enable_semantic_observation_xy_factors:=true \
  enable_semantic_observation_z_factors:=true \
  sam3_queue_dir:=/tmp/sam3_street00 \
  trajectory_save_path:=/data/result/hesai_graph.csv \
  frontend_trajectory_save_path:=/data/result/hesai_frontend.csv \
  object_save_path:=/data/result/hesai_objects.csv \
  semantic_cloud_save_path:=/data/result/hesai_sam3_map.ply \
  rviz:=true
```

Hesai 点云的 `timestamp` 字段在 i2Nav 数据集中是绝对 Unix 秒，launch 默认使用：

```text
lidar_point_time_mode:=absolute
lidar_point_time_scale:=1.0
```

其他驱动若输出相对纳秒，改为：

```text
lidar_point_time_mode:=relative lidar_point_time_scale:=1e-9
```

## 4. Livox Mid360 完整启动

终端 2：

```bash
roslaunch fast_livo2_global_localization hybrid_localization_mid360.launch \
  livox_custom_topic:=/livox/mid360/points \
  imu_topic:=/adi/adis16465/imu \
  camera_topic:=/avt_camera/left/image/compressed \
  start_livox_converter:=true \
  subscribe_camera:=true \
  enable_visual_frontend:=true \
  visual_observation_only:=true \
  enable_visual_loop_factors:=true \
  enable_sam3_semantics:=true \
  enable_semantic_observation_factors:=true \
  enable_semantic_observation_xy_factors:=true \
  enable_semantic_observation_z_factors:=true \
  sam3_queue_dir:=/tmp/sam3_street00 \
  trajectory_save_path:=/data/result/mid360_graph.csv \
  frontend_trajectory_save_path:=/data/result/mid360_frontend.csv \
  object_save_path:=/data/result/mid360_objects.csv \
  semantic_cloud_save_path:=/data/result/mid360_sam3_map.ply \
  rviz:=true
```

该 launch 会额外启动 `livox_custom_to_pointcloud2_node`，将 `CustomMsg` 转为保留逐点 `offset_time` 的 `/livox/mid360/points_xyzirt`。若驱动已输出带有效逐点时间的 `PointCloud2`，使用：

```bash
roslaunch fast_livo2_global_localization hybrid_localization_mid360.launch \
  start_livox_converter:=false \
  pointcloud_topic:=/your/mid360_pointcloud2 \
  enable_sam3_semantics:=true \
  sam3_queue_dir:=/tmp/sam3_street00 \
  rviz:=true
```

## 5. 播放 i2Nav bag

Hesai：

```bash
rosbag play --clock /path/to/street00.bag \
  --topics /adi/adis16465/imu \
           /hesai/at128/points \
           /avt_camera/left/image/compressed
```

Mid360：

```bash
rosbag play --clock /path/to/street00.bag \
  --topics /adi/adis16465/imu \
           /livox/mid360/points \
           /avt_camera/left/image/compressed
```

必须保留 bag 起始静止段用于 IMU 初始化。若使用仿真时间，在启动 launch 前执行 `rosparam set use_sim_time true`。

## 6. 启动的节点与保存接口

| 节点 | 功能 |
|---|---|
| `pdf_hybrid_localization_node` | 统一时间调度、IMU 初始化、去畸变 ESKF-LIO、稀疏 patch 视觉地图、图像时刻直接更新和前端 CSV |
| `semantic_gtsam_pose_graph_node` | KLT/PnP 只观测调试、非相邻视觉闭环、多帧语义 XY/Z 因子、iSAM2 和最终 CSV |
| `camera_lidar_queue_exporter.py` | 相机/点云同步，并等待 `/hybrid/frontend/odometry` 时间覆盖后导出队列 |
| `sam3_image_mask_service.py` | 独立环境中的真实 SAM3 CUDA 推理 |
| `projected_semantic_bev_mapper.py` | 语义投影、深度/几何过滤、BEV 和 PLY 保存 |
| `livox_custom_to_pointcloud2_node` | 仅 Mid360 启动 |
| `rviz` | 轨迹、语义地图和约束 Marker |

`trajectory_save_path` 是 GTSAM 最终轨迹，`frontend_trajectory_save_path` 是未经过后端修正的 ESKF 轨迹。CSV 在运行中按关键帧检查点刷新；PLY 在语义地图更新和节点退出时保存。长实验不要写入 `/tmp`。

## 7. 语义地图和因子检查

关键 topic：

| Topic | 内容 |
|---|---|
| `/sam3/projected_semantic_points` | 单次 SAM3 结果投影到 LiDAR 后的 map-frame 语义点 |
| `/hybrid/semantic_observation` | 保留原始时间戳、供图后端使用的单帧观测 |
| `/sam3/semantic_cloud_map` | SAM3 投影融合地图 |
| `/sam3/semantic_cloud_map/stats` | 语义帧时间范围、最新位姿、位姿范围和窗口路径长度 |
| `/hybrid/semantic_graph/map` | 使用优化关键帧位姿重建的多帧语义子图地图 |
| `/hybrid/visual/debug/projection` | LiDAR 深度投影图，用于检查相机内外参 |
| `/hybrid/visual/debug/tracks` | KLT 跟踪图 |
| `/hybrid/visual/debug/pnp_inliers` | PnP-RANSAC 内点图 |
| `/hybrid/visual/debug/image_cloud_dt` | 图像与配对点云时间差 |
| `/hybrid/visual/direct_debug` | patch 直接 ESKF 更新的有效残差与状态 |
| `/hybrid/visual/debug/loop` | 非相邻视觉闭环候选、内点与拒绝原因 |
| `/semantic_slam/semantic_xy_observation_debug` | candidate/inlier/outlier/applied 语义 XY 匹配 |
| `/semantic_slam/semantic_z_observation_debug` | Z 门限后的有效线和灰色超限候选 |
| `/hybrid/semantic_graph/stats` | 观测、子图、因子、匹配质量、地图和同步统计 |

```bash
rostopic hz /hybrid/semantic_observation
rostopic echo -n 1 /sam3/semantic_cloud_map/stats
rostopic echo -n 1 /hybrid/semantic_graph/map/width
rostopic echo -n 1 /hybrid/semantic_graph/stats
```

运行一段时间后至少应满足：

```text
semantic_messages > 0
semantic_observations_received == semantic_observations_associated
semantic_keyframes > 0
semantic_map_points > 0
registered_cloud_queue_drops == 0
registered_cloud_pose_drops == 0
scheduler_queue_drops == 0
scheduler_stale_drops == 0
visual_attempts > 0
visual_accepts > 0
```

启用 SAM3 后还应看到 `sam3_camera_labels_applied > 0` 和
`visual_dynamic_rejections > 0`。`visual_observation_only=true` 只禁止相邻
KLT/PnP 进入 GTSAM，不会关闭前端 patch 直接更新，也不会关闭非相邻视觉闭环。

默认每 3 个互不重叠原始观测构成一个子图，因此 `semantic_keyframes` 约为 `semantic_observations_associated / 3`。场景有足够重叠和结构时，`semantic_observation_xy_factors`、`semantic_observation_z_factors` 应增长。RViz 默认启用 `Optimized Semantic Graph Map` 和 `SAM3 Fused Semantic Map`；`Current Semantic Observation` 默认关闭，可手动勾选检查单帧投影。

若 `/sam3/semantic_cloud_map` 看起来停在起点，先连续读取两次 stats。`latest_stamp`
不增长表示导出/SAM3 队列停滞；时间增长但 `latest_pose_xyz` 不变表示车辆尚未运动或
前端位姿未更新；`pose_path_length_m` 增长则说明地图正在随轨迹构建。该 topic 默认只
保留 `accumulation_window_sec` 内的局部地图，不会永久保留起点点云。

导出器日志中的 `pending_pairs` 表示正在等待前端位姿追上，短时非零正常。`stamp`
模式下 `timeout_drops` 应保持 0；`capacity_drops` 增长时应降低 bag 播放速度、增大
`max_pending_sync_pairs` 或减少前端负载。

## 8. 评估

有在线 `nav_msgs/Odometry` 真值时：

```bash
roslaunch fast_livo2_global_localization hybrid_localization_evaluation.launch \
  ground_truth_topic:=/ground_truth/odom \
  save_path:=/data/result/hybrid_metrics.csv
```

i2Nav CSV 真值可离线评估：

```bash
python3 ~/workspace/fast_livo2_global_ws/src/fast_livo2_global_localization/tools/evo_local_ned_eval.py \
  --gt /path/to/street00_trajectory.csv \
  --est /data/result/hesai_frontend.csv /data/result/hesai_graph.csv \
  --names frontend semantic_graph \
  --out_dir /data/result/evo \
  --time_mode relative \
  --gt_time_shift 27 \
  --run_rpe
```

四组消融必须使用相同 bag 区间、播放速率和真值时间偏移。脚本会使用互相隔离的
ROS master，自动启动真实 SAM3 服务，并拒绝覆盖已有结果：

```bash
cd ~/workspace/fast_livo2_global_ws/src
export OUTPUT_ROOT=/data/result/street00_ablation
export DURATION=1401
export START=0
export RATE=0.5

./fast_livo2_global_localization/tools/run_street00_ablation.sh lio
./fast_livo2_global_localization/tools/run_street00_ablation.sh lio_visual
./fast_livo2_global_localization/tools/run_street00_ablation.sh lio_visual_sam3
./fast_livo2_global_localization/tools/run_street00_ablation.sh full_graph
```

四种模式分别是纯 LIO、LIO+patch 直接视觉、LIO+视觉+SAM3、以及再加入
非相邻视觉闭环和语义 XY/Z 因子的完整 GTSAM。GPU/CPU 无法实时处理时应降低
`RATE`，不能允许传感器订阅队列丢帧后再比较精度。

统一计算 APE、每 1 m RPE、yaw RMSE、Z RMSE 和 Z MAE：

```bash
cd ~/workspace/fast_livo2_global_ws/src
OUTPUT_ROOT=/data/result/street00_ablation \
  ./fast_livo2_global_localization/tools/evaluate_street00_ablation.sh
```

汇总结果位于 `evaluation/ablation_metrics.csv`。是否优于 FAST-LIVO2 必须以
完整数据集和同一评估设置为准，不能由短片段或单次运行保证。

## 9. 关键配置

| 文件 | 关键内容 |
|---|---|
| `hybrid_localization_common.yaml` | ESKF、点面更新、局部地图、语义缓存和输出 |
| `hybrid_localization_hesai.yaml` | AT128 点时间、噪声和外参 |
| `hybrid_localization_mid360.yaml` | Mid360 点时间、噪声和外参 |
| `semantic_gtsam_pose_graph.yaml` | iSAM2、普通约束、多帧语义子图和 XY/Z 因子 |
| `sam3_hesai.yaml` | AT128 相机投影、队列等待、BEV 和 PLY |
| `sam3_mid360.yaml` | Mid360 相机投影、队列等待、BEV 和 PLY |

主要语义参数：

| 参数 | 说明 |
|---|---|
| `graph/semantic_submap_observations` | 每个不重叠语义子图使用的原始观测数，默认 3 |
| `graph/semantic_observation_min/max_index_gap` | 可匹配子图的关键帧间隔 |
| `graph/semantic_observation_correspondence_distance` | 同标签互为最近邻半径 |
| `graph/semantic_observation_ransac_inlier_distance` | XY RANSAC 内点门限 |
| `graph/semantic_observation_min_inliers` | XY 最少内点数 |
| `graph/semantic_observation_min_spread_ratio` | 抑制单直线退化匹配 |
| `graph/semantic_observation_max_xy/yaw_correction` | 单条语义修正硬门限 |
| `graph/semantic_observation_sigma_xy/z` | GTSAM 语义因子噪声 |
| `max_pending_sync_pairs` | 等待前端位姿的相机/点云对上限 |
| `max_pending_sync_wait_sec` | 非 `stamp` 模式的墙钟等待上限；`stamp` 模式设为 `0`，仅按队列容量限流 |
| `export_rate` | SAM3 请求频率，当前真实推理约 1 Hz |
| `queue_image_format` | SAM3 输入队列图像格式，默认 `jpg`，避免高负载下 PNG 编码开销 |
| `queue_jpeg_quality` | JPEG 队列图像质量，默认 95 |
| `opencv_num_threads` | exporter/mapper 的 OpenCV 线程数，默认 1，避免与 LIO/SAM3 过度争抢 CPU |
| `accumulation_window_sec` | `/sam3/semantic_cloud_map` 的滑动时间窗口；它是局部语义地图，不是永久全局点云 |
| `measurement_scheduler/process_rate_hz` | 主线程检查待处理事件的墙钟频率 |
| `measurement_scheduler/max_events_per_tick` | 每轮最多处理的 LiDAR/Image 事件数，避免饿死状态和图后端回调 |
| `measurement_scheduler/reorder_window_sec` | LiDAR/Image 跨传感器重排序水位 |
| `visual_frontend/minimum_ncc` | patch 光度更新的最低归一化互相关 |
| `visual_frontend/photometric_huber_delta` | 直接视觉残差 Huber 门限 |
| `visual_frontend/semantic_class_weights` | SAM3 静态类别对视觉观测的信息权重 |
| `visual_loop/minimum_index_gap` | 非相邻视觉闭环最小关键帧间隔 |
| `visual_loop/minimum_pnp_inliers` | PnP-RANSAC 闭环最少内点数 |
| `graph/visual_loop_sigma_*` | GTSAM 视觉闭环 6DoF 噪声 |

纯几何消融可使用：

```bash
roslaunch fast_livo2_global_localization hybrid_localization_hesai.launch \
  enable_sam3_semantics:=false \
  enable_semantic_observation_factors:=false \
  rviz:=true
```
