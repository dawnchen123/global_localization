# 独立 LiDAR/IMU + SAM3 语义图定位

本包直接读取原始 LiDAR、IMU 和相机，不订阅 FAST-LIVO2 或其他 SLAM 里程计。默认 launch 会运行独立 ESKF-LIO 前端、GTSAM 后端、SAM3 队列导出器、投影语义 BEV、轨迹/地图保存接口和 RViz；SAM3 模型服务需要在独立 Python 环境中先启动。

```text
IMU/LiDAR/Image 独立入口 -> 时间水位调度 -> 去畸变 ESKF-LIO
LiDAR 深度 + 图像 patch -> 图像时刻直接 ESKF 更新 -> /hybrid/frontend/odometry
KLT/PnP -> 仅发布标定/同步观测，不作为相邻帧图因子
相机 + LiDAR + 前端位姿 -> SAM3 队列 -> 单帧投影语义观测和 BEV 地图
多帧语义子图 -> MNN + RANSAC/Huber 候选/内点/离群调试
经长时间间隔和空间重访验证的闭环/语义因子 -> GTSAM iSAM2 -> /hybrid/odometry
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

默认不再为每帧写 `color_*.png` 和 `debug_*.png`，避免长 bag 产生数 GB 的队列
文件；只有短时人工检查时才在上述命令追加 `--save_visualizations`。Hesai/Mid360
配置中的 `cleanup_processed_queue_files: true` 会在 mapper 已发布该帧的投影语义点后
清理该请求的图像、NPZ、标签和结果 JSON。它不会自动删除以前实验留下的旧队列目录。

## 3. Hesai AT128 完整启动

终端 2：

```bash
roslaunch fast_livo2_global_localization hybrid_localization_hesai.launch \
  pointcloud_topic:=/hesai/at128/points \
  imu_topic:=/adi/adis16465/imu \
  camera_topic:=/avt_camera/left/image/compressed \
  subscribe_wheel:=true \
  subscribe_camera:=true \
  enable_visual_frontend:=true \
  visual_observation_only:=true \
  enable_visual_loop_factors:=true \
  enable_sam3_semantics:=true \
  enable_sam3_visual_mask:=true \
  enable_sam3_lidar_filter:=false \
  enable_semantic_observation_factors:=true \
  enable_semantic_observation_xy_factors:=false \
  enable_semantic_observation_z_factors:=true \
  sam3_queue_dir:=/tmp/sam3_street00 \
  trajectory_save_path:=/data/result/hesai_graph.csv \
  frontend_trajectory_save_path:=/data/result/hesai_frontend.csv \
  object_save_path:=/data/result/hesai_objects.csv \
  semantic_cloud_save_path:=/data/result/hesai_sam3_map.ply \
  rviz:=true
```

这是长回放的稳定起点：SAM3 仍会发布 `/sam3/projected_semantic_points` 和
`/sam3/semantic_cloud_map`，但积压或过期的 SAM3 结果不会删掉 LiDAR 配准点，未独立
验证的语义/视觉闭环也不会改写轨迹。该 i2Nav 配置默认使用已标定的
`/insprobe/ranger/odometer` 前端速度约束，bag 播放必须包含该 topic；没有 Ranger
数据时显式传入 `subscribe_wheel:=false`。确认完整回放稳定后，再显式开启反馈实验。

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
  subscribe_wheel:=true \
  start_livox_converter:=true \
  subscribe_camera:=true \
  enable_visual_frontend:=true \
  visual_observation_only:=true \
  enable_visual_loop_factors:=false \
  enable_sam3_semantics:=true \
  enable_sam3_visual_mask:=false \
  enable_sam3_lidar_filter:=false \
  enable_semantic_observation_factors:=false \
  enable_semantic_observation_xy_factors:=false \
  enable_semantic_observation_z_factors:=false \
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
           /insprobe/ranger/odometer \
           /avt_camera/left/image/compressed
```

Mid360：

```bash
rosbag play --clock /path/to/street00.bag \
  --topics /adi/adis16465/imu \
           /livox/mid360/points \
           /insprobe/ranger/odometer \
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
`visual_dynamic_rejections > 0`。`visual_observation_only=true` 保留图像投影、
KLT/PnP 观测和调试输出，但不会将 patch 直接更新写入前端 ESKF，也不会将相邻
KLT/PnP 相对位姿写入 GTSAM。非相邻视觉闭环由
`enable_visual_loop_factors` 单独控制。视觉闭环每秒保留一条 ORB/深度关键帧，
但默认每 `4 s` 才执行一次耗时的候选检索和 PnP 验证；这不会删除中间历史，
而是避免回访段的匹配计算阻塞前端。若 stats 中出现
`visual_loop_ground_z_correction` 但 `visual_loop_z_constrained=false`，表示该
候选超过单条 Z 修正门限，XY 因子仍可独立应用而 Z 不会被拉动。

默认每 5 个互不重叠原始观测构成一个子图，因此 `semantic_keyframes` 约为 `semantic_observations_associated / 5`。稳定启动中 `semantic_observation_*_factors` 保持为零是正常的，语义地图仍会累积。只有显式启用语义反馈且检测到长时间间隔的空间重访时，这些计数才应增长。RViz 默认启用 `Optimized Semantic Graph Map` 和 `SAM3 Fused Semantic Map`；`Current Semantic Observation` 默认关闭，可手动勾选检查单帧投影。

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

## 9. A+C+D 回环鲁棒性实验

`semantic_gtsam_pose_graph_consensus_continuous_wheel_z.yaml` 是 A+C 主线、D 补充的
图后端 profile。它不改变 ESKF-LIO 前端标定，也不把 SAM3 语义因子默认并入本次
对照；先隔离验证视觉闭环、连续输出、轮速和地面 Z 的贡献。完整 street00 回放示例：

```bash
cd ~/workspace/fast_livo2_global_ws/src
export OUTPUT_ROOT=/data/result/street00_acd
export DURATION=1401
export START=0
export RATE=1.0
export POST_RUN_WAIT=60
export ENABLE_SAM3=false
export ENABLE_SEMANTIC_OBSERVATION_FACTORS=false
export ENABLE_SEMANTIC_XY_FACTORS=false
export ENABLE_SEMANTIC_Z_FACTORS=false
export GRAPH_TUNING_CONFIG=$PWD/fast_livo2_global_localization/config/semantic_gtsam_pose_graph_consensus_continuous_wheel_z.yaml

./fast_livo2_global_localization/tools/run_street00_ablation.sh full_graph
```

该命令仍启用相机和非相邻视觉闭环；`graph_tuning_config` 在基础 YAML 和 launch
布尔参数之后加载，因此 profile 内的 `enable_wheel_factors`、
`enable_sequential_ground_z` 会生效。不要同时显式传入
`ENABLE_WHEEL_FACTORS=false` 或 `ENABLE_SEQUENTIAL_GROUND_Z=false`。

检查 `/hybrid/semantic_graph/stats`：

| 字段 | 正常含义 |
|---|---|
| `visual_loop_support_waits` | 第一条已通过 LiDAR 验证的候选正在等待相邻帧复核，不是失败。 |
| `visual_loop_support_confirmations` | 两条候选在参考邻域、XY/yaw/Z 校正上达成一致。 |
| `visual_loop_factors` | 实际进入 iSAM2 的 DCS 鲁棒闭环数；没有足够独立支持时保持 0 是预期保护行为。 |
| `visual_loop_support_disagreements` | 两个 LiDAR 验证候选不一致，第二条被拒绝并重新开始支持窗口。 |
| `visual_loop_alternative_*` | 同一检索周期内，参考帧去相关的备选 PnP/LiDAR 候选数量、实际尝试数和成功替代首选的次数。首选正在等待多帧支持时不会切换备选。 |
| `correction_target_*` / `correction_applied_*` | 图优化要求的校正与在线已平滑施加的校正。 |
| `correction_lag_*` | 连续输出层尚未施加的暂存校正；应在车辆持续运动后收敛。 |
| `wheel_rejections` / `sequential_ground_rejections` | 原始 LIO 与轮速弧长不一致，或地面几何/多帧一致性不通过而被安全抑制。 |

`trajectory_save_path` 保存的仍是未平滑的 iSAM 优化结果，用于 EVO；连续校正只影响
实时 `/hybrid/odometry`、TF 和校正点云，避免在线消费方在一个回调内接收大跳变。

## 10. 关键配置

| 文件 | 关键内容 |
|---|---|
| `hybrid_localization_common.yaml` | ESKF、点面更新、局部地图、语义缓存和输出 |
| `hybrid_localization_hesai.yaml` | AT128 点时间、噪声和外参 |
| `hybrid_localization_mid360.yaml` | Mid360 点时间、噪声和外参 |
| `semantic_gtsam_pose_graph.yaml` | iSAM2、普通约束、多帧语义子图和 XY/Z 因子 |
| `semantic_gtsam_pose_graph_consensus_continuous_wheel_z.yaml` | A+C+D 实验：LiDAR 验证视觉闭环共识、DCS、连续输出、轮速和地面 Z 弱约束 |
| `sam3_hesai.yaml` | AT128 相机投影、队列等待、BEV 和 PLY |
| `sam3_mid360.yaml` | Mid360 相机投影、队列等待、BEV 和 PLY |

前端鲁棒性参数：

| 参数 | 说明 |
|---|---|
| `imu_time_offset` / `lidar_time_offset` / `camera_time_offset` | 统一时钟偏移，采用 `corrected_stamp = message_stamp + offset`；应由离线时间标定给出，i2Nav 默认均为 0。 |
| `strict_sensor_frame_validation` | 启动时校验 `body_from_lidar`、`body_from_imu` 和相机外参是否为有限的 SO(3) 刚体变换。 |
| `lidar_odometry/observability_eigen_ratio` | 点面 Hessian 的可观性特征值比例；不可观方向不会写入当前扫描校正。 |
| `lidar_odometry/max_mean_normalized_residual` | Huber 后扫描创新量门限，超限扫描不更新 ESKF 或局部地图。 |
| `lidar_odometry/map_insertion_*` | 比状态更新更严格的局部地图插入门限，避免弱约束或高残差帧污染地图。 |
| `semantic_lidar_filter/enabled` | 将 SAM3 相机标签投影到原始 LiDAR；动态点不参与扫描匹配、注册点云和局部地图插入。 |
| `semantic_lidar_filter/sync_tolerance_sec` | 光流传播标签与扫描末端的最大时间差。 |
| `semantic_lidar_filter/max_source_age_sec` | 原始 SAM3 标签的最大年龄，默认 `0.75 s`；超时结果不会参与 LiDAR 配准。 |
| `visual_frontend/sam3_max_source_age_sec` | SAM3 视觉 mask 的原始标签最大年龄，默认 `0.75 s`；超过该值仅保留原始图像观测。 |

`enable_sam3_semantics:=true` 只启用语义队列、投影地图和语义观测。`enable_sam3_visual_mask`、
`enable_sam3_lidar_filter` 和 `enable_semantic_observation_*_factors` 都是独立的实验开关，默认关闭。
在 `/hybrid/status` 中检查 `sensor_frame_contract`、`lio_observable_directions`、
`lio_mean_normalized_residual`、`sam3_lidar_mask_scans` 和
`sam3_lidar_dynamic_rejections`；若掩码持续不可用，应先检查相机标签 topic、
`sam3_source_age`、相机时间偏移及相机外参。

主要语义参数：

| 参数 | 说明 |
|---|---|
| `graph/semantic_submap_observations` | 每个不重叠语义子图使用的原始观测数，默认 5 |
| `graph/semantic_observation_min/max_index_gap` | 可匹配子图的关键帧间隔 |
| `graph/semantic_observation_min_time_separation_sec` | 因子候选的最小真实时间间隔；默认 `90 s`，禁止相邻街段匹配。 |
| `graph/semantic_observation_minimum_interval_sec` / `max_factors` | 两次已应用语义因子的最小间隔和总上限，默认 `180 s` / `2`，抑制相关误差累积。 |
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
| `semantic_batch_voxel_size` | 单帧投影语义点进入全局累计前的按类别体素压缩尺寸；保留空间覆盖并限制长回放内存 |
| `mapper_max_results_per_cycle` | mapper 每个轮询周期最多处理的已完成 SAM3 请求；默认 1，避免积压时瞬时大内存分配 |
| `cleanup_processed_queue_files` | 成功发布投影语义点后删除该请求的临时队列文件；默认配置启用，旧实验目录不会被追溯删除 |
| `measurement_scheduler/process_rate_hz` | 主线程检查待处理事件的墙钟频率 |
| `measurement_scheduler/max_events_per_tick` | 每轮最多处理的 LiDAR/Image 事件数，避免饿死状态和图后端回调 |
| `measurement_scheduler/reorder_window_sec` | LiDAR/Image 跨传感器重排序水位 |
| `visual_frontend/minimum_ncc` | patch 光度更新的最低归一化互相关 |
| `visual_frontend/photometric_huber_delta` | 直接视觉残差 Huber 门限 |
| `visual_frontend/semantic_class_weights` | SAM3 静态类别对视觉观测的信息权重 |
| `visual_loop/minimum_index_gap` | 非相邻视觉闭环最小关键帧间隔 |
| `visual_loop/maximum_database_size` | 保留的完整 ORB/深度历史数；默认 1600，覆盖 street00 的 1400 秒回放。 |
| `visual_loop/debug_image_history_size` | 仅为 RViz 调试保留的最近灰度图数量；旧关键帧仍可匹配，但其调试图以空白参考图显示。 |
| `visual_loop/retrieval_interval_sec` | 保留关键帧的同时执行数据库检索/PnP 的最小间隔；默认 `4 s`，防止长回放回访段的候选匹配抢占前端。 |
| `visual_loop/enable_global_retrieval_fallback` | 允许在原始里程计空间半径之外，以多哈希 ORB 倒排投票从完整历史库补充候选；仅影响候选召回，后续 PnP、LiDAR 与图优化门限不变。 |
| `visual_loop/maximum_global_retrieval_candidates` | 每轮最多补充的全局候选数；需与回放 CPU 余量一起设置。`0` 或关闭开关时完全保持空间检索行为。 |
| `visual_loop/global_retrieval_feature_count` | 每个关键帧进入全局倒排表的高响应 ORB 特征数；只影响检索内存和召回，不改变完整描述子的 PnP 匹配。 |
| `visual_loop/global_retrieval_min_votes` | 一个历史关键帧需要得到的 ORB 多哈希投票数；较高值更保守，较低值提高召回并增加后续 BF 匹配量。 |
| `visual_loop/global_retrieval_min_table_count` | 投票必须来自的最少独立哈希表数，用于抑制单一二进制码碰撞。 |
| `visual_loop/minimum_global_geometric_candidates` | 从全局候选中预留给 PnP 几何验证的最少席位，避免空间候选按匹配数排序后完全挤掉远距离回访。 |
| `visual_loop/maximum_verified_candidates` | 每次检索最多输出的、已通过 PnP 门限的候选数；图端仍逐条执行 LiDAR 验证和多帧一致性。 |
| `visual_loop/candidate_reference_min_separation_sec` | 同一检索中两个候选参考帧的最小时间间隔，避免相邻图像构成高度相关的替代因子。 |
| `visual_loop/minimum_pnp_inliers` | PnP-RANSAC 闭环最少内点数 |
| `graph/visual_loop_sigma_*` | GTSAM 视觉闭环 6DoF 噪声 |
| `graph/visual_loop_min_support` | 同一参考邻域需要的连续 LiDAR 验证视觉候选数；`1` 保持历史单候选行为。 |
| `graph/visual_loop_support_*` | 相邻候选的参考/当前关键帧窗口与 implied map-from-raw XY/yaw/Z 一致性门限。 |
| `graph/visual_loop_use_dcs` / `dcs_k` | 使用 GTSAM Dynamic Covariance Scaling 作为视觉回环软开关；大残差闭环会被动态降权。 |
| `continuous_correction_*` | 只限制在线 `map->odom` 的 XY/Z/旋转变化率，不修改 iSAM 优化轨迹或最终 EVO CSV。 |
| `graph/wheel_max_*disagreement` | Ranger 弧长和原始 LIO 关键帧平移不一致时拒绝该轮速因子。 |
| `graph/sequential_ground_min_support` | 地面 Z 残差需要连续多帧重复成立后才加入；`max_step` 限制单个关键帧的垂直注入。 |
| `graph/*_use_dcs` | 对 wheel 或 sequential-ground 因子使用 DCS 鲁棒核，默认关闭。 |
| `graph/visual_loop_ground_z_max_correction` | 单条视觉回环允许的地面 Z 修正上限；超过该值时保留 XY 回环验证结果，但不约束 Z。 |

在完成稳定基线后，可用如下命令进行受控语义重访实验。只有 `/hybrid/semantic_graph/stats` 中
确认因子来自长时间间隔重访、且全程 APE/RPE 改善时，才保留该配置：

```bash
roslaunch fast_livo2_global_localization hybrid_localization_hesai.launch \
  enable_sam3_semantics:=true \
  enable_sam3_lidar_filter:=false \
  enable_sam3_visual_mask:=false \
  enable_semantic_observation_factors:=true \
  enable_semantic_observation_xy_factors:=true \
  enable_semantic_observation_z_factors:=true \
  rviz:=true
```
