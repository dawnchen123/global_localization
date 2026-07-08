# FAST-LIVO2 Global Localization - Structured Prior Matching

This package implements the Chapter 3 pipeline as:

```text
FAST-LIVO2 local point cloud
  -> local occupancy / boundary / traversable / semantic BEV
  -> simplified geometric BEV
     (road area, road edge, building/structure, lane-line candidates)
  -> constrained matching against geographic priors
     (road network, building footprint, lane mask, entrance, CAD/BIM, OSM-style masks)
  -> low-rate absolute pose constraint
  -> EKF fusion with FAST-LIVO2 odometry
```

The important change is that the system no longer treats a LiDAR BEV as an RGB
image and forces strong feature matching against satellite imagery.  Both sides
are converted to structured intermediate layers first.  In tree-heavy scenes the
matcher uses the simplified geometric BEV and ignores vegetation labels.

## Main Nodes

- `global_localizer_node`
  - subscribes FAST-LIVO2 odometry/clouds
  - maintains the EKF state
  - publishes local BEV structure images and geographic crop metadata
  - gates external absolute constraints and fuses accepted constraints

- `local_geometric_semantic_mapper.py`
  - accumulates `/cloud_registered`
  - estimates local ground and filters semantic features to the ground-relative height band, default `[-0.2m, +1.5m]`, to suppress tree canopy
  - publishes `/local_semantic_map/label`, `/color`, `/confidence`
  - labels unknown, open ground, structure, vegetation and obstacle cells
  - publishes `/local_geometric_map/label`, `/color`, `/confidence`
  - drops vegetation and keeps road area, road edge, building/structure and lane-line candidates for matching

- `camera_lidar_queue_exporter.py`
  - exports synchronized camera image, raw LiDAR points and FAST-LIVO2 pose for image-mask semantic projection

- `sam3_image_mask_service.py`
  - runs in the SAM3 Python environment
  - segments camera images into road, building, tree and grass masks

- `projected_semantic_bev_mapper.py`
  - projects SAM3 image labels back to LiDAR points
  - filters projection by per-pixel LiDAR depth consistency
  - refines projected labels with SuMa++-style consistency checks:
    local mask support, range-edge penalty and LiDAR height/span geometry
  - accumulates semantic points in the FAST-LIVO2 map frame
  - fuses labels in map-frame voxels with semantic votes, margin and observation strength
  - publishes `/sam3/projected_semantic_points` as a colored semantic PointCloud2 with `label` and `confidence`
  - can publish labels in the semantic-SLAM internal label space for fusion with RangeNet++

- `semantic_voxel_mapper_node`
  - recommended semantic-SLAM mapper
  - subscribes FAST-LIVO2 odometry and RangeNet++ labeled LiDAR clouds
  - optionally subscribes SAM3/SegFormer-projected image semantic clouds
  - fuses labels in a C++ semantic voxel map
  - publishes `/semantic_cloud_map`, `/semantic_bev/label`, `/semantic_bev/color`, `/semantic_bev/traversable`

- `rangenet_geometry_adapter_node`
  - local RangeNet++ compatibility fallback when real RangeNet++ is not installed
  - subscribes `/cloud_registered`
  - publishes `/rangenet/semantic_points` with fields `x y z rgb label confidence`
  - only uses fast geometric labels, so it is for pipeline bring-up, not final semantic quality

- `segformer_mask_projector_node`
  - C++ projector for the second stage
  - consumes a mono8 SegFormer label mask and a PointCloud2
  - projects image labels to points and publishes `/segformer/projected_semantic_points`

- `eloftr_queue_exporter.py`
  - keeps the historical filename for compatibility
  - exports BEV structure, satellite crop, local simplified geometry labels and metadata

- `geo_prior_constraint_match_service.py`
  - runs outside ROS or in any Python environment with OpenCV/Numpy
  - reads queue requests
  - loads optional road/building/traversable/lane/entrance priors
  - slices priors to the current satellite crop and derives road edges/building outlines
  - searches dx/dy/yaw and writes `result_*.json`

- `eloftr_queue_result_publisher.py`
  - publishes accepted results as `/map_match_pose_external`
  - publishes overlays and heatmaps for diagnosis

## Build

```bash
cd ~/workspace/fast_livo2_global_ws
catkin_make
source devel/setup.bash
```

## Run ROS Side

```bash
roslaunch fast_livo2_global_localization global_localization.launch \
  run_satellite_downloader:=false \
  run_local_semantic_mapper:=true \
  run_eloftr_file_queue:=true
```

Check the structured local map:

```bash
rostopic hz /local_semantic_map/label
rqt_image_view /local_semantic_map/color
rostopic hz /local_geometric_map/label
rqt_image_view /local_geometric_map/color
```

Check matching inputs:

```bash
rostopic hz /local_bev/match_image
rostopic hz /local_rgb_bev/image
rostopic hz /satellite_map/aligned_crop
rostopic hz /map_match/input_meta
watch -n 1 'ls -lh /tmp/eloftr_queue/input /tmp/eloftr_queue/output 2>/dev/null'
```

## RangeNet++ Semantic SLAM

This is the recommended semantic mapping path. RangeNet++ is the real-time
LiDAR semantic source; SAM3 can be added as a lower-rate image instance/semantic
source to improve buildings, road surfaces and vegetation boundaries.

If real RangeNet++ is not installed, a local C++ compatibility adapter can be
started explicitly for pipeline debugging:

```text
/cloud_registered
  -> rangenet_geometry_adapter_node
  -> /rangenet/semantic_points
  -> semantic_voxel_mapper_node
  -> /semantic_cloud_map + /semantic_bev/*
```

```bash
roslaunch fast_livo2_global_localization semantic_slam.launch \
  run_rangenet_adapter:=true
```

This lets the whole semantic voxel map and BEV pipeline run without a neural
model. It is not a neural network replacement for final semantic quality, so it
is disabled by default.

For real RangeNet++, use the bridge launch. It disables the fallback adapter
automatically and runs RangeNet++ on the raw Hesai scan. The labeled cloud stays
in the LiDAR frame; `semantic_voxel_mapper_node` is responsible for transforming
and accumulating it in FAST-LIVO2 `camera_init`.

```bash
roslaunch fast_livo2_global_localization rangenetpp_semantic_slam.launch
```

Default data path:

```text
/hesai/at128/points              sensor_msgs/PointCloud2, raw LiDAR frame
  -> rangenetpp_ros_node          RangeNet++ inference in LiDAR frame
/rangenet/semantic_points         PointCloud2 x y z rgb label confidence, raw LiDAR frame
  -> semantic_voxel_mapper_node   transformed to camera_init/map by odometry and T_body_lidar
```

For old Livox/Mid-360 bags only, enable the legacy converter:

```bash
roslaunch fast_livo2_global_localization rangenetpp_semantic_slam.launch \
  run_livox_converter:=true \
  input_cloud_topic:=/livox/mid360/points_xyzirt \
  input_cloud_in_map_frame:=false \
  output_cloud_in_map_frame:=false
```

Configure the bridge in
`config/rangenetpp_inference_bridge.yaml`:

```yaml
backend_python_path: /home/dawn/software/rangenetpp_ros_backend
backend_module: rangenetpp_backend
backend_class: RangeNetPPBackend
model_dir: /home/dawn/models/rangenetpp
backend_label_space: semantic_kitti
output_label_mode: semantic_kitti
```

The backend wrapper must expose:

```python
class RangeNetPPBackend:
    def __init__(self, model_dir, model_config, model_weights, device, **kwargs):
        ...

    def infer(self, points_xyzi):
        # points_xyzi is Nx4 float32: x y z intensity
        # return labels or (labels, confidence)
        ...
```

A template is provided at
`scripts/rangenetpp_backend_template.py`. Copy it to the directory configured by
`backend_python_path` and replace the template body with your real RangeNet++
runtime/model loading code.

The bridge publishes:

```text
FAST-LIVO2 publishes:
  /aft_mapped_to_init

RangeNet++ bridge publishes:
  /rangenet/semantic_points
    sensor_msgs/PointCloud2 fields: x y z rgb label confidence
```

If RangeNet++ publishes SemanticKITTI ids, keep
`lidar_label_mode: semantic_kitti` in
`config/semantic_slam.yaml`. If your wrapper already publishes the internal
labels below, set `lidar_label_mode: internal`.

Internal labels:

```text
0 unknown
1 road
2 sidewalk / terrain
3 building / wall / fence
4 vegetation / trunk / pole
5 dynamic object
6 other object
```

Start the C++ semantic mapper:

```bash
source ~/workspace/fast_livo2_global_ws/devel/setup.bash
roslaunch fast_livo2_global_localization rangenetpp_semantic_slam.launch
```

To fuse SAM3 projected image semantics as a second source, run the SAM3 ROS
queue mapper and the SAM3 service, then enable the projected semantic input in
the RangeNet++ semantic-SLAM launch:

```bash
# Terminal 1: FAST-LIVO2.
roslaunch fast_livo mapping_i2nav.launch

# Terminal 2: export synchronized camera/LiDAR/odom requests and publish
# /sam3/projected_semantic_points after SAM3 results are ready.
# Defaults are /avt_camera/left/image/compressed and /hesai/at128/points.
roslaunch fast_livo2_global_localization sam3_projected_semantic_mapping.launch

# Terminal 3: run inside the sam3 mamba/conda environment.
mamba activate sam3
python3 ~/workspace/fast_livo2_global_ws/src/fast_livo2_global_localization/scripts/sam3_image_mask_service.py \
  --queue_dir /tmp/sam3_projected_semantic_queue \
  --sam3_root /home/dawn/software/sam3 \
  --sam3_checkpoint /home/dawn/software/sam3/checkpoints/sam3.pt \
  --device cuda \
  --sam3_dtype bf16

# Terminal 4: RangeNet++ real-time stream + semantic voxel fusion + pose graph.
roslaunch fast_livo2_global_localization rangenetpp_semantic_slam.launch \
  use_sam3_projected_semantic:=true
```

SAM3 projected labels are converted to the internal label space before fusion:
road stays road, building becomes building, tree/grass become vegetation, and
car/person/bicycle-style masks become dynamic. This prevents image labels from
being interpreted as sidewalk/building incorrectly and lets the mapper drop
dynamic points before voxel fusion.

Check outputs:

```bash
rostopic hz /sam3/projected_semantic_points
rostopic echo -n1 /local_semantic_map/projection_stats
rostopic echo -n1 /sam3/projected_semantic_cloud_map/stats
rostopic hz /semantic_cloud_map
rostopic echo -n1 /semantic_cloud_map/stats
rqt_image_view /semantic_bev/color
rqt_image_view /semantic_bev/traversable
```

In RViz, add `PointCloud2` for `/semantic_cloud_map` and color by `RGB8`.

Recommended startup order:

```bash
# 1. Start FAST-LIVO2 first. It must publish odometry.
roslaunch fast_livo mapping_i2nav.launch

# 2. Start/play the dataset if you are using rosbag.
# rosbag play --clock <bag>

# 3. Start real RangeNet++ bridge + semantic map.
roslaunch fast_livo2_global_localization rangenetpp_semantic_slam.launch

# 3b. If SAM3 projected semantics are running, enable fusion:
# roslaunch fast_livo2_global_localization rangenetpp_semantic_slam.launch use_sam3_projected_semantic:=true

# For fallback-only debugging, use:
# roslaunch fast_livo2_global_localization semantic_slam.launch run_rangenet_adapter:=true
```

Check the bridge and mapper:

```bash
rostopic hz /hesai/at128/points
rostopic hz /cloud_registered
rostopic hz /rangenet/semantic_points
rostopic echo -n1 /rangenet/semantic_points/fields
rostopic hz /semantic_cloud_map
rostopic echo -n1 /semantic_cloud_map/stats
rostopic echo -n1 /semantic_slam/z_constraint
rostopic hz /semantic_corrected_odom
rostopic hz /semantic_graph_corrected_odom
rostopic echo -n1 /semantic_pose_graph/stats
```

Semantic XYZ correction is enabled by default in `config/semantic_slam.yaml`.
It uses multi-frame semantic anchors:

- road/sidewalk cells become local ground anchors;
- building/other static cells become coarse XY object anchors;
- building/vegetation/other static cells become coarse Z object anchors;
- each new semantic frame estimates robust XY and Z residuals to existing anchors;
- the correction is applied before voxel fusion and also published as
  `/semantic_corrected_odom`.

This is the first practical layer before a full factor-graph backend.  It
reduces visible layer separation and return-pass XY misalignment in
`/semantic_cloud_map` without modifying FAST-LIVO2 internals.  Important tuning
knobs:

- `max_xy_correction_step`: lower it if XY correction jitters; raise it if drift is fast.
- `xy_correction_alpha`: lower for smoother XY correction.
- `min_xy_constraint_matches`: raise it if object matches are unstable.
- `max_xy_match_residual`: lower it to reject wrong building/object associations.
- `max_yaw_correction_step_deg`: lower it if yaw correction oscillates.
- `max_se2_residual_rms`: lower it to reject weak geometric matches.
- `pose_constraint_interval_sec`: semantic SE(2) constraints are applied at this low rate.
- `min_semantic_correction_trust`: lower it if you want corrections to fade more in unobserved areas.
- `max_z_correction_step`: lower it if correction jitters; raise it if Z drift is fast.
- `z_correction_alpha`: lower for smoother correction.
- `min_z_constraint_matches`: raise it if false corrections appear in sparse scenes.
- `z_anchor_resolution`: smaller anchors are stricter; larger anchors are smoother.
- `enable_local_semantic_map_constraint`: accumulate several semantic frames into
  a local submap, then match that local submap to mature global anchors at low
  frequency. Keep this on for drift correction; turn it off only for debugging
  per-frame constraints.
- `local_semantic_window_sec` / `local_semantic_min_frames`: increase these if
  `z_last_mad` is high or local matches are unstable; decrease them if correction
  reacts too slowly.

Watch these fields while replaying a bag:

```bash
rostopic echo -n1 /semantic_slam/z_constraint
```

Useful fields are `x_correction`, `y_correction`, `xy_matches`,
`yaw_correction_deg`, `se2_residual_rms`, `pose_constraint_accepted`,
`semantic_correction_trust`, `z_correction`, `z_slope_x`, `z_slope_y`,
`semantic_z_correction_trust`, `z_last_mad`, `z_constraint_stale_frames`,
`local_semantic_frames`, `local_semantic_points`, `local_semantic_constraints`,
`z_slope_samples`, and `z_slope_spread`. If `z_last_mad` stays high or
`semantic_z_correction_trust` decays, the mapper is rejecting unstable Z
constraints instead of forcing them into the corrected odometry.

`semantic_slam.launch` also starts a keyframe-level semantic pose graph by
default.  It subscribes to `/aft_mapped_to_init` and `/rangenet/semantic_points`,
creates static semantic keyframes, detects low-frequency revisit constraints
from road/building/sidewalk geometry, and publishes:

```text
/semantic_graph_corrected_odom
/semantic_pose_graph/path
/semantic_pose_graph/map
/semantic_pose_graph/stats
```

Use `/semantic_graph_corrected_odom` for RTK evaluation.  Important pose-graph
debug fields are `keyframes`, `loop_edges`, `xy_loop_edges`, `z_loop_edges`,
`loop_xy_rejected_ambiguous`, `last_loop_inliers`,
`last_loop_building_inliers`, `last_loop_rms`, `last_loop_z_median`,
`last_loop_z_mad`, `last_loop_spread_ratio`, `last_loop_score_ratio`,
`last_loop_xy_delta`, `last_loop_yaw_delta_deg`, `last_loop_weight`,
`last_opt_mean_residual`, `last_opt_max_step`, `output_mode`,
`output_orientation_mode`, `allow_unsafe_full_output`, `output_dx`,
`output_dy`, `output_dz`, and `output_dyaw_deg`.

The default output mode is now `safe`: `/semantic_graph_corrected_odom`,
`/semantic_pose_graph/path`, and `/semantic_pose_graph/map` apply Z correction
and only publish very small XY/Yaw corrections after strict graph-health gates
pass. This is deliberate: current RTK/evo evaluation shows local semantic
similarity can create false XY/Yaw matches. The backend therefore separates two
constraint types:

```text
building / static-object points -> structural XY/Yaw loop constraint
road / sidewalk ground points   -> Z loop constraint
```

XY/Yaw loop constraints are accepted only when they have enough structural
inliers, a healthy `last_loop_spread_ratio`, a clear
`last_loop_score_ratio/last_loop_score_gap`, and small proposed
`last_loop_xy_delta/last_loop_yaw_delta_deg`. Large local coarse search is
disabled by default because it previously accepted false large corrections.
`output_mode: full` is dangerous/debug only and is downgraded to `safe` unless
`allow_unsafe_full_output: true` is set explicitly.

If `loop_edges` remains zero, the current semantic geometry is not producing
accepted loop/revisit constraints; inspect segmentation quality and relax
`config/semantic_keyframe_pose_graph.yaml` only after checking the inlier/RMS
fields.

## RTK/IMU GroundTruth Evaluation

The evaluator builds a reference trajectory from dataset RTK and IMU topics:

```text
/novatel/oem7/fix       -> RTK ENU position
/adi/adis16465/imu      -> GroundTruth yaw/orientation
/aft_mapped_to_init     -> FAST-LIVO2 raw trajectory
/semantic_graph_corrected_odom -> semantic keyframe graph corrected trajectory
```

Start it after FAST-LIVO2 and the semantic mapper are publishing odometry:

```bash
roslaunch fast_livo2_global_localization trajectory_groundtruth_evaluation.launch
```

```bash
roslaunch fast_livo2_global_localization trajectory_ned_recorder.launch
```
Check live RMSE:

```bash
rostopic echo -n1 /trajectory_eval/rmse
rostopic hz /ground_truth/odom
```

If RMSE stays at `n=0`, inspect the diagnostic counters in
`/trajectory_eval/rmse`: `received=0` means the odometry topic is wrong or not
publishing; increasing `no_gt_match` means RTK/odom timestamps cannot be paired;
increasing `waiting_alignment` means the node is still collecting the initial
alignment window; small `alignment_spread` means the initial motion segment is
too short for a stable SE2 fit. The default `gt_match_mode: stamp_or_latest` first tries
header-stamp pairing, then falls back to the latest fresh RTK sample for bags
where FAST-LIVO2 odometry uses runtime stamps.

It publishes RViz paths:

```text
/trajectory_eval/ground_truth_path
/trajectory_eval/raw_aligned_path
/trajectory_eval/semantic_aligned_path
```

CSV logs are written by default to:

```text
/tmp/trajectory_eval/ground_truth.csv
/tmp/trajectory_eval/raw_fast_livo2_eval.csv
/tmp/trajectory_eval/semantic_corrected_eval.csv
```

To save the raw FAST-LIVO2 and fused odometry trajectories directly in the
plain local-NED text format `t pos_n pos_e pos_d qx qy qz qw`, start:

```bash
roslaunch fast_livo2_global_localization trajectory_ned_recorder.launch
```

Default outputs:

```text
/tmp/trajectory_records/fast_livo2_local_ned.txt
/tmp/trajectory_records/fused_local_ned.txt
```

The recorder subscribes to `/aft_mapped_to_init` and
`/semantic_graph_corrected_odom` by default. Change
`config/trajectory_ned_recorder.yaml` if you need `/semantic_corrected_odom`,
`/global_fused_odom`, `relative` time, or `coordinate_mode: as_is`.

Default alignment mode is `se2_window`: it estimates one initial yaw +
translation + Z offset from the first short motion segment and then freezes that
alignment, so the RMSE reflects drift instead of arbitrary `camera_init` frame
choice. Change parameters in
`config/trajectory_groundtruth_evaluator.yaml` if the RTK/IMU timestamps or
heading convention differ. If the IMU message has no valid orientation
quaternion, the evaluator falls back to integrating `angular_velocity.z` for GT
yaw; RTK still defines the GroundTruth position.

For offline evo evaluation of the saved local-NED trajectories:

```bash
cd /home/dawn/document/phd_exp/evo_tools
bash run_street02_evo_eval.sh street00
```

The wrapper defaults to `GT_TIME_SHIFT=27.0`, `ALIGN=se3`, and writes to
`street00/evo_eval_shift27.0_se3`. Override these without editing the script:

```bash
GT_TIME_SHIFT=0 ALIGN=origin bash run_street02_evo_eval.sh street00
```

In addition to evo APE/RPE logs, the evaluator writes
`components_fast_livo2_local_ned.csv` and `components_fused_local_ned.csv`
with aligned `err_xy`, `err_z`, `err_3d`, and `yaw_err_deg` columns.

If your RangeNet++ topic is not `/rangenet/semantic_points`, pass it directly:

```bash
roslaunch fast_livo2_global_localization semantic_slam.launch \
  rangenet_semantic_topic:=/your_rangenet_topic
```

Second stage, after a SegFormer runtime publishes a mono8 mask on
`/segformer/label_mask`, enable the C++ projector and fusion:

```bash
roslaunch fast_livo2_global_localization semantic_slam.launch run_segformer_projector:=true
```

This automatically enables `/segformer/projected_semantic_points` fusion in the
mapper.  The projector configuration is in
`config/segformer_mask_projector.yaml`.

The SegFormer projector uses the RangeNet++ point geometry by default:

```text
cloud=/rangenet/semantic_points
mask=/segformer/label_mask
output=/segformer/projected_semantic_points
```

Override names at launch time when your runtime uses different topics:

```bash
roslaunch fast_livo2_global_localization semantic_slam.launch \
  run_segformer_projector:=true \
  rangenet_semantic_topic:=/your_rangenet_topic \
  segformer_projector_cloud_topic:=/your_rangenet_topic \
  segformer_mask_topic:=/your_segformer_mask
```

## Legacy SAM3 Semantic SLAM

Use this when the goal is just to build and inspect a semantic point-cloud map,
without satellite/geographic matching.

Start the ROS-side exporter and mapper:

```bash
roslaunch fast_livo2_global_localization sam3_projected_semantic_mapping.launch
```

Run the SAM3 image segmentation service in the `sam3` environment:

```bash
mamba activate sam3

python3 ~/workspace/fast_livo2_global_ws/src/fast_livo2_global_localization/scripts/sam3_image_mask_service.py \
  --queue_dir /tmp/sam3_projected_semantic_queue \
  --sam3_root /home/dawn/software/sam3 \
  --sam3_checkpoint /home/dawn/software/sam3/checkpoints/sam3.pt \
  --device cuda \
  --sam3_dtype bf16
```

Check the semantic map:

```bash
rostopic hz /sam3/projected_semantic_cloud_map
rostopic echo -n1 /sam3/projected_semantic_cloud_map/stats
rqt_image_view /local_semantic_map/projection_debug
rqt_image_view /local_semantic_map/color
```

In RViz, add a `PointCloud2` display for `/sam3/projected_semantic_cloud_map` and color by
`RGB8`. The point cloud fields are `x`, `y`, `z`, `rgb`, `label` and
`confidence`.

The semantic mapper follows the same practical idea as SuMa++: single-frame
semantic labels are noisy measurements, not final map truth.  The default config
therefore keeps a point only when image-mask support, projection depth,
range-edge consistency and LiDAR ground/height-span geometry agree enough.  The
most useful tuning parameters are in
`config/sam3_projected_semantic_mapper.yaml`:

- `min_semantic_support_ratio`: raise it when class boundaries bleed into roads.
- `projection_depth_tolerance_m`: lower it when labels leak through foreground objects.
- `road_reject_ground_height_m`: lower it when road labels appear on walls/trees.
- `semantic_cloud_min_votes`: raise it for cleaner but slower-to-appear maps.
- `semantic_cloud_voxel_size`: raise it for more stable and faster map fusion.

Optional PLY export can be enabled in
`config/sam3_projected_semantic_mapper.yaml`:

```yaml
semantic_cloud_save_path: "/tmp/semantic_cloud_map.ply"
semantic_cloud_save_every_sec: 10.0
```

## Run Constrained Prior Matcher

Diagnostic mode, with weak priors inferred from the satellite crop:

```bash
python3 ~/workspace/fast_livo2_global_ws/src/fast_livo2_global_localization/scripts/geo_prior_constraint_match_service.py \
  --queue_dir /tmp/eloftr_queue \
  --search_radius_m 35 \
  --search_step_m 2 \
  --yaw_degs=-10,-5,0,5,10
```

SAM3-assisted geometry mode, run inside your `sam3` mamba/conda environment:

```bash
mamba activate sam3

python3 ~/workspace/fast_livo2_global_ws/src/fast_livo2_global_localization/scripts/geo_prior_constraint_match_service.py \
  --queue_dir /tmp/eloftr_queue \
  --prior_yaml /home/dawn/maps/fast_livo2_global_localization/geographic_prior.yml \
  --road_network_mask_path /home/dawn/maps/fast_livo2_global_localization/road_network_mask.png \
  --road_surface_mask_path /home/dawn/maps/fast_livo2_global_localization/road_surface_mask.png \
  --building_mask_path /home/dawn/maps/fast_livo2_global_localization/building_mask.png \
  --sam3_root /home/dawn/software/sam3 \
  --device cuda \
  --sam3_dtype bf16 \
  --use_sam3_local \
  --use_sam3_sat \
  --sam3_local_mode replace \
  --sam3_sat_mode merge \
  --search_radius_m 35 \
  --search_step_m 2 \
  --yaw_degs=-10,-5,0,5,10
```

In this mode, SAM3 only extracts road/building masks from `/local_rgb_bev/image`
and `/satellite_map/aligned_crop`. On the satellite side, explicit road-network,
road-surface and building masks remain the primary prior. SAM3 detections are
gated to those priors and used only as supplementary road/building evidence.
It still does not match raw RGB images directly.

Final-method mode, with real prior masks:

```bash
python3 ~/workspace/fast_livo2_global_ws/src/fast_livo2_global_localization/scripts/geo_prior_constraint_match_service.py \
  --queue_dir /tmp/eloftr_queue \
  --prior_yaml /home/dawn/maps/fast_livo2_global_localization/geographic_prior.yml \
  --road_network_mask_path /home/dawn/maps/fast_livo2_global_localization/road_network_mask.png \
  --road_surface_mask_path /home/dawn/maps/fast_livo2_global_localization/road_surface_mask.png \
  --road_mask_path /home/dawn/maps/fast_livo2_global_localization/road_mask.png \
  --building_mask_path /home/dawn/maps/fast_livo2_global_localization/building_mask.png \
  --traversable_mask_path /home/dawn/maps/fast_livo2_global_localization/traversable_mask.png \
  --lane_mask_path /home/dawn/maps/fast_livo2_global_localization/lane_mask.png \
  --entrance_csv_path /home/dawn/maps/fast_livo2_global_localization/entrances.csv
```

The satellite/geographic prior keeps only structured layers by default:
road-surface/road-network support, building footprints, derived road/building
edges, optional lane masks and optional entrances. Vegetation labels are dropped
unless `--keep_vegetation_prior` is explicitly set. If a road-network centerline
mask is supplied without a road-surface mask, it is buffered by
`--road_network_buffer_m` meters to form weak road-surface support.

`entrances.csv` supports either `east,north` columns in the matcher ENU frame or
`lat,lon` when the prior YAML defines a geographic origin/center.

## Outputs

- `/map_match_pose_external`: raw accepted absolute constraint from the matcher
- `/map_match_pose`: gated constraint after C++ validation
- `/global_fused_odom`: EKF-fused global odometry
- `/map_match/similarity_heatmap`: candidate score heatmap
- `/map_match/sat_bev_overlay`: local simplified geometry over geographic priors
- `/map_match/local_semantic_color`: local semantic labels used by the matcher
- `/local_geometric_map/color`: local simplified geometry used by the matcher
- `/geographic_prior_map/color`: colorized prior labels

## Notes

The old SAM3 service is still present as an experimental/legacy path, but the
recommended implementation is now `geo_prior_constraint_match_service.py`.
