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
  - publishes `/semantic_cloud_map` as a colored semantic PointCloud2 with `label` and `confidence`

- `semantic_voxel_mapper_node`
  - recommended semantic-SLAM mapper
  - subscribes FAST-LIVO2 odometry and RangeNet++ labeled LiDAR clouds
  - optionally subscribes SegFormer-projected image semantic clouds
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

This is the new recommended semantic mapping path.  The old SAM3/local
heuristic segmentation path should be treated as legacy.

If real RangeNet++ is not installed, the launch starts a local C++ compatibility
adapter by default:

```text
/cloud_registered
  -> rangenet_geometry_adapter_node
  -> /rangenet/semantic_points
  -> semantic_voxel_mapper_node
  -> /semantic_cloud_map + /semantic_bev/*
```

This lets the whole semantic voxel map and BEV pipeline run immediately.  It is
not a neural network replacement for final semantic quality.

With real RangeNet++ installed, disable the adapter:

```bash
roslaunch fast_livo2_global_localization semantic_slam.launch \
  run_rangenet_adapter:=false \
  rangenet_semantic_cloud_in_map_frame:=false \
  lidar_label_mode:=semantic_kitti \
  rangenet_semantic_topic:=/rangenet/semantic_points
```

Real RangeNet++ should publish:

```text
FAST-LIVO2 publishes:
  /aft_mapped_to_init

RangeNet++ publishes:
  /rangenet/semantic_points
    sensor_msgs/PointCloud2 fields: x y z label [confidence]
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
roslaunch fast_livo2_global_localization semantic_slam.launch
```

Check outputs:

```bash
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

# 3. No local RangeNet++ needed by default.
# semantic_slam.launch starts rangenet_geometry_adapter_node:
#   /cloud_registered -> /rangenet/semantic_points

# 4. Start semantic voxel mapping.
roslaunch fast_livo2_global_localization semantic_slam.launch
```

Check the adapter and mapper:

```bash
rostopic hz /cloud_registered
rostopic hz /rangenet/semantic_points
rostopic echo -n1 /rangenet/semantic_points/fields
rostopic hz /semantic_cloud_map
rostopic echo -n1 /semantic_cloud_map/stats
```

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
rostopic hz /semantic_cloud_map
rostopic echo -n1 /semantic_cloud_map/stats
rqt_image_view /local_semantic_map/projection_debug
rqt_image_view /local_semantic_map/color
```

In RViz, add a `PointCloud2` display for `/semantic_cloud_map` and color by
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
