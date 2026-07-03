#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-resolution satellite tile downloader for FAST-LIVO2 wide-area localization.

Compared with the first version:
1. `coverage_m` means a 3 km x 3 km area by default, not a 3 km radius square.
2. The downloaded mosaic is high-resolution. Default zoom is 19. The full-resolution
   image is saved on disk for matching; the ROS image can be downsampled for display.
3. IMU yaw is recorded in the sidecar YAML. Downloading XYZ tiles is north-up, while
   yaw is used by the C++ node for north-aligned BEV projection and crop alignment.
"""
import math
import os
import time
from io import BytesIO

import rospy
from sensor_msgs.msg import NavSatFix, Imu, Image
from cv_bridge import CvBridge

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from PIL import Image as PILImage
    import numpy as np
    import tf.transformations as tft
except Exception:  # pragma: no cover
    requests = None
    HTTPAdapter = None
    Retry = None
    PILImage = None
    np = None
    tft = None

R_EARTH = 6378137.0
TILE_SIZE = 256


def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def meters_per_pixel(lat, zoom):
    return math.cos(math.radians(lat)) * 2.0 * math.pi * R_EARTH / (TILE_SIZE * (2.0 ** zoom))


def yaw_from_imu(msg):
    q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    try:
        return tft.euler_from_quaternion(q)[2]
    except Exception:
        return 0.0


class SatelliteTileDownloader:
    def __init__(self):
        self.gps_topic = rospy.get_param('~gps_topic', '/fix')
        self.imu_topic = rospy.get_param('~imu_topic', '/imu')
        self.coverage_m = float(rospy.get_param('~coverage_m', 3000.0))
        self.coverage_is_radius = bool(rospy.get_param('~coverage_is_radius', False))
        self.zoom = int(rospy.get_param('~zoom', 19))
        self.max_tiles = int(rospy.get_param('~max_tiles', 2500))
        self.reload_distance_m = float(rospy.get_param('~reload_distance_m', 1000.0))
        self.cache_dir = rospy.get_param('~cache_dir', '/tmp/fast_livo2_global_localization')
        self.tile_url_template = rospy.get_param('~tile_url_template', '')
        self.force_redownload = bool(rospy.get_param('~force_redownload', False))
        self.publish_downsample_scale = float(rospy.get_param('~publish_downsample_scale', 0.15))
        self.user_agent = rospy.get_param('~user_agent', 'Mozilla/5.0 fast_livo2_global_localization/0.4')
        self.retry_times = int(rospy.get_param('~retry_times', 5))
        self.retry_backoff = float(rospy.get_param('~retry_backoff', 0.8))
        self.tile_sleep_sec = float(rospy.get_param('~tile_sleep_sec', 0.06))
        self.verify_ssl = bool(rospy.get_param('~verify_ssl', True))
        self.tile_cache_dir = os.path.join(self.cache_dir, 'tile_cache_z{}'.format(self.zoom))
        os.makedirs(self.tile_cache_dir, exist_ok=True)
        self.session = None
        if requests is not None:
            self.session = requests.Session()
            if Retry is not None and HTTPAdapter is not None:
                # urllib3 compatibility:
                #   urllib3 >= 1.26 uses allowed_methods
                #   urllib3 <= 1.25 used method_whitelist
                try:
                    retry = Retry(total=self.retry_times, connect=self.retry_times, read=self.retry_times,
                                  backoff_factor=self.retry_backoff,
                                  status_forcelist=[429, 500, 502, 503, 504],
                                  allowed_methods=frozenset(['GET']))
                except TypeError:
                    retry = Retry(total=self.retry_times, connect=self.retry_times, read=self.retry_times,
                                  backoff_factor=self.retry_backoff,
                                  status_forcelist=[429, 500, 502, 503, 504],
                                  method_whitelist=frozenset(['GET']))
                adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
                self.session.mount('https://', adapter)
                self.session.mount('http://', adapter)
        self.last_center = None
        self.latest_yaw = 0.0
        os.makedirs(self.cache_dir, exist_ok=True)
        self.bridge = CvBridge()
        self.pub_image = rospy.Publisher('/satellite_map/image', Image, queue_size=1, latch=True)
        self.sub_imu = rospy.Subscriber(self.imu_topic, Imu, self.imu_cb, queue_size=30)
        self.sub_gps = rospy.Subscriber(self.gps_topic, NavSatFix, self.gps_cb, queue_size=5)
        rospy.loginfo('satellite_tile_downloader v5_eval_fix waiting GPS on %s, IMU on %s', self.gps_topic, self.imu_topic)

    def imu_cb(self, msg):
        if tft is not None:
            self.latest_yaw = yaw_from_imu(msg)

    def enu_distance(self, lat1, lon1, lat2, lon2):
        lat0 = math.radians(lat1)
        dx = R_EARTH * math.cos(lat0) * math.radians(lon2-lon1)
        dy = R_EARTH * math.radians(lat2-lat1)
        return math.sqrt(dx*dx + dy*dy)

    def gps_cb(self, msg):
        if msg.status.status < 0:
            return
        if requests is None or PILImage is None or np is None:
            rospy.logerr('Install dependencies first: pip3 install requests pillow numpy')
            return
        if not self.tile_url_template:
            rospy.logerr('~tile_url_template is empty. Configure a licensed XYZ satellite/orthophoto tile server.')
            return
        lat, lon = msg.latitude, msg.longitude
        if self.last_center and not self.force_redownload:
            moved = self.enu_distance(self.last_center[0], self.last_center[1], lat, lon)
            if moved < self.reload_distance_m:
                return
        try:
            mosaic_path, yaml_path = self.download_mosaic(lat, lon, self.latest_yaw)
            self.last_center = (lat, lon)
            rospy.loginfo('High-resolution satellite mosaic saved: %s', mosaic_path)
            self.publish_preview(mosaic_path)
        except Exception as exc:
            rospy.logerr('Satellite tile download failed: %s', str(exc))

    def fetch_tile(self, x, y, z):
        cache_path = os.path.join(self.tile_cache_dir, '{}_{}_{}.jpg'.format(z, x, y))
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 512:
            try:
                return PILImage.open(cache_path).convert('RGB')
            except Exception:
                pass
        url = self.tile_url_template.format(x=x, y=y, z=z)
        headers = {'User-Agent': self.user_agent, 'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8'}
        last_exc = None
        for k in range(max(1, self.retry_times)):
            try:
                sess = self.session if self.session is not None else requests
                r = sess.get(url, timeout=(8.0, 30.0), headers=headers, verify=self.verify_ssl)
                r.raise_for_status()
                with open(cache_path, 'wb') as f:
                    f.write(r.content)
                return PILImage.open(BytesIO(r.content)).convert('RGB')
            except Exception as exc:
                last_exc = exc
                time.sleep(self.retry_backoff * (1.5 ** k))
        raise last_exc

    def download_mosaic(self, lat, lon, yaw):
        mpp = meters_per_pixel(lat, self.zoom)
        half_m = self.coverage_m if self.coverage_is_radius else self.coverage_m * 0.5
        half_px = half_m / mpp
        cx_tile, cy_tile = latlon_to_tile(lat, lon, self.zoom)
        cx_px, cy_px = cx_tile * TILE_SIZE, cy_tile * TILE_SIZE
        min_px_x, max_px_x = cx_px - half_px, cx_px + half_px
        min_px_y, max_px_y = cy_px - half_px, cy_px + half_px
        min_tx = int(math.floor(min_px_x / TILE_SIZE))
        max_tx = int(math.floor(max_px_x / TILE_SIZE))
        min_ty = int(math.floor(min_px_y / TILE_SIZE))
        max_ty = int(math.floor(max_px_y / TILE_SIZE))
        nx, ny = max_tx - min_tx + 1, max_ty - min_ty + 1
        tile_count = nx * ny
        if tile_count > self.max_tiles:
            raise RuntimeError('requested {} tiles at zoom {} exceeds max_tiles {}; lower zoom or coverage_m'.format(tile_count, self.zoom, self.max_tiles))

        rospy.loginfo('Downloading satellite map: center=(%.9f %.9f), coverage=%.1fm, zoom=%d, mpp=%.3f, tiles=%dx%d',
                      lat, lon, self.coverage_m, self.zoom, mpp, nx, ny)
        mosaic = PILImage.new('RGB', (nx * TILE_SIZE, ny * TILE_SIZE), color=(210, 210, 210))
        failed_tiles = 0
        for ix, tx in enumerate(range(min_tx, max_tx + 1)):
            for iy, ty in enumerate(range(min_ty, max_ty + 1)):
                try:
                    tile = self.fetch_tile(tx, ty, self.zoom)
                except Exception as exc:
                    failed_tiles += 1
                    rospy.logwarn('Tile z/x/y=%d/%d/%d failed after retry: %s', self.zoom, tx, ty, str(exc))
                    tile = PILImage.new('RGB', (TILE_SIZE, TILE_SIZE), color=(210, 210, 210))
                mosaic.paste(tile, (ix * TILE_SIZE, iy * TILE_SIZE))
                time.sleep(self.tile_sleep_sec)

        if failed_tiles > 0:
            rospy.logwarn('Satellite download finished with %d/%d failed tiles. If this is frequent, lower zoom, increase tile_sleep_sec, set a proxy, or use offline tiles.', failed_tiles, tile_count)

        local_cx = int(round(cx_px - min_tx * TILE_SIZE))
        local_cy = int(round(cy_px - min_ty * TILE_SIZE))
        half = int(round(half_px))
        crop = mosaic.crop((local_cx-half, local_cy-half, local_cx+half, local_cy+half))
        stamp = time.strftime('%Y%m%d_%H%M%S')
        mosaic_path = os.path.join(self.cache_dir, 'satellite_mosaic_z{}_{}m_{}.png'.format(self.zoom, int(self.coverage_m), stamp))
        yaml_path = os.path.join(self.cache_dir, 'satellite_mosaic.yml')
        crop.save(mosaic_path)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write('image_path: "{}"\n'.format(mosaic_path))
            f.write('center_lat: {:.12f}\n'.format(lat))
            f.write('center_lon: {:.12f}\n'.format(lon))
            f.write('center_yaw_rad: {:.9f}\n'.format(yaw))
            f.write('zoom: {}\n'.format(self.zoom))
            f.write('coverage_m: {:.3f}\n'.format(self.coverage_m))
            f.write('coverage_is_radius: {}\n'.format(str(self.coverage_is_radius).lower()))
            f.write('half_size_m: {:.3f}\n'.format(half_m))
            f.write('meters_per_pixel: {:.9f}\n'.format(mpp))
            f.write('width: {}\n'.format(crop.size[0]))
            f.write('height: {}\n'.format(crop.size[1]))
        rospy.loginfo('Satellite mosaic full resolution: %dx%d px, %.3f m/px', crop.size[0], crop.size[1], mpp)
        return mosaic_path, yaml_path

    def publish_preview(self, mosaic_path):
        img = PILImage.open(mosaic_path).convert('RGB')
        scale = max(0.02, min(1.0, self.publish_downsample_scale))
        if scale < 1.0:
            img = img.resize((max(1, int(img.size[0]*scale)), max(1, int(img.size[1]*scale))), PILImage.BILINEAR)
        cv_img = np.array(img)[:, :, ::-1].copy()
        ros_img = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        ros_img.header.stamp = rospy.Time.now()
        ros_img.header.frame_id = 'earth'
        self.pub_image.publish(ros_img)


if __name__ == '__main__':
    rospy.init_node('satellite_tile_downloader')
    SatelliteTileDownloader()
    rospy.spin()
