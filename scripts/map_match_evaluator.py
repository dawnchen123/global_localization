#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate /map_match_pose against realtime /fix in the same local ENU frame.

Default convention:
  origin_mode = satellite_center
  ENU origin = center_lat / center_lon in satellite_mosaic.yml

Published topics:
  /map_match_error          std_msgs/Float64, horizontal error norm in meters
  /map_match_error_vector   geometry_msgs/Vector3Stamped, x/y errors and norm
  /map_match_fix            sensor_msgs/NavSatFix, map_match_pose converted back to WGS84

CSV columns:
  stamp,gps_lat,gps_lon,gps_x,gps_y,map_x,map_y,err_x,err_y,err_norm,gps_std,time_diff
"""
import math
import os
import csv
from collections import deque

import rospy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Vector3Stamped
from std_msgs.msg import Float64

R_EARTH = 6378137.0


def _read_simple_yaml_value(path, key):
    if not os.path.exists(path):
        return None
    target = key + ':'
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith(target):
                val = line[len(target):].strip().strip('"\'')
                return val
    return None


def _stamp_to_float(stamp):
    try:
        if stamp and stamp.to_sec() > 1e-6:
            return stamp.to_sec()
    except Exception:
        pass
    return rospy.Time.now().to_sec()


class MapMatchEvaluator:
    def __init__(self):
        self.gps_topic = rospy.get_param('~gps_topic', '/fix')
        self.map_topic = rospy.get_param('~map_match_pose_topic', '/map_match_pose')
        self.satellite_yaml = rospy.get_param('~satellite_yaml', '/tmp/fast_livo2_global_localization/satellite_mosaic.yml')
        self.origin_mode = rospy.get_param('~origin_mode', 'satellite_center')  # satellite_center or first_gps
        self.max_time_diff = float(rospy.get_param('~max_time_diff', 3.0))
        self.use_latest_gps_if_unsync = bool(rospy.get_param('~use_latest_gps_if_unsync', True))
        self.max_gps_std = float(rospy.get_param('~max_gps_std', 10.0))
        self.csv_path = rospy.get_param('~csv_path', '/tmp/fast_livo2_global_localization/map_match_error.csv')
        self.queue_size = int(rospy.get_param('~queue_size', 200))
        self.print_every_n = int(rospy.get_param('~print_every_n', 10))
        self.min_fix_status = int(rospy.get_param('~min_fix_status', NavSatStatus.STATUS_FIX))

        self.origin_lat = None
        self.origin_lon = None
        self.gps_buf = deque(maxlen=self.queue_size)
        self.n = 0
        self.sum_e2 = 0.0
        self.sum_e = 0.0
        self.max_e = 0.0

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        write_header = not os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if write_header:
            self.csv_writer.writerow(['stamp','gps_lat','gps_lon','gps_x','gps_y','map_x','map_y','err_x','err_y','err_norm','gps_std','time_diff'])
            self.csv_file.flush()

        if self.origin_mode == 'satellite_center':
            lat_s = _read_simple_yaml_value(self.satellite_yaml, 'center_lat')
            lon_s = _read_simple_yaml_value(self.satellite_yaml, 'center_lon')
            if lat_s is not None and lon_s is not None:
                self.origin_lat = float(lat_s)
                self.origin_lon = float(lon_s)
                rospy.loginfo('MapMatchEvaluator ENU origin from satellite center: lat %.12f lon %.12f', self.origin_lat, self.origin_lon)
            else:
                rospy.logwarn('Cannot read satellite center from %s. Will fall back to first GPS fix.', self.satellite_yaml)
                self.origin_mode = 'first_gps'

        self.pub_err = rospy.Publisher('/map_match_error', Float64, queue_size=20)
        self.pub_vec = rospy.Publisher('/map_match_error_vector', Vector3Stamped, queue_size=20)
        self.pub_match_fix = rospy.Publisher('/map_match_fix', NavSatFix, queue_size=20)

        self.sub_gps = rospy.Subscriber(self.gps_topic, NavSatFix, self.gps_cb, queue_size=50)
        self.sub_match = rospy.Subscriber(self.map_topic, PoseWithCovarianceStamped, self.match_cb, queue_size=20)

        rospy.loginfo('MapMatchEvaluator started. gps=%s map_match=%s origin_mode=%s csv=%s', self.gps_topic, self.map_topic, self.origin_mode, self.csv_path)

    def gps_cb(self, msg):
        if msg.status.status < self.min_fix_status:
            return
        gps_std = self.gps_std(msg)
        if gps_std > self.max_gps_std:
            rospy.logwarn_throttle(2.0, 'GPS std %.2f exceeds max_gps_std %.2f, ignore this fix.', gps_std, self.max_gps_std)
            return
        if self.origin_lat is None or self.origin_lon is None:
            self.origin_lat = msg.latitude
            self.origin_lon = msg.longitude
            rospy.loginfo('MapMatchEvaluator ENU origin from first GPS: lat %.12f lon %.12f', self.origin_lat, self.origin_lon)
        stamp = _stamp_to_float(msg.header.stamp)
        self.gps_buf.append((stamp, msg, gps_std))

    def match_cb(self, msg):
        if self.origin_lat is None or self.origin_lon is None:
            rospy.logwarn_throttle(2.0, 'Waiting ENU origin and GPS fix before evaluating map_match_pose.')
            return
        if not self.gps_buf:
            rospy.logwarn_throttle(2.0, 'No valid GPS fix buffered yet.')
            return
        t = _stamp_to_float(msg.header.stamp)
        best = min(self.gps_buf, key=lambda item: abs(item[0] - t))
        dt = abs(best[0] - t)
        if dt > self.max_time_diff:
            if not self.use_latest_gps_if_unsync:
                rospy.logwarn_throttle(1.0, 'No synchronized GPS for map_match_pose: time diff %.3fs > %.3fs', dt, self.max_time_diff)
                return
            rospy.logwarn_throttle(2.0, 'Use nearest GPS although time diff %.3fs > %.3fs. For bag replay, consider increasing _max_time_diff.', dt, self.max_time_diff)
        _, gps, gps_std = best
        gps_x, gps_y = self.latlon_to_enu(gps.latitude, gps.longitude)
        map_x = msg.pose.pose.position.x
        map_y = msg.pose.pose.position.y
        if not all(math.isfinite(v) for v in [gps_x, gps_y, map_x, map_y]):
            return
        err_x = map_x - gps_x
        err_y = map_y - gps_y
        err = math.sqrt(err_x * err_x + err_y * err_y)

        self.n += 1
        self.sum_e += err
        self.sum_e2 += err * err
        self.max_e = max(self.max_e, err)
        rmse = math.sqrt(self.sum_e2 / max(1, self.n))
        mean = self.sum_e / max(1, self.n)

        self.pub_err.publish(Float64(err))
        v = Vector3Stamped()
        v.header.stamp = msg.header.stamp
        v.header.frame_id = msg.header.frame_id or 'global_enu'
        v.vector.x = err_x
        v.vector.y = err_y
        v.vector.z = err
        self.pub_vec.publish(v)
        self.pub_match_fix.publish(self.enu_to_navsat(map_x, map_y, msg.header.stamp))

        self.csv_writer.writerow(['{:.6f}'.format(t),
                                  '{:.12f}'.format(gps.latitude), '{:.12f}'.format(gps.longitude),
                                  '{:.6f}'.format(gps_x), '{:.6f}'.format(gps_y),
                                  '{:.6f}'.format(map_x), '{:.6f}'.format(map_y),
                                  '{:.6f}'.format(err_x), '{:.6f}'.format(err_y), '{:.6f}'.format(err),
                                  '{:.6f}'.format(gps_std), '{:.6f}'.format(dt)])
        self.csv_file.flush()

        if self.n % self.print_every_n == 0 or self.n == 1:
            rospy.loginfo('map_match error: current=%.3fm mean=%.3fm rmse=%.3fm max=%.3fm n=%d dt=%.3fs',
                          err, mean, rmse, self.max_e, self.n, dt)

    def gps_std(self, msg):
        if msg.position_covariance_type == NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            return 999.0 if self.max_gps_std < 999.0 else 20.0
        cov_x = max(0.0, msg.position_covariance[0])
        cov_y = max(0.0, msg.position_covariance[4])
        return math.sqrt(0.5 * (cov_x + cov_y)) if (cov_x + cov_y) > 1e-12 else 0.0

    def latlon_to_enu(self, lat, lon):
        lat0 = math.radians(self.origin_lat)
        dlat = math.radians(lat - self.origin_lat)
        dlon = math.radians(lon - self.origin_lon)
        x = R_EARTH * math.cos(lat0) * dlon
        y = R_EARTH * dlat
        return x, y

    def enu_to_latlon(self, x, y):
        lat0 = math.radians(self.origin_lat)
        lat = self.origin_lat + math.degrees(y / R_EARTH)
        lon = self.origin_lon + math.degrees(x / (R_EARTH * math.cos(lat0)))
        return lat, lon

    def enu_to_navsat(self, x, y, stamp):
        lat, lon = self.enu_to_latlon(x, y)
        msg = NavSatFix()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map_match_wgs84'
        msg.status.status = NavSatStatus.STATUS_FIX
        msg.status.service = NavSatStatus.SERVICE_GPS
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = 0.0
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        msg.position_covariance[0] = 4.0
        msg.position_covariance[4] = 4.0
        msg.position_covariance[8] = 100.0
        return msg

    def shutdown(self):
        try:
            self.csv_file.close()
        except Exception:
            pass


if __name__ == '__main__':
    rospy.init_node('map_match_evaluator')
    node = MapMatchEvaluator()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()
