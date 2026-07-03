#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>

namespace
{

enum SemanticLabel : uint32_t
{
  LABEL_UNKNOWN = 0,
  LABEL_ROAD = 1,
  LABEL_SIDEWALK = 2,
  LABEL_BUILDING = 3,
  LABEL_VEGETATION = 4,
  LABEL_DYNAMIC = 5,
  LABEL_OTHER = 6
};

struct GridKey
{
  int x = 0;
  int y = 0;
  bool operator==(const GridKey& rhs) const
  {
    return x == rhs.x && y == rhs.y;
  }
};

struct GridKeyHash
{
  std::size_t operator()(const GridKey& k) const
  {
    const std::size_t hx = std::hash<int>()(k.x);
    const std::size_t hy = std::hash<int>()(k.y);
    return hx ^ (hy + 0x9e3779b97f4a7c15ULL + (hx << 6) + (hx >> 2));
  }
};

struct GridState
{
  float z_min = std::numeric_limits<float>::infinity();
  float z_max = -std::numeric_limits<float>::infinity();
  int count = 0;
};

struct PointXYZI
{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float intensity = 0.0f;
};

struct LabeledPoint
{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  uint32_t label = LABEL_UNKNOWN;
  float confidence = 0.0f;
};

const sensor_msgs::PointField* findField(const sensor_msgs::PointCloud2& msg, const std::string& name)
{
  for (const auto& field : msg.fields)
  {
    if (field.name == name)
    {
      return &field;
    }
  }
  return nullptr;
}

double readFieldAsDouble(const sensor_msgs::PointCloud2& msg,
                         const sensor_msgs::PointField& field,
                         const std::size_t point_index)
{
  const uint8_t* ptr = &msg.data[point_index * msg.point_step + field.offset];
  switch (field.datatype)
  {
    case sensor_msgs::PointField::INT8:
    {
      int8_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::UINT8:
    {
      uint8_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::INT16:
    {
      int16_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::UINT16:
    {
      uint16_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::INT32:
    {
      int32_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::UINT32:
    {
      uint32_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::FLOAT32:
    {
      float v = 0.0f;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<double>(v);
    }
    case sensor_msgs::PointField::FLOAT64:
    {
      double v = 0.0;
      std::memcpy(&v, ptr, sizeof(v));
      return v;
    }
    default:
      return std::numeric_limits<double>::quiet_NaN();
  }
}

float rgbFloat(uint8_t r, uint8_t g, uint8_t b)
{
  const uint32_t rgb = (static_cast<uint32_t>(r) << 16) |
                       (static_cast<uint32_t>(g) << 8) |
                       static_cast<uint32_t>(b);
  float out = 0.0f;
  std::memcpy(&out, &rgb, sizeof(out));
  return out;
}

float labelRgbFloat(uint32_t label)
{
  switch (label)
  {
    case LABEL_ROAD:
      return rgbFloat(70, 70, 70);
    case LABEL_SIDEWALK:
      return rgbFloat(160, 120, 80);
    case LABEL_BUILDING:
      return rgbFloat(220, 40, 40);
    case LABEL_VEGETATION:
      return rgbFloat(20, 170, 40);
    case LABEL_DYNAMIC:
      return rgbFloat(255, 180, 0);
    case LABEL_OTHER:
      return rgbFloat(120, 120, 220);
    default:
      return rgbFloat(180, 180, 180);
  }
}

}  // namespace

class RangeNetGeometryAdapter
{
 public:
  RangeNetGeometryAdapter() : nh_(), pnh_("~")
  {
    pnh_.param<std::string>("input_cloud_topic", input_cloud_topic_, "/cloud_registered");
    pnh_.param<std::string>("output_cloud_topic", output_cloud_topic_, "/rangenet/semantic_points");
    pnh_.param<double>("grid_resolution", grid_resolution_, 0.60);
    pnh_.param<double>("min_range_m", min_range_m_, 0.5);
    pnh_.param<double>("max_range_m", max_range_m_, 120.0);
    pnh_.param<double>("z_min", z_min_, -5.0);
    pnh_.param<double>("z_max", z_max_, 25.0);
    pnh_.param<int>("point_stride", point_stride_, 1);
    pnh_.param<int>("max_points", max_points_, 250000);
    pnh_.param<double>("road_max_height_above_ground", road_max_height_, 0.28);
    pnh_.param<double>("sidewalk_max_height_above_ground", sidewalk_max_height_, 0.55);
    pnh_.param<double>("building_min_height_above_ground", building_min_height_, 0.75);
    pnh_.param<double>("vegetation_min_height_above_ground", vegetation_min_height_, 0.40);
    pnh_.param<double>("structure_min_z_span", structure_min_z_span_, 1.00);
    pnh_.param<double>("flat_max_z_span", flat_max_z_span_, 0.35);
    pnh_.param<double>("intensity_vegetation_threshold", intensity_vegetation_threshold_, 0.0);
    pnh_.param<bool>("use_intensity_hint", use_intensity_hint_, false);
    pnh_.param<bool>("drop_unknown", drop_unknown_, true);

    sub_ = nh_.subscribe(input_cloud_topic_, 2, &RangeNetGeometryAdapter::cloudCb, this);
    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_cloud_topic_, 2);
    diagnostic_timer_ = nh_.createWallTimer(ros::WallDuration(3.0), &RangeNetGeometryAdapter::diagnosticCb, this);

    ROS_WARN("rangenet_geometry_adapter is a local compatibility fallback, not real RangeNet++. "
             "Replace it with RangeNet++ when the model/runtime is installed.");
    ROS_INFO("rangenet_geometry_adapter started input=%s output=%s grid=%.2f",
             input_cloud_topic_.c_str(), output_cloud_topic_.c_str(), grid_resolution_);
  }

 private:
  GridKey keyFor(const PointXYZI& p) const
  {
    const double inv = 1.0 / std::max(0.05, grid_resolution_);
    return GridKey{
      static_cast<int>(std::floor(static_cast<double>(p.x) * inv)),
      static_cast<int>(std::floor(static_cast<double>(p.y) * inv))
    };
  }

  bool readCloud(const sensor_msgs::PointCloud2& msg, std::vector<PointXYZI>& points) const
  {
    const sensor_msgs::PointField* fx = findField(msg, "x");
    const sensor_msgs::PointField* fy = findField(msg, "y");
    const sensor_msgs::PointField* fz = findField(msg, "z");
    const sensor_msgs::PointField* fi = findField(msg, "intensity");
    if (!fx || !fy || !fz)
    {
      ROS_WARN_THROTTLE(2.0, "Input cloud lacks x/y/z fields");
      return false;
    }

    const std::size_t n = static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
    const int stride = std::max(1, point_stride_);
    points.clear();
    points.reserve(std::min<std::size_t>(n, static_cast<std::size_t>(std::max(1, max_points_))));
    for (std::size_t i = 0; i < n; i += static_cast<std::size_t>(stride))
    {
      const double x = readFieldAsDouble(msg, *fx, i);
      const double y = readFieldAsDouble(msg, *fy, i);
      const double z = readFieldAsDouble(msg, *fz, i);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
      {
        continue;
      }
      const double range = std::sqrt(x * x + y * y + z * z);
      if (range < min_range_m_ || range > max_range_m_ || z < z_min_ || z > z_max_)
      {
        continue;
      }
      PointXYZI p;
      p.x = static_cast<float>(x);
      p.y = static_cast<float>(y);
      p.z = static_cast<float>(z);
      p.intensity = fi ? static_cast<float>(readFieldAsDouble(msg, *fi, i)) : 0.0f;
      points.push_back(p);
      if (max_points_ > 0 && static_cast<int>(points.size()) >= max_points_)
      {
        break;
      }
    }
    return !points.empty();
  }

  std::unordered_map<GridKey, GridState, GridKeyHash> buildGrid(const std::vector<PointXYZI>& points) const
  {
    std::unordered_map<GridKey, GridState, GridKeyHash> grid;
    grid.reserve(points.size() / 4 + 1);
    for (const auto& p : points)
    {
      GridState& s = grid[keyFor(p)];
      s.z_min = std::min(s.z_min, p.z);
      s.z_max = std::max(s.z_max, p.z);
      s.count += 1;
    }
    return grid;
  }

  LabeledPoint classify(const PointXYZI& p,
                        const GridState& s,
                        const GridState* neighbor_state) const
  {
    const float ground = std::min(s.z_min, neighbor_state ? neighbor_state->z_min : s.z_min);
    const float span = std::max(s.z_max - s.z_min,
                                neighbor_state ? neighbor_state->z_max - neighbor_state->z_min : 0.0f);
    const float dz = p.z - ground;

    LabeledPoint out;
    out.x = p.x;
    out.y = p.y;
    out.z = p.z;
    out.label = LABEL_OTHER;
    out.confidence = 0.45f;

    if (dz <= road_max_height_ && span <= flat_max_z_span_)
    {
      out.label = LABEL_ROAD;
      out.confidence = 0.72f;
      return out;
    }

    if (dz <= sidewalk_max_height_ && span <= flat_max_z_span_ * 1.8)
    {
      out.label = LABEL_SIDEWALK;
      out.confidence = 0.58f;
      return out;
    }

    if (dz >= building_min_height_ && span >= structure_min_z_span_)
    {
      out.label = LABEL_BUILDING;
      out.confidence = 0.62f;
      return out;
    }

    if (dz >= vegetation_min_height_)
    {
      out.label = LABEL_VEGETATION;
      out.confidence = 0.50f;
      if (use_intensity_hint_ && p.intensity > intensity_vegetation_threshold_)
      {
        out.confidence = 0.58f;
      }
      return out;
    }

    if (drop_unknown_)
    {
      out.label = LABEL_UNKNOWN;
      out.confidence = 0.0f;
    }
    return out;
  }

  const GridState* neighborState(const std::unordered_map<GridKey, GridState, GridKeyHash>& grid,
                                 const GridKey& k) const
  {
    const std::array<GridKey, 9> keys = {{
      {k.x, k.y}, {k.x + 1, k.y}, {k.x - 1, k.y}, {k.x, k.y + 1}, {k.x, k.y - 1},
      {k.x + 1, k.y + 1}, {k.x + 1, k.y - 1}, {k.x - 1, k.y + 1}, {k.x - 1, k.y - 1}
    }};
    const GridState* best = nullptr;
    for (const auto& nk : keys)
    {
      auto it = grid.find(nk);
      if (it == grid.end())
      {
        continue;
      }
      if (!best || it->second.z_min < best->z_min)
      {
        best = &it->second;
      }
    }
    return best;
  }

  void publishLabeledCloud(const std::vector<LabeledPoint>& points, const std_msgs::Header& header)
  {
    sensor_msgs::PointCloud2 msg;
    msg.header = header;
    msg.height = 1;
    msg.width = static_cast<uint32_t>(points.size());
    sensor_msgs::PointCloud2Modifier modifier(msg);
    modifier.setPointCloud2Fields(
      6,
      "x", 1, sensor_msgs::PointField::FLOAT32,
      "y", 1, sensor_msgs::PointField::FLOAT32,
      "z", 1, sensor_msgs::PointField::FLOAT32,
      "rgb", 1, sensor_msgs::PointField::FLOAT32,
      "label", 1, sensor_msgs::PointField::UINT32,
      "confidence", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(points.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_rgb(msg, "rgb");
    sensor_msgs::PointCloud2Iterator<uint32_t> iter_label(msg, "label");
    sensor_msgs::PointCloud2Iterator<float> iter_conf(msg, "confidence");
    for (const auto& p : points)
    {
      *iter_x = p.x;
      *iter_y = p.y;
      *iter_z = p.z;
      *iter_rgb = labelRgbFloat(p.label);
      *iter_label = p.label;
      *iter_conf = p.confidence;
      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_rgb;
      ++iter_label;
      ++iter_conf;
    }
    pub_.publish(msg);
  }

  void cloudCb(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    ++input_count_;
    last_input_wall_ = ros::WallTime::now();

    std::vector<PointXYZI> points;
    if (!readCloud(*msg, points))
    {
      return;
    }

    const auto grid = buildGrid(points);
    std::vector<LabeledPoint> labeled;
    labeled.reserve(points.size());
    std::array<int, 7> counts;
    counts.fill(0);
    for (const auto& p : points)
    {
      const GridKey k = keyFor(p);
      auto it = grid.find(k);
      if (it == grid.end())
      {
        continue;
      }
      const GridState* ns = neighborState(grid, k);
      LabeledPoint lp = classify(p, it->second, ns);
      if (lp.label == LABEL_UNKNOWN && drop_unknown_)
      {
        continue;
      }
      if (lp.label < counts.size())
      {
        counts[lp.label] += 1;
      }
      labeled.push_back(lp);
    }

    publishLabeledCloud(labeled, msg->header);
    ROS_INFO_THROTTLE(3.0,
                      "rangenet geometry adapter labeled=%zu road=%d sidewalk=%d building=%d vegetation=%d other=%d",
                      labeled.size(), counts[LABEL_ROAD], counts[LABEL_SIDEWALK],
                      counts[LABEL_BUILDING], counts[LABEL_VEGETATION], counts[LABEL_OTHER]);
  }

  void diagnosticCb(const ros::WallTimerEvent&)
  {
    const ros::WallTime now = ros::WallTime::now();
    if (input_count_ == 0)
    {
      ROS_WARN("rangenet_geometry_adapter has not received input PointCloud2 on %s", input_cloud_topic_.c_str());
    }
    else if ((now - last_input_wall_).toSec() > 3.0)
    {
      ROS_WARN("rangenet_geometry_adapter input stalled: topic=%s last_age=%.1fs",
               input_cloud_topic_.c_str(), (now - last_input_wall_).toSec());
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  ros::WallTimer diagnostic_timer_;

  std::string input_cloud_topic_;
  std::string output_cloud_topic_;
  double grid_resolution_ = 0.60;
  double min_range_m_ = 0.5;
  double max_range_m_ = 120.0;
  double z_min_ = -5.0;
  double z_max_ = 25.0;
  double road_max_height_ = 0.28;
  double sidewalk_max_height_ = 0.55;
  double building_min_height_ = 0.75;
  double vegetation_min_height_ = 0.40;
  double structure_min_z_span_ = 1.00;
  double flat_max_z_span_ = 0.35;
  double intensity_vegetation_threshold_ = 0.0;
  bool use_intensity_hint_ = false;
  bool drop_unknown_ = true;
  int point_stride_ = 1;
  int max_points_ = 250000;
  int input_count_ = 0;
  ros::WallTime last_input_wall_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rangenet_geometry_adapter");
  RangeNetGeometryAdapter node;
  ros::spin();
  return 0;
}
