/**
 * @file      PointCloudUtils.h
 * @brief     Native C++ point cloud utilities to replace PCL dependency
 * @author    Seungwon Choi
 * @date      2025-10-03
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>
#include <map>
#include <filesystem>
#include <Eigen/Dense>
#include <nanoflann.hpp>

namespace lidar_slam {
namespace util {

// ===== Point Types =====

/**
 * @brief Basic 3D point structure
 */
struct Point3D {
    float x, y, z;
    
    Point3D() : x(0.0f), y(0.0f), z(0.0f) {}
    Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    // Vector operations
    Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }
    
    Point3D operator-(const Point3D& other) const {
        return Point3D(x - other.x, y - other.y, z - other.z);
    }
    
    Point3D operator*(float scalar) const {
        return Point3D(x * scalar, y * scalar, z * scalar);
    }
    
    // Distance calculations
    float distance_to(const Point3D& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    
    float squared_distance_to(const Point3D& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return dx*dx + dy*dy + dz*dz;
    }
    
    // Norm calculations
    float norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    float squared_norm() const {
        return x*x + y*y + z*z;
    }
    
    // Conversion to/from Eigen
    Eigen::Vector3f to_eigen() const {
        return Eigen::Vector3f(x, y, z);
    }
    
    static Point3D from_eigen(const Eigen::Vector3f& vec) {
        return Point3D(vec.x(), vec.y(), vec.z());
    }
};

/**
 * @brief Point cloud container class
 */
class PointCloud {
public:
    using Point = Point3D;
    using Ptr = std::shared_ptr<PointCloud>;
    using ConstPtr = std::shared_ptr<const PointCloud>;
    
    // ===== Constructors =====
    PointCloud() = default;
    explicit PointCloud(size_t reserve_size) {
        points.reserve(reserve_size);
    }
    
    // ===== Basic Operations =====
    
    /**
     * @brief Add a point to the cloud
     */
    void push_back(const Point3D& point) {
        points.push_back(point);
    }
    
    /**
     * @brief Add a point to the cloud
     */
    void push_back(float x, float y, float z) {
        points.emplace_back(x, y, z);
    }
    
    /**
     * @brief Get point by index
     */
    const Point3D& operator[](size_t index) const {
        return points[index];
    }
    
    Point3D& operator[](size_t index) {
        return points[index];
    }
    
    /**
     * @brief Get point by index with bounds checking
     */
    const Point3D& at(size_t index) const {
        return points.at(index);
    }
    
    Point3D& at(size_t index) {
        return points.at(index);
    }
    
    /**
     * @brief Get number of points
     */
    size_t size() const {
        return points.size();
    }
    
    /**
     * @brief Check if cloud is empty
     */
    bool empty() const {
        return points.empty();
    }
    
    /**
     * @brief Clear all points
     */
    void clear() {
        points.clear();
    }
    
    /**
     * @brief Reserve memory for points
     */
    void reserve(size_t size) {
        points.reserve(size);
    }
    
    /**
     * @brief Resize the point cloud
     */
    void resize(size_t size) {
        points.resize(size);
    }
    
    // ===== Utility Methods =====
    
    /**
     * @brief Get bounding box of the point cloud
     */
    struct BoundingBox {
        Point3D min_point, max_point;
        bool is_valid = false;
    };
    
    BoundingBox get_bounding_box() const {
        BoundingBox bbox;
        if (points.empty()) {
            return bbox;
        }
        
        bbox.min_point = bbox.max_point = points[0];
        bbox.is_valid = true;
        
        for (const auto& point : points) {
            bbox.min_point.x = std::min(bbox.min_point.x, point.x);
            bbox.min_point.y = std::min(bbox.min_point.y, point.y);
            bbox.min_point.z = std::min(bbox.min_point.z, point.z);
            
            bbox.max_point.x = std::max(bbox.max_point.x, point.x);
            bbox.max_point.y = std::max(bbox.max_point.y, point.y);
            bbox.max_point.z = std::max(bbox.max_point.z, point.z);
        }
        
        return bbox;
    }
    
    /**
     * @brief Get centroid of the point cloud
     */
    Point3D get_centroid() const {
        if (points.empty()) {
            return Point3D();
        }
        
        Point3D centroid;
        for (const auto& point : points) {
            centroid.x += point.x;
            centroid.y += point.y;
            centroid.z += point.z;
        }
        
        float inv_size = 1.0f / static_cast<float>(points.size());
        centroid.x *= inv_size;
        centroid.y *= inv_size;
        centroid.z *= inv_size;
        
        return centroid;
    }
    
    /**
     * @brief Transform all points by a 4x4 transformation matrix
     */
    void transform(const Eigen::Matrix4f& transformation) {
        for (auto& point : points) {
            Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1.0f);
            Eigen::Vector4f transformed = transformation * homogeneous_point;
            
            point.x = transformed.x();
            point.y = transformed.y();
            point.z = transformed.z();
        }
    }
    
    /**
     * @brief Create a transformed copy of the point cloud
     */
    PointCloud::Ptr transformed_copy(const Eigen::Matrix4f& transformation) const {
        auto result = std::make_shared<PointCloud>();
        result->points.reserve(points.size());
        
        for (const auto& point : points) {
            Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1.0f);
            Eigen::Vector4f transformed = transformation * homogeneous_point;
            
            result->push_back(transformed.x(), transformed.y(), transformed.z());
        }
        
        return result;
    }
    
    /**
     * @brief Copy constructor helper
     */
    PointCloud::Ptr copy() const {
        auto result = std::make_shared<PointCloud>();
        result->points = this->points;
        return result;
    }
    
    /**
     * @brief Append another point cloud to this one
     */
    PointCloud& operator+=(const PointCloud& other) {
        points.reserve(points.size() + other.points.size());
        for (const auto& point : other.points) {
            points.push_back(point);
        }
        return *this;
    }
    
    /**
     * @brief Iterator access
     */
    auto begin() const { return points.begin(); }
    auto end() const { return points.end(); }
    auto begin() { return points.begin(); }
    auto end() { return points.end(); }

private:
    std::vector<Point3D> points;
};

// ===== Type Aliases for Compatibility =====

using PointType = Point3D;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;

// ===== Utility Functions =====

/**
 * @brief Load KITTI binary point cloud file
 * @param filename Path to .bin file
 * @return Pointer to loaded point cloud
 */
PointCloud::Ptr load_kitti_binary(const std::string& filename);

/**
 * @brief Save point cloud to KITTI binary format
 * @param cloud Point cloud to save
 * @param filename Output file path
 * @return True if successful
 */
bool save_kitti_binary(const PointCloud::ConstPtr& cloud, const std::string& filename);

/**
 * @brief Transform point cloud with transformation matrix
 * @param input Input point cloud
 * @param output Output point cloud
 * @param transformation 4x4 transformation matrix
 */
void transform_point_cloud(const PointCloud::ConstPtr& input,
                          PointCloud::Ptr& output,
                          const Eigen::Matrix4f& transformation);

/**
 * @brief Copy point cloud
 * @param input Input point cloud
 * @param output Output point cloud
 */
void copy_point_cloud(const PointCloud::ConstPtr& input, PointCloud::Ptr& output);

/**
 * @brief Save point cloud to PLY binary format
 * @param filename Output file path
 * @param cloud Point cloud to save
 * @return True if successful
 */
bool save_point_cloud_ply(const std::string& filename, const PointCloud::ConstPtr& cloud);


/**
 * @brief Save point cloud to PCD binary format
 * @param filename Output file path
 * @param cloud Point cloud to save
 * @return True if successful
 */
bool save_point_cloud_pcd(const std::string& filename, const PointCloud::ConstPtr& cloud);
// ===== Spatial Data Structures and Filters =====

/**
 * @brief Point cloud adapter for nanoflann
 */
struct PointCloudAdapter {
    const PointCloud& cloud;
    
    explicit PointCloudAdapter(const PointCloud& cloud_) : cloud(cloud_) {}
    
    inline size_t kdtree_get_point_count() const { return cloud.size(); }
    
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        const Point3D& point = cloud.at(idx);
        switch (dim) {
            case 0: return point.x;
            case 1: return point.y; 
            case 2: return point.z;
            default: return 0.0f;
        }
    }
    
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

/**
 * @brief KD-Tree for fast nearest neighbor search using nanoflann
 */
class KdTree {
public:
    using TreeType = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloudAdapter>,
        PointCloudAdapter,
        3
    >;
    
    KdTree() : tree_(nullptr) {}
    ~KdTree() = default;
    
    /**
     * @brief Set input point cloud and build tree
     */
    void setInputCloud(const PointCloud::ConstPtr& cloud) {
        if (!cloud || cloud->empty()) {
            tree_.reset();
            return;
        }
        
        adapter_ = std::make_unique<PointCloudAdapter>(*cloud);
        tree_ = std::make_unique<TreeType>(3, *adapter_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree_->buildIndex();
    }
    
    /**
     * @brief Find k nearest neighbors
     */
    int nearestKSearch(const Point3D& query_point, int k, 
                      std::vector<int>& indices, std::vector<float>& distances) const {
        if (!tree_ || k <= 0) {
            indices.clear();
            distances.clear();
            return 0;
        }
        
        indices.resize(k);
        distances.resize(k);
        
        float query[3] = {query_point.x, query_point.y, query_point.z};
        
        std::vector<uint32_t> nanoflann_indices(k);
        size_t num_results = tree_->knnSearch(query, k, nanoflann_indices.data(), distances.data());
        
        // Convert uint32_t indices to int
        for (size_t i = 0; i < num_results; ++i) {
            indices[i] = static_cast<int>(nanoflann_indices[i]);
        }
        
        indices.resize(num_results);
        distances.resize(num_results);
        
        return static_cast<int>(num_results);
    }
    
    /**
     * @brief Find neighbors within radius
     */
    int radiusSearch(const Point3D& query_point, float radius,
                    std::vector<int>& indices, std::vector<float>& distances) const {
        if (!tree_ || radius <= 0) {
            indices.clear();
            distances.clear();
            return 0;
        }
        
        float query[3] = {query_point.x, query_point.y, query_point.z};
        
        std::vector<nanoflann::ResultItem<uint32_t, float>> matches;
        nanoflann::SearchParameters params;
        
        size_t num_results = tree_->radiusSearch(query, radius * radius, matches, params);
        
        indices.resize(num_results);
        distances.resize(num_results);
        
        for (size_t i = 0; i < num_results; ++i) {
            indices[i] = static_cast<int>(matches[i].first);
            distances[i] = std::sqrt(matches[i].second); // Convert squared distance to distance
        }
        
        return static_cast<int>(num_results);
    }
    
private:
    std::unique_ptr<PointCloudAdapter> adapter_;
    std::unique_ptr<TreeType> tree_;
};

/**
 * @brief Voxel grid downsampling filter
 */
class VoxelGrid {
public:
    VoxelGrid() : leaf_size_(0.01f) {}
    
    void setLeafSize(float size) { leaf_size_ = size; }
    
    void setInputCloud(const PointCloud::ConstPtr& cloud) { input_cloud_ = cloud; }
    
    void filter(PointCloud& output) {
        if (!input_cloud_ || input_cloud_->empty() || leaf_size_ <= 0) {
            output.clear();
            return;
        }
        
        // Use map to store weighted centroids for each voxel
        std::map<VoxelKey, WeightedCentroid> voxel_map;
        
        // Process points one by one with weighted averaging
        for (size_t i = 0; i < input_cloud_->size(); ++i) {
            const Point3D& point = input_cloud_->at(i);
            VoxelKey voxel_key = get_voxel_key(point);
            
            auto& weighted_centroid = voxel_map[voxel_key];
            weighted_centroid.add_point(point);
        }
        
        output.clear();
        output.reserve(voxel_map.size());
        
        // Extract final centroids from each voxel
        for (const auto& voxel : voxel_map) {
            output.push_back(voxel.second.get_centroid());
        }
    }
    
private:
    struct WeightedCentroid {
        Point3D centroid;
        float weight;
        
        WeightedCentroid() : centroid(0, 0, 0), weight(0.0f) {}
        
        void add_point(const Point3D& new_point) {
            if (weight == 0.0f) {
                // First point: just copy it
                centroid = new_point;
                weight = 1.0f;
            } else {
                // Weighted average: weight/(weight+1) * centroid + 1/(weight+1) * new_point
                float total_weight = weight + 1.0f;
                float old_ratio = weight / total_weight;
                float new_ratio = 1.0f / total_weight;
                
                centroid.x = old_ratio * centroid.x + new_ratio * new_point.x;
                centroid.y = old_ratio * centroid.y + new_ratio * new_point.y;
                centroid.z = old_ratio * centroid.z + new_ratio * new_point.z;
                
                weight = total_weight;
            }
        }
        
        Point3D get_centroid() const {
            return centroid;
        }
        
        float get_weight() const {
            return weight;
        }
    };
    
    struct VoxelKey {
        int x, y, z;
        
        VoxelKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
        
        bool operator<(const VoxelKey& other) const {
            if (x != other.x) return x < other.x;
            if (y != other.y) return y < other.y;
            return z < other.z;
        }
        
        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    
    VoxelKey get_voxel_key(const Point3D& point) const {
        int vx = static_cast<int>(std::floor(point.x / leaf_size_));
        int vy = static_cast<int>(std::floor(point.y / leaf_size_));
        int vz = static_cast<int>(std::floor(point.z / leaf_size_));
        return VoxelKey(vx, vy, vz);
    }
    
    float leaf_size_;
    PointCloud::ConstPtr input_cloud_;
};

/**
 * @brief Crop box filter for spatial filtering
 */
class CropBox {
public:
    CropBox() : min_pt_(-1, -1, -1), max_pt_(1, 1, 1), negative_(false) {}
    
    void setMin(const Eigen::Vector4f& min_pt) {
        min_pt_ = Point3D(min_pt[0], min_pt[1], min_pt[2]);
    }
    
    void setMax(const Eigen::Vector4f& max_pt) {
        max_pt_ = Point3D(max_pt[0], max_pt[1], max_pt[2]);
    }
    
    void setNegative(bool negative) { negative_ = negative; }
    
    void setInputCloud(const PointCloud::ConstPtr& cloud) { input_cloud_ = cloud; }
    
    void filter(PointCloud& output) {
        if (!input_cloud_) {
            output.clear();
            return;
        }
        
        output.clear();
        output.reserve(input_cloud_->size());
        
        for (const Point3D& point : *input_cloud_) {
            bool inside = (point.x >= min_pt_.x && point.x <= max_pt_.x &&
                          point.y >= min_pt_.y && point.y <= max_pt_.y &&
                          point.z >= min_pt_.z && point.z <= max_pt_.z);
            
            if (inside != negative_) {
                output.push_back(point);
            }
        }
    }
    
private:
    Point3D min_pt_, max_pt_;
    bool negative_;
    PointCloud::ConstPtr input_cloud_;
};

/**
 * @brief Range filter for distance-based filtering
 */
class RangeFilter {
public:
    RangeFilter() : min_range_(0.0f), max_range_(100.0f) {}
    
    void setRadiusLimits(float min_radius, float max_radius) {
        min_range_ = min_radius;
        max_range_ = max_radius;
    }
    
    void setInputCloud(const PointCloud::ConstPtr& cloud) { input_cloud_ = cloud; }
    
    void filter(PointCloud& output) {
        if (!input_cloud_) {
            output.clear();
            return;
        }
        
        output.clear();
        output.reserve(input_cloud_->size());
        
        for (const Point3D& point : *input_cloud_) {
            float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (range >= min_range_ && range <= max_range_) {
                output.push_back(point);
            }
        }
    }
    
private:
    float min_range_, max_range_;
    PointCloud::ConstPtr input_cloud_;
};

} // namespace util
} // namespace lidar_slam