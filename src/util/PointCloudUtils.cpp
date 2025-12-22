/**
 * @file      PointCloudUtils.cpp
 * @brief     Native C++ point cloud utilities implementation
 * @author    Seungwon Choi
 * @date      2025-10-03
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "PointCloudUtils.h"
#include "util/LogUtils.h"
#include <filesystem>

namespace lidar_slam {
namespace util {

PointCloud::Ptr load_kitti_binary(const std::string& filename) {
    auto cloud = std::make_shared<PointCloud>();
    
    // Check if file exists
    if (!std::filesystem::exists(filename)) {
        LOG_ERROR("KITTI binary file does not exist: {}", filename);
        return cloud;
    }
    
    // Open binary file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open KITTI binary file: {}", filename);
        return cloud;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // KITTI format: each point is 4 floats (x, y, z, intensity)
    // We only need x, y, z
    size_t num_points = file_size / (4 * sizeof(float));
    cloud->reserve(num_points);
    
    LOG_DEBUG("Loading KITTI binary file: {} ({} points)", filename, num_points);
    
    // Read points
    std::vector<float> buffer(4); // x, y, z, intensity
    for (size_t i = 0; i < num_points; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), 4 * sizeof(float));
        
        if (file.gcount() != 4 * sizeof(float)) {
            LOG_WARN("Incomplete read at point {} in file {}", i, filename);
            break;
        }
        
        // Add point (x, y, z) - ignore intensity
        cloud->push_back(buffer[0], buffer[1], buffer[2]);
    }
    
    file.close();
    LOG_DEBUG("Successfully loaded {} points from {}", cloud->size(), filename);
    
    return cloud;
}

bool save_kitti_binary(const PointCloud::ConstPtr& cloud, const std::string& filename) {
    if (!cloud || cloud->empty()) {
        LOG_ERROR("Cannot save empty point cloud to {}", filename);
        return false;
    }
    
    // Create directory if it doesn't exist
    std::filesystem::path file_path(filename);
    std::filesystem::create_directories(file_path.parent_path());
    
    // Open binary file for writing
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file for writing: {}", filename);
        return false;
    }
    
    // Write points in KITTI format (x, y, z, intensity)
    std::vector<float> buffer(4);
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& point = (*cloud)[i];
        buffer[0] = point.x;
        buffer[1] = point.y;
        buffer[2] = point.z;
        buffer[3] = 0.0f; // Set intensity to 0 as we don't use it
        
        file.write(reinterpret_cast<const char*>(buffer.data()), 4 * sizeof(float));
    }
    
    file.close();
    LOG_DEBUG("Successfully saved {} points to {}", cloud->size(), filename);
    
    return true;
}

void transform_point_cloud(const PointCloud::ConstPtr& input,
                          PointCloud::Ptr& output,
                          const Eigen::Matrix4f& transformation) {
    if (!input) {
        LOG_ERROR("Input point cloud is null");
        return;
    }
    
    if (!output) {
        output = std::make_shared<PointCloud>();
    }
    
    output->clear();
    output->reserve(input->size());
    
    // Transform each point
    for (size_t i = 0; i < input->size(); ++i) {
        const auto& point = (*input)[i];
        Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1.0f);
        Eigen::Vector4f transformed = transformation * homogeneous_point;
        
        output->push_back(transformed.x(), transformed.y(), transformed.z());
    }
}

void copy_point_cloud(const PointCloud::ConstPtr& input, PointCloud::Ptr& output) {
    if (!input) {
        LOG_ERROR("Input point cloud is null");
        return;
    }
    
    if (!output) {
        output = std::make_shared<PointCloud>();
    }
    
    output->clear();
    output->reserve(input->size());
    
    // Copy each point
    for (size_t i = 0; i < input->size(); ++i) {
        output->push_back((*input)[i]);
    }
}

bool save_point_cloud_ply(const std::string& filename, const PointCloud::ConstPtr& cloud) {
    if (!cloud || cloud->empty()) {
        LOG_ERROR("Cannot save empty point cloud to PLY: {}", filename);
        return false;
    }
    
    // Create directory if it doesn't exist
    std::filesystem::path file_path(filename);
    std::filesystem::create_directories(file_path.parent_path());
    
    // Open binary file for writing
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open PLY file for writing: {}", filename);
        return false;
    }
    
    // Write PLY header
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << cloud->size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "end_header\n";
    
    // Write point data in binary format
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& point = (*cloud)[i];
        file.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
    }
    
    file.close();
    LOG_INFO("Successfully saved {} points to PLY: {}", cloud->size(), filename);
    
    return true;
}

bool save_point_cloud_pcd(const std::string& filename, const PointCloud::ConstPtr& cloud) {
    if (!cloud || cloud->empty()) {
        LOG_ERROR("Cannot save empty point cloud to PCD: {}", filename);
        return false;
    }
    
    // Create directory if it doesn't exist
    std::filesystem::path file_path(filename);
    std::filesystem::create_directories(file_path.parent_path());
    
    // Open binary file for writing
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open PCD file for writing: {}", filename);
        return false;
    }
    
    // Write PCD header
    file << "# .PCD v0.7 - Point Cloud Data file format\n";
    file << "VERSION 0.7\n";
    file << "FIELDS x y z\n";
    file << "SIZE 4 4 4\n";
    file << "TYPE F F F\n";
    file << "COUNT 1 1 1\n";
    file << "WIDTH " << cloud->size() << "\n";
    file << "HEIGHT 1\n";
    file << "VIEWPOINT 0 0 0 1 0 0 0\n";
    file << "POINTS " << cloud->size() << "\n";
    file << "DATA binary\n";
    
    // Write point data in binary format
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& point = (*cloud)[i];
        file.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
    }
    
    file.close();
    LOG_INFO("Successfully saved {} points to PCD: {}", cloud->size(), filename);
    
    return true;
}
} // namespace util
} // namespace lidar_slam