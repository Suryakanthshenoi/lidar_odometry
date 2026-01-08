/**
 * @file      Config.cpp
 * @brief     Configuration management implementation.
 * @author    Your Name
 * @date      2025-01-09
 * @copyright Copyright (c) 2025. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "ConfigUtils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <algorithm>
#include <cctype>

namespace lidar_slam {
namespace util {

// Simple YAML parser for our configuration
class SimpleYamlParser {
public:
    std::map<std::string, std::string> parse(const std::string& filename) {
        std::map<std::string, std::string> config;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Failed to open config file: " << filename << std::endl;
            return config;
        }
        
        std::string line;
        std::string current_section = "";
        
        while (std::getline(file, line)) {
            // Remove comments and trim whitespace
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            line = trim(line);
            
            if (line.empty()) continue;
            
            // Check if it's a section header (no leading spaces and ends with :)
            if (line.back() == ':' && line.find(' ') == std::string::npos) {
                current_section = line.substr(0, line.length() - 1) + ".";
                continue;
            }
            
            // Parse key-value pair
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = trim(line.substr(0, colon_pos));
                std::string value = trim(line.substr(colon_pos + 1));
                
                // Remove quotes from string values
                if (value.front() == '"' && value.back() == '"') {
                    value = value.substr(1, value.length() - 2);
                }
                
                config[current_section + key] = value;
            }
        }
        
        return config;
    }

private:
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(' ');
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(' ');
        return str.substr(first, (last - first + 1));
    }
};

bool ConfigManager::load_from_file(const std::string& config_file) {
    SimpleYamlParser parser;
    auto config_map = parser.parse(config_file);
    
    if (config_map.empty()) {
        std::cerr << "Failed to load configuration from " << config_file << std::endl;
        return false;
    }
    
    // Parse configuration values
    try {
        // Data paths
        if (config_map.find("data_directory") != config_map.end()) {
            m_config.data_directory = config_map["data_directory"];
        }
        if (config_map.find("ground_truth_directory") != config_map.end()) {
            m_config.ground_truth_directory = config_map["ground_truth_directory"];
        }
        if (config_map.find("output_directory") != config_map.end()) {
            m_config.output_directory = config_map["output_directory"];
        }
        
        // Point cloud processing
        if (config_map.find("point_cloud.voxel_size") != config_map.end()) {
            m_config.voxel_size = std::stof(config_map["point_cloud.voxel_size"]);
        }
        if (config_map.find("point_cloud.point_stride") != config_map.end()) {
            m_config.point_stride = std::stoi(config_map["point_cloud.point_stride"]);
        }
        if (config_map.find("point_cloud.map_voxel_size") != config_map.end()) {
            m_config.map_voxel_size = std::stof(config_map["point_cloud.map_voxel_size"]);
        }
        if (config_map.find("point_cloud.surfel_planarity_threshold") != config_map.end()) {
            m_config.surfel_planarity_threshold = std::stof(config_map["point_cloud.surfel_planarity_threshold"]);
        }
        if (config_map.find("point_cloud.max_range") != config_map.end()) {
            m_config.max_range = std::stof(config_map["point_cloud.max_range"]);
        }
        if (config_map.find("point_cloud.min_range") != config_map.end()) {
            m_config.min_range = std::stof(config_map["point_cloud.min_range"]);
        }
        
        // Feature extraction
        if (config_map.find("feature_extraction.min_plane_points") != config_map.end()) {
            m_config.min_plane_points = std::stoull(config_map["feature_extraction.min_plane_points"]);
        }
        if (config_map.find("feature_extraction.max_neighbors") != config_map.end()) {
            m_config.max_neighbors = std::stoull(config_map["feature_extraction.max_neighbors"]);
        }
        if (config_map.find("feature_extraction.max_plane_distance") != config_map.end()) {
            m_config.max_plane_distance = std::stof(config_map["feature_extraction.max_plane_distance"]);
        }
        if (config_map.find("feature_extraction.collinearity_threshold") != config_map.end()) {
            m_config.collinearity_threshold = std::stof(config_map["feature_extraction.collinearity_threshold"]);
        }
        if (config_map.find("feature_extraction.max_neighbor_distance") != config_map.end()) {
            m_config.max_neighbor_distance = std::stof(config_map["feature_extraction.max_neighbor_distance"]);
        }
        if (config_map.find("feature_extraction.feature_quality_threshold") != config_map.end()) {
            m_config.feature_quality_threshold = std::stof(config_map["feature_extraction.feature_quality_threshold"]);
        }
        if (config_map.find("feature_extraction.feature_voxel_size") != config_map.end()) {
            m_config.feature_voxel_size = std::stof(config_map["feature_extraction.feature_voxel_size"]);
        }
        
        // Odometry
        if (config_map.find("odometry.max_iterations") != config_map.end()) {
            m_config.max_iterations = std::stoull(config_map["odometry.max_iterations"]);
        }
        if (config_map.find("odometry.translation_threshold") != config_map.end()) {
            m_config.translation_threshold = std::stod(config_map["odometry.translation_threshold"]);
        }
        if (config_map.find("odometry.rotation_threshold") != config_map.end()) {
            m_config.rotation_threshold = std::stod(config_map["odometry.rotation_threshold"]);
        }
        if (config_map.find("odometry.max_correspondence_distance") != config_map.end()) {
            m_config.max_correspondence_distance = std::stof(config_map["odometry.max_correspondence_distance"]);
        }
        if (config_map.find("odometry.initial_guess_rotation") != config_map.end()) {
            m_config.initial_guess_rotation = std::stof(config_map["odometry.initial_guess_rotation"]);
        }
        
        // Robust estimation - PKO only
        if (config_map.find("robust_estimation.use_adaptive_m_estimator") != config_map.end()) {
            m_config.use_adaptive_m_estimator = config_map["robust_estimation.use_adaptive_m_estimator"] == "true";
        }
        if (config_map.find("robust_estimation.min_scale_factor") != config_map.end()) {
            m_config.min_scale_factor = std::stod(config_map["robust_estimation.min_scale_factor"]);

            std::cout<<"[ConfigManager] Set min_scale_factor to " << m_config.min_scale_factor << std::endl;
        }
        if (config_map.find("robust_estimation.max_scale_factor") != config_map.end()) {
            m_config.max_scale_factor = std::stod(config_map["robust_estimation.max_scale_factor"]);
        }
        
        // PKO (Probabilistic Kernel Optimization) parameters
        if (config_map.find("robust_estimation.num_alpha_segments") != config_map.end()) {
            m_config.num_alpha_segments = std::stoi(config_map["robust_estimation.num_alpha_segments"]);
        }
        if (config_map.find("robust_estimation.truncated_threshold") != config_map.end()) {
            m_config.truncated_threshold = std::stod(config_map["robust_estimation.truncated_threshold"]);
        }
        if (config_map.find("robust_estimation.gmm_components") != config_map.end()) {
            m_config.gmm_components = std::stoi(config_map["robust_estimation.gmm_components"]);
        }
        if (config_map.find("robust_estimation.gmm_sample_size") != config_map.end()) {
            m_config.gmm_sample_size = std::stoi(config_map["robust_estimation.gmm_sample_size"]);
        }
        if (config_map.find("robust_estimation.pko_kernel_type") != config_map.end()) {
            m_config.pko_kernel_type = config_map["robust_estimation.pko_kernel_type"];
        }
        
        // Estimator
        if (config_map.find("estimator.map_voxel_size") != config_map.end()) {
            m_config.map_voxel_size = std::stof(config_map["estimator.map_voxel_size"]);
        }
        if (config_map.find("estimator.keyframe_distance_threshold") != config_map.end()) {
            m_config.keyframe_distance_threshold = std::stod(config_map["estimator.keyframe_distance_threshold"]);
        }
        if (config_map.find("estimator.keyframe_rotation_threshold") != config_map.end()) {
            m_config.keyframe_rotation_threshold = std::stod(config_map["estimator.keyframe_rotation_threshold"]);
        }
        if (config_map.find("estimator.min_correspondence_points") != config_map.end()) {
            m_config.min_correspondence_points = std::stoull(config_map["estimator.min_correspondence_points"]);
        }
        if (config_map.find("estimator.max_solver_iterations") != config_map.end()) {
            m_config.max_solver_iterations = std::stoull(config_map["estimator.max_solver_iterations"]);
        }
        if (config_map.find("estimator.parameter_tolerance") != config_map.end()) {
            m_config.parameter_tolerance = std::stod(config_map["estimator.parameter_tolerance"]);
        }
        if (config_map.find("estimator.function_tolerance") != config_map.end()) {
            m_config.function_tolerance = std::stod(config_map["estimator.function_tolerance"]);
        }
        if (config_map.find("estimator.max_map_frames") != config_map.end()) {
            m_config.max_map_frames = std::stoull(config_map["estimator.max_map_frames"]);
        }
        if (config_map.find("estimator.use_surfel_correspondence") != config_map.end()) {
            m_config.use_surfel_correspondence = (config_map["estimator.use_surfel_correspondence"] == "true" ||
                                                   config_map["estimator.use_surfel_correspondence"] == "1");
        }
        
        // Viewer
        if (config_map.find("viewer.viewer_width") != config_map.end()) {
            m_config.viewer_width = std::stoi(config_map["viewer.viewer_width"]);
        }
        if (config_map.find("viewer.viewer_height") != config_map.end()) {
            m_config.viewer_height = std::stoi(config_map["viewer.viewer_height"]);
        }
        if (config_map.find("viewer.point_size") != config_map.end()) {
            m_config.point_size = std::stof(config_map["viewer.point_size"]);
        }
        if (config_map.find("viewer.feature_point_size") != config_map.end()) {
            m_config.feature_point_size = std::stof(config_map["viewer.feature_point_size"]);
        }
        if (config_map.find("viewer.trajectory_width") != config_map.end()) {
            m_config.trajectory_width = std::stof(config_map["viewer.trajectory_width"]);
        }
        if (config_map.find("viewer.auto_mode") != config_map.end()) {
            m_config.auto_mode = (config_map["viewer.auto_mode"] == "true");
        }
        if (config_map.find("viewer.show_point_cloud") != config_map.end()) {
            m_config.show_point_cloud = (config_map["viewer.show_point_cloud"] == "true");
        }
        if (config_map.find("viewer.show_features") != config_map.end()) {
            m_config.show_features = (config_map["viewer.show_features"] == "true");
        }
        if (config_map.find("viewer.show_trajectory") != config_map.end()) {
            m_config.show_trajectory = (config_map["viewer.show_trajectory"] == "true");
        }
        if (config_map.find("viewer.show_coordinate_frame") != config_map.end()) {
            m_config.show_coordinate_frame = (config_map["viewer.show_coordinate_frame"] == "true");
        }
        if (config_map.find("viewer.coordinate_frame_size") != config_map.end()) {
            m_config.coordinate_frame_size = std::stod(config_map["viewer.coordinate_frame_size"]);
        }
        if (config_map.find("viewer.coordinate_frame_width") != config_map.end()) {
            m_config.coordinate_frame_width = std::stod(config_map["viewer.coordinate_frame_width"]);
        }
        if (config_map.find("viewer.show_grid") != config_map.end()) {
            m_config.show_grid = (config_map["viewer.show_grid"] == "true");
        }
        if (config_map.find("viewer.follow_camera") != config_map.end()) {
            m_config.follow_camera = (config_map["viewer.follow_camera"] == "true");
        }
        if (config_map.find("viewer.top_view_follow") != config_map.end()) {
            m_config.top_view_follow = (config_map["viewer.top_view_follow"] == "true");
        }
        
        // Performance
        if (config_map.find("performance.num_threads") != config_map.end()) {
            m_config.num_threads = std::stoull(config_map["performance.num_threads"]);
        }
        if (config_map.find("performance.use_multithreading") != config_map.end()) {
            m_config.use_multithreading = (config_map["performance.use_multithreading"] == "true");
        }
        if (config_map.find("performance.max_frames_in_memory") != config_map.end()) {
            m_config.max_frames_in_memory = std::stoull(config_map["performance.max_frames_in_memory"]);
        }
        
        // KITTI specific settings
        if (config_map.find("kitti.sequence") != config_map.end()) {
            m_config.kitti_sequence = config_map["kitti.sequence"];
        }
        if (config_map.find("seq") != config_map.end()) {
            m_config.kitti_sequence = config_map["seq"];
        }
        if (config_map.find("kitti.start_frame") != config_map.end()) {
            m_config.kitti_start_frame = std::stoi(config_map["kitti.start_frame"]);
        }
        if (config_map.find("kitti.end_frame") != config_map.end()) {
            m_config.kitti_end_frame = std::stoi(config_map["kitti.end_frame"]);
        }
        if (config_map.find("kitti.frame_skip") != config_map.end()) {
            m_config.kitti_frame_skip = std::stoi(config_map["kitti.frame_skip"]);
        }
        if (config_map.find("kitti.velodyne_path") != config_map.end()) {
            m_config.kitti_velodyne_path = config_map["kitti.velodyne_path"];
        }
        if (config_map.find("kitti.poses_file") != config_map.end()) {
            m_config.kitti_poses_file = config_map["kitti.poses_file"];
        }
        
        // Player settings
        if (config_map.find("player.enable_viewer") != config_map.end()) {
            m_config.player_enable_viewer = (config_map["player.enable_viewer"] == "true");
        }
        if (config_map.find("player.enable_statistics") != config_map.end()) {
            m_config.player_enable_statistics = (config_map["player.enable_statistics"] == "true");
        }
        if (config_map.find("player.enable_console_statistics") != config_map.end()) {
            m_config.player_enable_console_statistics = (config_map["player.enable_console_statistics"] == "true");
        }
        if (config_map.find("player.step_mode") != config_map.end()) {
            m_config.player_step_mode = (config_map["player.step_mode"] == "true");
        }
        if (config_map.find("player.auto_ground_truth_path") != config_map.end()) {
            m_config.player_auto_ground_truth_path = (config_map["player.auto_ground_truth_path"] == "true");
        }
        
        // Loop closure detection settings
        if (config_map.find("loop_detector.enable_loop_detection") != config_map.end()) {
            m_config.loop_enable_loop_detection = (config_map["loop_detector.enable_loop_detection"] == "true");
        }
        if (config_map.find("loop_detector.enable_adjacent_optimization") != config_map.end()) {
            m_config.loop_enable_adjacent_optimization = (config_map["loop_detector.enable_adjacent_optimization"] == "true");
        }
        if (config_map.find("loop_detector.similarity_threshold") != config_map.end()) {
            m_config.loop_similarity_threshold = std::stof(config_map["loop_detector.similarity_threshold"]);
        }
        if (config_map.find("loop_detector.min_keyframe_gap") != config_map.end()) {
            m_config.loop_min_keyframe_gap = std::stoi(config_map["loop_detector.min_keyframe_gap"]);
        }
        if (config_map.find("loop_detector.max_search_distance") != config_map.end()) {
            m_config.loop_max_search_distance = std::stof(config_map["loop_detector.max_search_distance"]);
        }
        if (config_map.find("loop_detector.enable_debug_output") != config_map.end()) {
            m_config.loop_enable_debug_output = (config_map["loop_detector.enable_debug_output"] == "true");
        }
        
        // Keyframe management settings
        if (config_map.find("keyframe.window_size") != config_map.end()) {
            m_config.keyframe_window_size = std::stoi(config_map["keyframe.window_size"]);
        }
        
        // Pose Graph Optimization (PGO) settings
        if (config_map.find("pose_graph_optimization.enable_pgo") != config_map.end()) {
            m_config.pgo_enable_pgo = (config_map["pose_graph_optimization.enable_pgo"] == "true");
        }
        if (config_map.find("pose_graph_optimization.pgo_backend") != config_map.end()) {
            m_config.pgo_backend = config_map["pose_graph_optimization.pgo_backend"];
        }
        if (config_map.find("pose_graph_optimization.odometry_translation_noise") != config_map.end()) {
            m_config.pgo_odometry_translation_noise = std::stod(config_map["pose_graph_optimization.odometry_translation_noise"]);
        }
        if (config_map.find("pose_graph_optimization.odometry_rotation_noise") != config_map.end()) {
            m_config.pgo_odometry_rotation_noise = std::stod(config_map["pose_graph_optimization.odometry_rotation_noise"]);
        }
        if (config_map.find("pose_graph_optimization.loop_translation_noise") != config_map.end()) {
            m_config.pgo_loop_translation_noise = std::stod(config_map["pose_graph_optimization.loop_translation_noise"]);
        }
        if (config_map.find("pose_graph_optimization.loop_rotation_noise") != config_map.end()) {
            m_config.pgo_loop_rotation_noise = std::stod(config_map["pose_graph_optimization.loop_rotation_noise"]);
        }
        
        // Output settings
        if (config_map.find("output.save_map") != config_map.end()) {
            m_config.output_save_map = (config_map["output.save_map"] == "true");
        }
        if (config_map.find("output.map_voxel_size") != config_map.end()) {
            m_config.output_map_voxel_size = std::stof(config_map["output.map_voxel_size"]);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing configuration: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigManager::save_to_file(const std::string& config_file) const {
    std::ofstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file for writing: " << config_file << std::endl;
        return false;
    }
    
    file << "# LiDAR Odometry System Configuration\n";
    file << "# Generated automatically\n\n";
    
    // Data paths
    file << "data_directory: \"" << m_config.data_directory << "\"\n";
    file << "output_directory: \"" << m_config.output_directory << "\"\n\n";
    
    // Point cloud processing
    file << "processing:\n";
    file << "  voxel_size: " << m_config.voxel_size << "\n";
    file << "  max_range: " << m_config.max_range << "\n";
    file << "  min_range: " << m_config.min_range << "\n";
    
    // And so on for other sections...
    // (I'll implement the rest if needed)
    
    return true;
}

bool ConfigManager::validate_config() const {
    // Basic validation
    if (m_config.data_directory.empty() || m_config.output_directory.empty()) {
        return false;
    }
    
    if (m_config.voxel_size <= 0.0f || m_config.max_range <= m_config.min_range) {
        return false;
    }
    
    if (m_config.min_plane_points < 3 || m_config.max_neighbors < m_config.min_plane_points) {
        return false;
    }
    
    if (m_config.viewer_width <= 0 || m_config.viewer_height <= 0) {
        return false;
    }
    
    return true;
}

SystemConfig ConfigManager::create_default_config() {
    SystemConfig config;
    // Default values are already set in the struct definition
    return config;
}

} // namespace util
} // namespace lidar_slam
