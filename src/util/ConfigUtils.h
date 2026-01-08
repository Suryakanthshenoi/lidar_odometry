/**
 * @file      ConfigUtils.h
 * @brief     Configuration management for LiDAR odometry system.
 * @author    Your Name
 * @date      2025-01-09
 * @copyright Copyright (c) 2025. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <string>
#include <memory>

namespace lidar_slam {
namespace util {

/**
 * @brief System configuration structure
 */
struct SystemConfig {
    // ===== Data paths =====
    std::string data_directory = "./data";          ///< Data directory path
    std::string ground_truth_directory = "";        ///< Ground truth directory path
    std::string output_directory = "./output";      ///< Output directory path
    
    // ===== Point cloud processing =====
    float voxel_size = 0.1f;                        ///< Voxel grid size for downsampling
    int point_stride = 1;                           ///< Stride for point cloud (1 = all points, 3 = every 3rd)
    float max_range = 100.0f;                       ///< Maximum range for point cloud filtering
    float min_range = 0.5f;                         ///< Minimum range for point cloud filtering
    
    // ===== Feature extraction =====
    size_t min_plane_points = 5;                    ///< Minimum points for plane fitting
    size_t max_neighbors = 10;                      ///< Maximum neighbors for plane fitting
    float max_plane_distance = 0.05f;               ///< Maximum distance from plane (m)
    float collinearity_threshold = 0.05f;           ///< Collinearity threshold
    float max_neighbor_distance = 1.0f;             ///< Maximum neighbor search distance
    float feature_quality_threshold = 0.1f;         ///< Minimum quality for valid features
    float feature_voxel_size = 0.1f;                ///< Voxel size for feature extraction
    
    // ===== Odometry =====
    size_t max_iterations = 20;                     ///< Maximum optimization iterations
    double translation_threshold = 0.001;           ///< Translation convergence threshold (meters)
    double rotation_threshold = 0.001;              ///< Rotation convergence threshold (radians)
    float max_correspondence_distance = 1.0f;       ///< Maximum distance for correspondences
    float initial_guess_rotation = 0.1f;            ///< Initial rotation guess limit (rad)
    
    // ===== Estimator =====
    float map_voxel_size = 0.2f;                    ///< Local map voxel size
    float surfel_planarity_threshold = 0.1f;        ///< Planarity threshold for surfel creation
    double keyframe_distance_threshold = 1.0;       ///< Distance threshold for keyframe creation
    double keyframe_rotation_threshold = 0.2;       ///< Rotation threshold for keyframe creation
    size_t min_correspondence_points = 10;           ///< Minimum correspondence points
    size_t max_solver_iterations = 50;              ///< Maximum Ceres solver iterations
    double parameter_tolerance = 1e-8;              ///< Parameter tolerance for Ceres
    double function_tolerance = 1e-8;               ///< Function tolerance for Ceres
    size_t max_map_frames = 20;                     ///< Maximum frames in local map
    bool use_surfel_correspondence = true;          ///< Use surfel-based correspondence (true) or KDTree-based (false)
    
    // ===== Robust estimation =====
    bool use_adaptive_m_estimator = true;           ///< Enable adaptive M-estimator
    std::string loss_type = "huber";                ///< Loss function type
    std::string scale_method = "PKO";               ///< Scale factor calculation method ("MAD", "fixed", "std")
    double fixed_scale_factor = 1.0;               ///< Fixed scale factor when scale_method is "fixed"
    double mad_multiplier = 1.4826;                ///< MAD to standard deviation conversion
    double min_scale_factor = 0.01;                ///< Minimum scale factor (also PKO alpha lower bound)
    double max_scale_factor = 10.0;                ///< Maximum scale factor (also PKO alpha upper bound)
    
    // PKO (Probabilistic Kernel Optimization) parameters
    int num_alpha_segments = 1000;                 ///< PKO alpha segments
    double truncated_threshold = 10.0;             ///< PKO truncated threshold
    int gmm_components = 3;                        ///< GMM components for PKO
    int gmm_sample_size = 100;                     ///< GMM sample size for PKO
    std::string pko_kernel_type = "cauchy";        ///< PKO kernel type
    
    // ===== Viewer =====
    int viewer_width = 1280;                        ///< Viewer window width
    int viewer_height = 960;                        ///< Viewer window height
    float point_size = 2.0f;                        ///< Point cloud point size
    float feature_point_size = 5.0f;                ///< Feature point size
    float trajectory_width = 2.0f;                  ///< Trajectory line width
    bool auto_mode = false;                         ///< Auto playback mode
    bool show_point_cloud = true;                   ///< Show point cloud
    bool show_features = true;                      ///< Show extracted features
    bool show_trajectory = true;                    ///< Show camera trajectory
    bool show_coordinate_frame = true;              ///< Show coordinate frames
    double coordinate_frame_size = 3.0;             ///< Coordinate frame axis length
    double coordinate_frame_width = 4.0;            ///< Coordinate frame line width
    bool show_grid = true;                          ///< Show reference grid
    bool follow_camera = false;                     ///< Follow camera mode
    bool top_view_follow = false;                   ///< Top-down view follow mode
    
    // ===== Performance =====
    size_t num_threads = 0;                         ///< Number of threads (0 = auto)
    bool use_multithreading = true;                 ///< Enable multithreading
    size_t max_frames_in_memory = 1000;             ///< Maximum frames to keep in memory
    
    // ===== KITTI specific =====
    std::string kitti_sequence = "00";              ///< KITTI sequence number
    int kitti_start_frame = 0;                      ///< Start frame index
    int kitti_end_frame = -1;                       ///< End frame index (-1 for all)
    int kitti_frame_skip = 1;                       ///< Process every N frames
    std::string kitti_velodyne_path = "velodyne";   ///< Velodyne data path
    std::string kitti_poses_file = "poses.txt";     ///< Ground truth poses file
    
    // ===== Player settings =====
    bool player_enable_viewer = true;               ///< Enable 3D viewer
    bool player_enable_statistics = true;           ///< Enable statistics output
    bool player_enable_console_statistics = true;   ///< Enable console statistics
    bool player_step_mode = false;                  ///< Step-by-step processing mode
    bool player_auto_ground_truth_path = true;      ///< Auto construct GT path from sequence
    
    // ===== Loop closure detection =====
    bool loop_enable_loop_detection = true;         ///< Enable loop closure detection
    float loop_similarity_threshold = 0.3f;         ///< LiDAR Iris similarity threshold (lower is more similar)
    int loop_min_keyframe_gap = 50;                 ///< Minimum keyframe ID difference for loop detection
    float loop_max_search_distance = 10.0f;         ///< Maximum distance (meters) to search for loop candidates
    bool loop_enable_debug_output = false;          ///< Enable detailed debug logging
    bool loop_enable_adjacent_optimization = false; ///< Enable fast local loop closure optimization for adjacent keyframes (experimental)
    
    // ===== Keyframe management =====
    int keyframe_window_size = 10;                  ///< Number of recent keyframes to keep full data (for loop closure)
    
    // Note: Iris parameters will be automatically calculated from max_range
    
    // ===== Pose Graph Optimization (PGO) =====
    bool pgo_enable_pgo = true;                     ///< Enable pose graph optimization
    std::string pgo_backend = "manual";             ///< PGO backend: "gtsam" or "manual" (batch GN)
    double pgo_odometry_translation_noise = 0.5;    ///< Odometry constraint translation noise (lower = more trust)
    double pgo_odometry_rotation_noise = 0.5;       ///< Odometry constraint rotation noise (lower = more trust)
    double pgo_loop_translation_noise = 1.0;        ///< Loop closure constraint translation noise (lower = more trust)
    double pgo_loop_rotation_noise = 1.0;           ///< Loop closure constraint rotation noise (lower = more trust)
    
    // ===== Output settings =====
    bool output_save_map = false;                   ///< Save final map as PLY file
    float output_map_voxel_size = 0.2f;             ///< Voxel size for final map downsampling
    
    SystemConfig() = default;
};

/**
 * @brief Configuration manager class
 */
class ConfigManager {
public:
    /**
     * @brief Get singleton instance
     * @return Reference to singleton instance
     */
    static ConfigManager& instance() {
        static ConfigManager instance;
        return instance;
    }
    
    /**
     * @brief Load configuration from file
     * @param config_file Path to configuration file
     * @return True if loaded successfully
     */
    bool load_from_file(const std::string& config_file);
    
    /**
     * @brief Save configuration to file
     * @param config_file Path to configuration file
     * @return True if saved successfully
     */
    bool save_to_file(const std::string& config_file) const;
    
    /**
     * @brief Get system configuration
     * @return Reference to system configuration
     */
    const SystemConfig& get_config() const { return m_config; }
    
    /**
     * @brief Get mutable system configuration
     * @return Reference to system configuration
     */
    SystemConfig& get_config() { return m_config; }
    
    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void set_config(const SystemConfig& config) { m_config = config; }
    
    /**
     * @brief Get data directory path
     * @return Data directory path
     */
    std::string get_data_directory() const { return m_config.data_directory; }
    
    /**
     * @brief Get output directory path
     * @return Output directory path
     */
    std::string get_output_directory() const { return m_config.output_directory; }
    
    /**
     * @brief Validate configuration parameters
     * @return True if configuration is valid
     */
    bool validate_config() const;
    
    /**
     * @brief Create default configuration
     * @return Default system configuration
     */
    static SystemConfig create_default_config();

private:
    SystemConfig m_config;
    
    ConfigManager() : m_config(create_default_config()) {}
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
};

// ===== Convenience functions =====

/**
 * @brief Get reference to current system configuration
 * @return Reference to system configuration
 */
inline const SystemConfig& config() {
    return ConfigManager::instance().get_config();
}

/**
 * @brief Get mutable reference to current system configuration
 * @return Mutable reference to system configuration
 */
inline SystemConfig& mutable_config() {
    return ConfigManager::instance().get_config();
}

} // namespace util
} // namespace lidar_slam
