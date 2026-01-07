/**
 * @file      Estimator.h
 * @brief     LiDAR odometry estimator with ICP and mapping capabilities.
 * @author    Seungwon Choi
 * @date      2025-09-24
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include "../util/MathUtils.h"
#include "../util/ConfigUtils.h"
#include "../util/PointCloudUtils.h"
#include "../database/VoxelMap.h"
#include "LoopClosureDetector.h"

#include <memory>
#include <vector>
#include <map>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <optional>
#include <chrono>

// Forward declarations
namespace lidar_slam {
namespace database {
class LidarFrame;
}
namespace optimization {
class AdaptiveMEstimator;
class PoseGraphOptimizer;
class IterativeClosestPointOptimizer;
struct ICPConfig;
}
}

namespace lidar_slam {
namespace processing {

// Import types from util namespace  
using namespace lidar_slam::util;

/**
 * @brief LiDAR odometry estimator
 * 
 * This class performs:
 * - Frame-to-frame ICP registration
 * - Feature-based optimization with Ceres
 * - Local mapping and keyframe management
 * - Pose graph optimization
 */
class Estimator {
public:
    /**
     * @brief Constructor
     * @param config Configuration from YAML file
     */
    explicit Estimator(const util::SystemConfig& config);
    
    /**
     * @brief Destructor
     */
    ~Estimator();
    
    // ===== Main Processing =====
    
    /**
     * @brief Process new LiDAR frame
     * @param current_frame Current LiDAR frame
     * @return True if processing successful
     */
    bool process_frame(std::shared_ptr<database::LidarFrame> current_frame);
    
    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void update_config(const util::SystemConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const util::SystemConfig& get_config() const;
    
    /**
     * @brief Get local map for visualization
     * @return Const pointer to local map
     */
    PointCloudConstPtr get_local_map() const;

    /**
    * @brief Get current pose
    * @return Current pose as SE3f
    */
    const SE3f& get_current_pose() const;

    /**
     * @brief Get dual frame optimization statistics
     * @param avg_iterations Average iterations per optimization call
     * @param avg_time_ms Average time per optimization call in milliseconds
     */
    void get_optimization_statistics(double& avg_iterations, double& avg_time_ms) const;
    
    /**
     * @brief Get debug clouds for visualization
     * @param pre_icp_cloud Output pre-ICP cloud
     * @param post_icp_cloud Output post-ICP cloud
     */
    void get_debug_clouds(PointCloudConstPtr& pre_icp_cloud, PointCloudConstPtr& post_icp_cloud) const;
    
    /**
     * @brief Get number of keyframes
     * @return Number of keyframes created so far
     */
    size_t get_keyframe_count() const;
    
    /**
     * @brief Get keyframe by index
     * @param index Keyframe index
     * @return Keyframe at the given index, nullptr if index out of bounds
     */
    std::shared_ptr<database::LidarFrame> get_keyframe(size_t index) const;
    
    /**
     * @brief Get VoxelMap pointer for visualization
     * @return Raw pointer to VoxelMap (do not delete)
     */
    map::VoxelMap* get_voxel_map() const { return m_voxel_map.get(); }
    
    /**
     * @brief Enable/disable loop closure detection
     * @param enable Enable loop closure detection
     */
    void enable_loop_closure(bool enable);
    
    /**
     * @brief Set loop closure detection configuration
     * @param config Loop closure configuration
     */
    void set_loop_closure_config(const LoopClosureConfig& config);
    
    /**
     * @brief Get loop closure detection statistics
     * @return Number of loop closures detected
     */
    size_t get_loop_closure_count() const;
    
    /**
     * @brief Get optimized trajectory from pose graph optimization (for debugging)
     * @return Map of keyframe ID to optimized pose (as Eigen::Matrix4f)
     */
    std::map<int, Eigen::Matrix4f> get_optimized_trajectory() const;
    
    /**
     * @brief Save final map to PLY file
     * @param output_path Output file path (e.g., "/path/to/map.ply")
     * @param voxel_size Voxel size for downsampling (0 = no downsampling)
     * @return True if save successful
     */
    bool save_map_to_ply(const std::string& output_path, float voxel_size = 0.2f);

    /**
     * @brief Save final map to PCD file
     * @param output_path Output file path (e.g., "/path/to/map.pcd")
     * @param voxel_size Voxel size for downsampling (0 = no downsampling)
     * @return True if save successful
     */
    bool save_map_to_pcd(const std::string& output_path, float voxel_size = 0.2f);

private:
    // ===== Internal Processing =====
    
    /**
     * @brief Preprocess frame: downsample and extract features
     * @param frame Frame to preprocess (will be modified in-place)
     * @return True if preprocessing successful
     */
    bool preprocess_frame(std::shared_ptr<database::LidarFrame> frame);
    
    /**
     * @brief Initialize with first frame
     * @param frame First frame
     */
    void initialize_first_frame(std::shared_ptr<database::LidarFrame> frame);
    
    /**
     * @brief Estimate motion between frames using DualFrameICPOptimizer
     * @param current_frame Current frame with features
     * @param keyframe Reference keyframe with local map
     * @param initial_guess Initial transformation guess
     * @return Estimated transformation from keyframe to current
     */
    SE3f estimate_motion_dual_frame(std::shared_ptr<database::LidarFrame> current_frame,
                                   std::shared_ptr<database::LidarFrame> keyframe,
                                   const SE3f& initial_guess = SE3f());
    
    /**
     * @brief Select best keyframe for current frame
     * @param current_pose Current frame pose
     * @return Best keyframe for optimization, nullptr if none suitable
     */
    std::shared_ptr<database::LidarFrame> select_best_keyframe(const SE3f& current_pose);
    
    /**
     * @brief Check if current frame should be a keyframe
     * @param current_pose Current pose
     * @return True if should be keyframe
     */
    bool should_create_keyframe(const SE3f& current_pose);
    
    /**
     * @brief Create new keyframe
     * @param frame Frame to use as keyframe
     * @param feature_cloud Features of the frame as point cloud
     */
    void create_keyframe(std::shared_ptr<database::LidarFrame> frame);
    
    /**
     * @brief Process loop closure candidates and compute relative poses
     * @param current_keyframe Current keyframe
     * @param loop_candidates List of loop closure candidates
     */
    void process_loop_closures(std::shared_ptr<database::LidarFrame> current_keyframe, 
                              const std::vector<LoopCandidate>& loop_candidates);

    /**
     * @brief Apply pose graph optimization results to all keyframes
     */
    void apply_pose_graph_optimization();

    /**
     * @brief Rebuild local map for current keyframe after loop closure
     * Clears existing local map and rebuilds it from the updated global feature map
     */
    void rebuild_current_keyframe_local_map();

    /**
     * @brief Background thread function for loop detection and PGO
     * Runs continuously, waiting for loop queries and performing PGO when loops are detected
     */
    void loop_pgo_thread_function();

    /**
     * @brief Apply pending PGO result from background thread (called in main thread)
     * Checks if there's a pending result and applies it to all keyframes
     */
    void apply_pending_pgo_result_if_available();

    /**
     * @brief Propagate poses to keyframes added after PGO using relative transforms
     * @param last_optimized_kf_id Last keyframe ID included in PGO
     */
    void propagate_poses_after_pgo(int last_optimized_kf_id);

    /**
     * @brief Transform current keyframe's map using correction transform
     * @param correction Correction transform to apply
     */
    void transform_current_keyframe_map(const SE3f& correction);

    /**
     * @brief Run PGO for detected loop closure in background thread
     * @param current_keyframe Current keyframe where loop was detected
     * @param loop_candidates List of loop closure candidates
     * @return True if PGO succeeded
     */
    bool run_pgo_for_loop(std::shared_ptr<database::LidarFrame> current_keyframe,
                         const std::vector<LoopCandidate>& loop_candidates);

    

private:
    // Configuration
    util::SystemConfig m_config;
    
    // State
    bool m_initialized;
    SE3f m_T_wl_current;
    SE3f m_velocity;  // Velocity model: T_current = T_previous * m_velocity
    std::vector<SE3f> m_trajectory;
    int m_next_keyframe_id;  // Next keyframe ID to assign
    
    // Frames and features
    std::shared_ptr<database::LidarFrame> m_previous_frame;
    std::shared_ptr<database::LidarFrame> m_old_frame;  // For memory cleanup
    std::deque<std::shared_ptr<database::LidarFrame>> m_keyframes;
    
    
    // Debug visualization clouds
    PointCloudPtr m_debug_pre_icp_cloud;
    PointCloudPtr m_debug_post_icp_cloud;
    
    // Processing tools
    std::shared_ptr<optimization::IterativeClosestPointOptimizer> m_icp_optimizer;
    std::unique_ptr<map::FastVoxelGrid> m_fast_voxel_grid;  // Fast voxel filter with Morton code
    std::unique_ptr<map::VoxelMap> m_voxel_map;              // 2-Level hierarchical voxel map with surfels
    std::unique_ptr<util::VoxelGrid> m_voxel_filter;        // Legacy voxel filter (for map downsampling)
    std::shared_ptr<optimization::AdaptiveMEstimator> m_adaptive_estimator;
    
    // Loop closure detection and pose graph optimization
    std::unique_ptr<LoopClosureDetector> m_loop_detector;
    std::shared_ptr<optimization::PoseGraphOptimizer> m_pose_graph_optimizer;
    int m_last_successful_loop_keyframe_id;  // Last keyframe ID where loop closure succeeded
    std::map<int, SE3f> m_optimized_poses;  // Optimized poses from PGO (for debugging visualization)
    
    // Asynchronous loop detection and PGO
    std::thread m_loop_pgo_thread;                      // Background thread for loop+PGO
    std::atomic<bool> m_thread_running{true};           // Thread control flag
    std::atomic<bool> m_pgo_in_progress{false};         // PGO status flag
    
    // Query queue: Main thread → Background thread
    std::mutex m_query_mutex;
    std::deque<int> m_loop_query_queue;                 // Keyframe IDs to check for loops
    std::condition_variable m_query_cv;                 // Wake up background thread
    
    // Result queue: Background thread → Main thread
    std::mutex m_result_mutex;
    struct PGOResult {
        int last_optimized_kf_id;                        // Last keyframe ID optimized by PGO
        std::map<int, SE3f> optimized_poses;             // Optimized absolute poses
        SE3f last_kf_correction;                         // Correction transform for last keyframe
        std::chrono::steady_clock::time_point timestamp; // When PGO completed
    };
    std::optional<PGOResult> m_pending_result;           // Pending PGO result to apply
    
    // Keyframe protection
    std::mutex m_keyframes_mutex;                        // Protects m_keyframes deque
    
    // Loop closure storage
    struct LoopConstraint {
        int from_keyframe_id;
        int to_keyframe_id;
        SE3f relative_pose;
        double translation_noise;
        double rotation_noise;
    };
    std::vector<LoopConstraint> m_loop_constraints;  // All detected loop closures
    
    // Last keyframe for optimization
    std::shared_ptr<database::LidarFrame> m_last_keyframe;
    
    // Last keyframe pose for keyframe decision
    SE3f m_last_keyframe_pose;
    
    // Optimization statistics (changed from ICP to dual frame)
    mutable size_t m_total_optimization_iterations;
    mutable double m_total_optimization_time_ms;
    mutable size_t m_optimization_call_count;

    // ===== Timing Statistics =====
    struct TimingStats {
        double preprocessing_ms = 0.0;
        double icp_ms = 0.0;
        double map_update_ms = 0.0;
        double total_ms = 0.0;
    };
    
    std::vector<TimingStats> m_timing_history;
    size_t m_frame_count = 0;
    
    void print_timing_statistics() const;

};

} // namespace processing
} // namespace lidar_slam
