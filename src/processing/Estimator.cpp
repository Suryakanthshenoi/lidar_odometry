/**
 * @file      Estimator.cpp
 * @brief     Implementation of LiDAR odometry estimator.
 * @author    Seungwon Choi
 * @date      2025-09-24
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Estimator.h"
#include "../util/MathUtils.h"
#include "../util/PointCloudUtils.h"
#include "../database/LidarFrame.h"
#include "../database/VoxelMap.h"
#include "../optimization/AdaptiveMEstimator.h"
#include "../optimization/PoseGraphOptimizer.h"
#include "../optimization/IterativeClosestPointOptimizer.h"
#include "LoopClosureDetector.h"
#include "util/LogUtils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace lidar_slam {
namespace processing {

Estimator::Estimator(const util::SystemConfig& config)
    : m_config(config)
    , m_initialized(false)
    , m_T_wl_current()
    , m_velocity()
    , m_next_keyframe_id(0)
    , m_debug_pre_icp_cloud(new PointCloud())
    , m_debug_post_icp_cloud(new PointCloud())
    , m_last_successful_loop_keyframe_id(-1)  // Initialize to -1 (no successful loop closure yet)
    , m_last_keyframe_pose()
    , m_total_optimization_iterations(0)
    , m_total_optimization_time_ms(0.0)
    , m_optimization_call_count(0)
{
    // Initialize pose graph optimizer (Manual Batch GN)
    m_pose_graph_optimizer = std::make_shared<optimization::PoseGraphOptimizer>();
    LOG_INFO("PGO backend: Manual (Batch GN)");
    
    // Create AdaptiveMEstimator with PKO configuration only
    m_adaptive_estimator = std::make_shared<optimization::AdaptiveMEstimator>(
        config.use_adaptive_m_estimator,
        config.loss_type,
        config.min_scale_factor,
        config.max_scale_factor,
        config.num_alpha_segments,
        config.truncated_threshold,
        config.gmm_components,
        config.gmm_sample_size,
        config.pko_kernel_type
    );
    
    // Initialize IterativeClosestPointOptimizer with AdaptiveMEstimator and configuration
    optimization::ICPConfig dual_frame_config;
    dual_frame_config.max_iterations = config.max_iterations;
    dual_frame_config.translation_tolerance = config.translation_threshold;
    dual_frame_config.rotation_tolerance = config.rotation_threshold;
    dual_frame_config.max_correspondence_distance = config.max_correspondence_distance;
    dual_frame_config.outlier_rejection_ratio = 0.9;
    dual_frame_config.use_robust_loss = true;
    dual_frame_config.robust_loss_delta = 0.1;
    dual_frame_config.use_surfel_correspondence = config.use_surfel_correspondence;
    
    m_icp_optimizer = std::make_shared<optimization::IterativeClosestPointOptimizer>(dual_frame_config, m_adaptive_estimator);
    
    // Initialize fast voxel grid filter (Morton code + Robin Hood hashing)
    m_fast_voxel_grid = std::make_unique<map::FastVoxelGrid>(config.voxel_size);
    
    // Initialize 2-Level VoxelMap with precomputed surfels
    m_voxel_map = std::make_unique<map::VoxelMap>(config.map_voxel_size);
    m_voxel_map->SetHierarchyFactor(3);  // L1 = 3×3×3 L0 voxels
    m_voxel_map->SetPlanarityThreshold(config.surfel_planarity_threshold);
    m_voxel_map->SetComputeSurfels(config.use_surfel_correspondence);  // Skip surfel computation if using KDTree
    
    // Initialize legacy voxel filter for map downsampling
    m_voxel_filter = std::make_unique<util::VoxelGrid>();
    m_voxel_filter->setLeafSize(config.voxel_size);
    
    LOG_INFO("[Estimator] Using FastVoxelGrid with stride={}, voxel_size={}", 
                 config.point_stride, config.voxel_size);
    LOG_INFO("[Estimator] Using VoxelMap for incremental map with voxel_size={}", config.map_voxel_size);
    
    // Initialize loop closure detector
    LoopClosureConfig loop_config;
    loop_config.enable_loop_detection = config.loop_enable_loop_detection;
    loop_config.similarity_threshold = config.loop_similarity_threshold;
    loop_config.min_keyframe_gap = config.loop_min_keyframe_gap;
    loop_config.max_search_distance = config.loop_max_search_distance;
    loop_config.enable_debug_output = config.loop_enable_debug_output;
    // Iris parameters are now automatically calculated
    m_loop_detector = std::make_unique<LoopClosureDetector>(loop_config);
    
    // Start background thread for loop detection and PGO
    m_thread_running = true;
    m_loop_pgo_thread = std::thread(&Estimator::loop_pgo_thread_function, this);
}

Estimator::~Estimator() {
    // Stop background thread
    m_thread_running = false;
    m_query_cv.notify_all();  // Wake up thread if waiting
    
    if (m_loop_pgo_thread.joinable()) {
        m_loop_pgo_thread.join();
    }
}

bool Estimator::process_frame(std::shared_ptr<database::LidarFrame> current_frame) {
    auto start_time = std::chrono::high_resolution_clock::now();
    TimingStats timing;

    if (!current_frame || !current_frame->get_raw_cloud()) {
        LOG_WARN("[Estimator] Invalid frame or point cloud");
        return false;
    }
    
    // Step 0: Check and apply pending PGO result from background thread (non-blocking)
    apply_pending_pgo_result_if_available();
    
    // Step 1: Preprocess frame (downsample + feature extraction)
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    if (!preprocess_frame(current_frame)) {
        LOG_ERROR("[Estimator] Frame preprocessing failed");
        return false;
    }
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    timing.preprocessing_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    
    if (!m_initialized) {
        initialize_first_frame(current_frame);
        return true;
    }

    // Get feature cloud from frame
    auto feature_cloud = current_frame->get_feature_cloud();

    // Step 3: Use last keyframe for optimization
    if (!m_last_keyframe) {
        LOG_WARN("[Estimator] No keyframe available, using velocity model only");
        return true;
    }
    
    // Step 4: optimization::IterativeClosestPointOptimizer between current frame and last keyframe
    auto icp_start = std::chrono::high_resolution_clock::now();
    // Calculate initial guess from velocity model: transform from keyframe to current velocity estimate
    SE3f T_keyframe_current_guess = m_previous_frame->get_pose() * m_velocity;
    SE3f T_keyframe_current = estimate_motion_dual_frame(current_frame, m_last_keyframe, T_keyframe_current_guess); 
    auto icp_end = std::chrono::high_resolution_clock::now();
    timing.icp_ms = std::chrono::duration<double, std::milli>(icp_end - icp_start).count();
    
    // Convert result to world coordinate (keyframe pose is already in world coordinates)
    SE3f optimized_pose = T_keyframe_current;
    
    // Store post-optimization cloud in world coordinates for visualization
    auto map_update_start = std::chrono::high_resolution_clock::now();
    
    PointCloudPtr post_opt_cloud_world(new PointCloud());
    Eigen::Matrix4f T_wl_final = optimized_pose.Matrix();
    util::transform_point_cloud(feature_cloud, post_opt_cloud_world, T_wl_final);

    current_frame->set_feature_cloud_global(post_opt_cloud_world); // Cache world coordinate features
    
    auto transform_end = std::chrono::high_resolution_clock::now();
    double transform_ms = std::chrono::duration<double, std::milli>(transform_end - map_update_start).count();
    
    m_T_wl_current = optimized_pose;
    
    // Step 5: Update velocity model
    m_velocity = m_previous_frame->get_pose().Inverse() * m_T_wl_current;


    // Update frame pose and trajectory
    current_frame->set_pose(m_T_wl_current);
    m_trajectory.push_back(m_T_wl_current);
    
    // Set previous keyframe reference for dynamic pose calculation
    auto pose_calc_start = std::chrono::high_resolution_clock::now();
    if (m_last_keyframe) {
        current_frame->set_previous_keyframe(m_last_keyframe);
        // Calculate and store relative pose from keyframe to current frame
        SE3f relative_pose = m_last_keyframe->get_stored_pose().Inverse() * m_T_wl_current;
        current_frame->set_relative_pose(relative_pose);
    }
    auto pose_calc_end = std::chrono::high_resolution_clock::now();
    double pose_calc_ms = std::chrono::duration<double, std::milli>(pose_calc_end - pose_calc_start).count();
    
    // Step 6: Check for keyframe creation
    auto keyframe_start = std::chrono::high_resolution_clock::now();
    bool keyframe_created = false;
    if (should_create_keyframe(m_T_wl_current)) {
        // Transform feature cloud to world coordinates for keyframe storage
        create_keyframe(current_frame);
        keyframe_created = true;
    }
    auto keyframe_end = std::chrono::high_resolution_clock::now();
    double keyframe_ms = std::chrono::duration<double, std::milli>(keyframe_end - keyframe_start).count();
    
    auto map_update_end = std::chrono::high_resolution_clock::now();
    timing.map_update_ms = std::chrono::duration<double, std::milli>(map_update_end - map_update_start).count();
    
    // Update for next iteration - clean up old frame first
    m_old_frame = m_previous_frame;  // Save old frame
    m_previous_frame = current_frame;  // Update to new frame
    
    // Clean up old frame (non-keyframe) - keep only pose info
    if (m_old_frame && !m_old_frame->is_keyframe()) {
        m_old_frame->clear_non_keyframe_data();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    timing.total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Store timing and print statistics every 100 frames
    m_timing_history.push_back(timing);
    m_frame_count++;
    
    if (m_frame_count % 100 == 0) {
        print_timing_statistics();
    }
    
    LOG_DEBUG("[Estimator] Frame {} processed in {:.1f}ms (Preprocess: {:.1f}ms, ICP: {:.1f}ms, MapUpdate: {:.1f}ms)", 
                 m_frame_count, timing.total_ms, timing.preprocessing_ms, timing.icp_ms, timing.map_update_ms);
    
    return true;
}

void Estimator::initialize_first_frame(std::shared_ptr<database::LidarFrame> frame) {
    // Get initial pose from frame (could be set by other sensors)
    m_T_wl_current = frame->get_initial_pose();
    m_velocity = SE3f();      // Identity velocity
    frame->set_pose(m_T_wl_current);
    m_trajectory.push_back(m_T_wl_current);
    
    // First frame has no previous keyframe (it will become the first keyframe)
    frame->set_previous_keyframe(nullptr);
    frame->set_relative_pose(SE3f());  // Identity relative pose

    // Get feature cloud from preprocessed frame
    auto feature_cloud = frame->get_feature_cloud();
    if (!feature_cloud || feature_cloud->empty()) {
        LOG_ERROR("[Estimator] No feature cloud in first frame");
        m_initialized = true;
        return;
    }

    // Transform features to world coordinates using current pose
    PointCloudPtr feature_cloud_world(new PointCloud());
    Eigen::Matrix4f T_wl = m_T_wl_current.Matrix();
    util::transform_point_cloud(feature_cloud, feature_cloud_world, T_wl);
    
    // Set global feature cloud in frame
    frame->set_feature_cloud_global(feature_cloud_world);
    
    // Set as keyframe and initialize local map
    create_keyframe(frame);
    
    m_previous_frame = frame;
    m_last_keyframe_pose = m_T_wl_current;
    
    m_initialized = true;
}

SE3f Estimator::estimate_motion_dual_frame(std::shared_ptr<database::LidarFrame> current_frame,
                                          std::shared_ptr<database::LidarFrame> keyframe,
                                          const SE3f& initial_guess) {
    if (!current_frame || !keyframe) {
        LOG_WARN("[Estimator] Invalid frames for dual frame optimization");
        return initial_guess;
    }
    
    auto current_features = current_frame->get_feature_cloud();
    auto keyframe_local_map = keyframe->get_local_map();
    
    if (!current_features || !keyframe_local_map || current_features->empty() || keyframe_local_map->empty()) {
        LOG_WARN("[Estimator] Invalid feature clouds for dual frame optimization");
        return initial_guess;
    }
    
    // Convert SE3f to SE3f for optimization::IterativeClosestPointOptimizer
    SE3f initial_transform_sophus(initial_guess.RotationMatrix(), initial_guess.Translation());
    SE3f optimized_transform_sophus;

    // Debug log for initial guess (only in debug mode)
    LOG_DEBUG("Check Initial Guess: Translation ({:.3f}, {:.3f}, {:.3f}), Rotation ({:.3f}, {:.3f}, {:.3f})",
                 initial_guess.Translation().x(), initial_guess.Translation().y(), initial_guess.Translation().z(),
                 initial_guess.Rotation().Log().x(), initial_guess.Rotation().Log().y(), initial_guess.Rotation().Log().z());
    
    // Perform ICP optimization using VoxelMap
    bool success = m_icp_optimizer->optimize(
        m_voxel_map.get(),          // VoxelMap with precomputed surfels
        current_frame,              // target frame (current)
        initial_transform_sophus,   // initial relative transform
        optimized_transform_sophus  // optimized relative transform (output)
    );
    
    if (!success) {
        LOG_WARN("[Estimator] Dual frame optimization failed, using initial guess");
        return initial_guess;
    }
    
    // Convert back to SE3f
    SE3f T_keyframe_current(optimized_transform_sophus.RotationMatrix(), optimized_transform_sophus.Translation());
    
    // Collect optimization statistics
    m_total_optimization_iterations += 10; // TODO: Get actual iterations from optimization::IterativeClosestPointOptimizer
    m_total_optimization_time_ms += 1.0;   // TODO: Get actual time from optimization::IterativeClosestPointOptimizer
    m_optimization_call_count++;
    
    LOG_DEBUG("[Estimator] Dual frame optimization completed successfully");
    
    return T_keyframe_current;
}

std::shared_ptr<database::LidarFrame> Estimator::select_best_keyframe(const SE3f& current_pose) {
    if (m_keyframes.empty()) {
        return nullptr;
    }
    
    // Use the most recent (latest) keyframe for temporal consistency
    std::shared_ptr<database::LidarFrame> best_keyframe = nullptr;
    
    // Find the most recent keyframe with a valid local map
    for (auto it = m_keyframes.rbegin(); it != m_keyframes.rend(); ++it) {
        if (*it && (*it)->get_local_map() && !(*it)->get_local_map()->empty()) {
            best_keyframe = *it;
            break;  // Use the most recent valid keyframe
        }
    }
    
    if (best_keyframe) {
        Vector3f translation_diff = current_pose.Translation() - best_keyframe->get_pose().Translation();
        double distance = translation_diff.norm();
        LOG_DEBUG("[Estimator] Selected most recent keyframe at distance {:.2f}m", distance);
    } else {
        LOG_DEBUG("[Estimator] No suitable keyframe found");
    }
    
    return best_keyframe;
}

bool Estimator::should_create_keyframe(const SE3f& current_pose) {
   
    if (m_keyframes.empty()) {
        return true;
    }
    
    // Calculate distance and rotation from last keyframe
    Vector3f translation_diff = current_pose.Translation() - m_last_keyframe_pose.Translation();
    double distance = translation_diff.norm();
    
    SO3f rotation_diff = m_last_keyframe_pose.Rotation().Inverse() * current_pose.Rotation();
    double rotation_angle = rotation_diff.Log().norm();

    // Debug log for keyframe check
    LOG_DEBUG("[Estimator] Keyframe check: Δt={:.2f}m, Δr={:.2f}° (thresholds: {:.2f}m, {:.2f}°)", 
                  distance, rotation_angle * 180.0 / M_PI,
                  m_config.keyframe_distance_threshold, m_config.keyframe_rotation_threshold);
    
    return (distance > m_config.keyframe_distance_threshold || rotation_angle > m_config.keyframe_rotation_threshold);
}

void Estimator::create_keyframe(std::shared_ptr<database::LidarFrame> frame)
{
    auto kf_total_start = std::chrono::high_resolution_clock::now();
    
    // Step A: Set keyframe ID and relative pose
    auto step_a_start = std::chrono::high_resolution_clock::now();
    frame->set_keyframe_id(m_next_keyframe_id++);
    
    // Calculate and store relative pose from previous keyframe
    if (!m_keyframes.empty()) {
        auto previous_keyframe = m_keyframes.back();
        SE3f prev_pose = previous_keyframe->get_pose();
        SE3f curr_pose = frame->get_pose();
        
        // Compute relative pose: T_prev_curr = T_prev^-1 * T_curr
        SE3f relative_pose_raw = prev_pose.Inverse() * curr_pose;
        
        // Normalize rotation matrix for numerical stability
        Eigen::Matrix3f rotation_matrix = relative_pose_raw.RotationMatrix();
        Eigen::Matrix3f normalized_rotation = util::MathUtils::normalize_rotation_matrix(rotation_matrix);
        SE3f relative_pose(normalized_rotation, relative_pose_raw.Translation());
        
        frame->set_relative_pose(relative_pose);
        
        LOG_DEBUG("[Estimator] Set relative pose for keyframe {}: t_norm={:.3f}, r_norm={:.3f}°", 
                     frame->get_keyframe_id(), 
                     relative_pose.Translation().norm(),
                     relative_pose.Rotation().Log().norm() * 180.0f / M_PI);
        
        // Incremental PGO: add keyframe with odometry constraint
        if (m_config.pgo_enable_pgo) {
            m_pose_graph_optimizer->add_keyframe_with_odom(
                previous_keyframe->get_keyframe_id(),
                frame->get_keyframe_id(),
                frame->get_pose(),
                relative_pose,
                m_config.pgo_odometry_translation_noise,
                m_config.pgo_odometry_rotation_noise
            );
        }
    } else {
        // First keyframe: set identity relative pose
        frame->set_relative_pose(SE3f());
        LOG_DEBUG("[Estimator] First keyframe: set identity relative pose");
        
        // Incremental PGO: add first keyframe with prior
        if (m_config.pgo_enable_pgo) {
            m_pose_graph_optimizer->add_first_keyframe(
                frame->get_keyframe_id(),
                frame->get_pose()
            );
        }
    }
    
    // Add to keyframes list (thread-safe)
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        m_keyframes.push_back(frame);
    }
    auto step_a_end = std::chrono::high_resolution_clock::now();
    double step_a_ms = std::chrono::duration<double, std::milli>(step_a_end - step_a_start).count();

    // Check if frame has global feature cloud
    auto global_feature_cloud = frame->get_feature_cloud_global();
    if (!global_feature_cloud) {
        LOG_WARN("[Estimator] Frame has no global feature cloud, using local feature cloud");
        // For first frame, transform local features to global (identity transform)
        auto local_feature_cloud = frame->get_feature_cloud();
        if (local_feature_cloud && !local_feature_cloud->empty()) {
            PointCloudPtr global_cloud = std::make_shared<PointCloud>();
            *global_cloud = *local_feature_cloud;  // Copy for first frame
            frame->set_feature_cloud_global(global_cloud);
            global_feature_cloud = global_cloud;
        } else {
            LOG_ERROR("[Estimator] Frame has no feature clouds at all!");
            return;
        }
    }

    // ========== VoxelMap-based incremental map update ==========
    auto voxelmap_start = std::chrono::high_resolution_clock::now();
    
    // Update VoxelMap: add new points and remove distant voxels
    Eigen::Vector3f current_position = frame->get_pose().Translation();
    Eigen::Vector3d sensor_position = current_position.cast<double>();
    double max_distance = m_config.max_range * 1.2;
    
    m_voxel_map->UpdateVoxelMap(global_feature_cloud, sensor_position, max_distance, true);
    
    // Rebuild KDTree if using KDTree-based correspondence
    if (!m_config.use_surfel_correspondence) {
        m_voxel_map->RebuildKdTree();
    }
    
    auto voxelmap_end = std::chrono::high_resolution_clock::now();
    double voxelmap_ms = std::chrono::duration<double, std::milli>(voxelmap_end - voxelmap_start).count();
    
    // Get point cloud for local_map (for visualization and compatibility)
    auto getcloud_start = std::chrono::high_resolution_clock::now();
    auto local_map_cloud = m_voxel_map->GetPointCloud();
    frame->set_local_map(local_map_cloud);
    auto getcloud_end = std::chrono::high_resolution_clock::now();
    double getcloud_ms = std::chrono::duration<double, std::milli>(getcloud_end - getcloud_start).count();
    
    // Sliding window cleanup: keep last N keyframes with full data
    int current_kf_id = frame->get_keyframe_id();
    int window_size = m_config.keyframe_window_size;
    int oldest_to_keep = current_kf_id - window_size + 1;
    
    if (oldest_to_keep > 0) {
        int kf_to_clear = oldest_to_keep - 1;
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        for (const auto& kf : m_keyframes) {
            if (kf->get_keyframe_id() == kf_to_clear) {
                kf->clear_heavy_data_for_old_keyframe();
                LOG_DEBUG("[Estimator] Cleared heavy data for keyframe {} (outside window [{}, {}])", 
                             kf_to_clear, oldest_to_keep, current_kf_id);
                break;
            }
        }
    }
    
    // Update last keyframe reference
    m_last_keyframe = frame;
    m_last_keyframe_pose = m_last_keyframe->get_pose();
    
    // Add keyframe to loop detector database and query queue for async processing
    auto loop_start = std::chrono::high_resolution_clock::now();
    if (m_loop_detector && m_config.loop_enable_loop_detection) {
        m_loop_detector->add_keyframe(frame);
        
        int current_keyframe_id = frame->get_keyframe_id();
        int keyframes_since_last_loop = current_keyframe_id - m_last_successful_loop_keyframe_id;
        bool allow_detection = (keyframes_since_last_loop >= m_config.loop_min_keyframe_gap);
        
        if (allow_detection) {
            {
                std::lock_guard<std::mutex> lock(m_query_mutex);
                m_loop_query_queue.push_back(current_keyframe_id);
                LOG_DEBUG("[Estimator] Added KF {} to loop query queue (queue size: {})", 
                             current_keyframe_id, m_loop_query_queue.size());
            }
            m_query_cv.notify_one();
        } else {
            LOG_DEBUG("[Estimator] Loop detection skipped: only {} keyframes since last loop (need {})",
                         keyframes_since_last_loop, m_config.loop_min_keyframe_gap);
        }
    }
    auto loop_end = std::chrono::high_resolution_clock::now();
    double loop_ms = std::chrono::duration<double, std::milli>(loop_end - loop_start).count();
    
    auto kf_total_end = std::chrono::high_resolution_clock::now();
    double kf_total_ms = std::chrono::duration<double, std::milli>(kf_total_end - kf_total_start).count();
    
    // Suppress unused variable warnings
    (void)step_a_ms;
    (void)voxelmap_ms;
    (void)getcloud_ms;
    (void)loop_ms;
    (void)kf_total_ms;
}


void Estimator::update_config(const util::SystemConfig& config) {
    m_config = config;
    
    // Update voxel filter
    m_voxel_filter->setLeafSize(m_config.voxel_size);
}

const util::SystemConfig& Estimator::get_config() const {
    return m_config;
}

PointCloudConstPtr Estimator::get_local_map() const {

    if(m_last_keyframe) {
        return m_last_keyframe->get_local_map();
    }  
    else
    {
        return nullptr;
    }
}


void Estimator::get_debug_clouds(PointCloudConstPtr& pre_icp_cloud, PointCloudConstPtr& post_icp_cloud) const {
    pre_icp_cloud = m_debug_pre_icp_cloud;
    post_icp_cloud = m_debug_post_icp_cloud;
}

bool Estimator::preprocess_frame(std::shared_ptr<database::LidarFrame> frame) {
    auto raw_cloud = frame->get_raw_cloud();
    if (!raw_cloud || raw_cloud->empty()) {
        LOG_ERROR("[Estimator] Invalid raw cloud");
        return false;
    }
    
    // Step 1: Fast voxel grid downsampling with stride (Morton code + Robin Hood hashing)
    PointCloudPtr downsampled_cloud = std::make_shared<PointCloud>();
    m_fast_voxel_grid->filter(*raw_cloud, *downsampled_cloud, m_config.point_stride);
    
    if (downsampled_cloud->empty()) {
        LOG_ERROR("[Estimator] Downsampled cloud is empty");
        return false;
    }
    
    // Step 2: Use downsampled cloud directly as feature cloud (no separate feature extraction)
    // Feature extraction is removed - ICP will compute normals from correspondences
    
    // Step 3: Set processed clouds in the frame
    frame->set_processed_cloud(downsampled_cloud);
    frame->set_feature_cloud(downsampled_cloud);  // Use same cloud as features
    
    LOG_DEBUG("[Estimator] Preprocessing: {} -> {} points (stride={}, voxels={})", 
                  raw_cloud->size(), downsampled_cloud->size(), 
                  m_config.point_stride, m_fast_voxel_grid->getVoxelCount());
    
    return true;
}

void Estimator::get_optimization_statistics(double& avg_iterations, double& avg_time_ms) const {
    if (m_optimization_call_count > 0) {
        avg_iterations = static_cast<double>(m_total_optimization_iterations) / m_optimization_call_count;
        avg_time_ms = m_total_optimization_time_ms / m_optimization_call_count;
    } else {
        avg_iterations = 0.0;
        avg_time_ms = 0.0;
    }
}

const SE3f& Estimator::get_current_pose() const {
    return m_T_wl_current;
}

size_t Estimator::get_keyframe_count() const {
    return m_keyframes.size();
}

std::shared_ptr<database::LidarFrame> Estimator::get_keyframe(size_t index) const {
    if (index >= m_keyframes.size()) {
        return nullptr;
    }
    return m_keyframes[index];
}

void Estimator::enable_loop_closure(bool enable) {
    if (m_loop_detector) {
        LoopClosureConfig config = m_loop_detector->get_config();
        config.enable_loop_detection = enable;
        m_loop_detector->update_config(config);
        LOG_INFO("[Estimator] Loop closure detection {}", enable ? "enabled" : "disabled");
    }
}

void Estimator::set_loop_closure_config(const LoopClosureConfig& config) {
    if (m_loop_detector) {
        m_loop_detector->update_config(config);
    }
}

size_t Estimator::get_loop_closure_count() const {
    // For now, return 0 since we haven't implemented PGO yet
    // This will be updated when we add pose graph optimization
    return 0;
}

std::map<int, Eigen::Matrix4f> Estimator::get_optimized_trajectory() const {
    std::map<int, Eigen::Matrix4f> result;
    for (const auto& [id, pose] : m_optimized_poses) {
        result[id] = pose.Matrix();
    }
    return result;
}

void Estimator::process_loop_closures(std::shared_ptr<database::LidarFrame> current_keyframe, 
                                     const std::vector<LoopCandidate>& loop_candidates) {
    
    if (loop_candidates.empty()) {
        return;
    }
    
    // Use only the best candidate (first one, already sorted by similarity score)
    const auto& candidate = loop_candidates[0];
    
    LOG_INFO("[Estimator] Processing best loop closure candidate for ICP optimization");
    
    // Find the matched keyframe in our database
    std::shared_ptr<database::LidarFrame> matched_keyframe = nullptr;
    
    for (const auto& kf : m_keyframes) {
        if (static_cast<size_t>(kf->get_keyframe_id()) == candidate.match_keyframe_id) {
            matched_keyframe = kf;
            break;
        }
    }
    
    if (!matched_keyframe) {
        LOG_WARN("[Estimator] Could not find matched keyframe {} in database", candidate.match_keyframe_id);
        return;
    }

    // Get feature clouds from both keyframes
    auto local_feature_matched = matched_keyframe->get_feature_cloud();
    auto local_feature_current = current_keyframe->get_feature_cloud();

    if (!local_feature_matched || !local_feature_current ||
        local_feature_matched->empty() || local_feature_current->empty()) {
        LOG_WARN("[Estimator] Empty feature clouds for loop {} <-> {}", 
                    candidate.query_keyframe_id, candidate.match_keyframe_id);
        return;
    }

    SE3f T_current_l2l;
    float inlier_ratio = 0.0f;

    bool icp_success = m_icp_optimizer->optimize_loop(
        current_keyframe,             // source frame (has fresh kdtree built)
        matched_keyframe,             // target frame (will use local map as features)
        T_current_l2l,                // optimized relative transform (output)
        inlier_ratio                  // inlier ratio (output)
    );

    if (!icp_success) {
        LOG_WARN("[Estimator] Loop closure ICP failed for {} <-> {}", 
                    candidate.query_keyframe_id, candidate.match_keyframe_id);
        return;
    }
    
    // Validate loop closure using inlier ratio
    const float min_inlier_ratio = 0.3f;  // Minimum 30% inliers required
    if (inlier_ratio < min_inlier_ratio) {
        LOG_WARN("[Estimator] Loop closure rejected: inlier ratio {:.2f}% < {:.2f}% for {} <-> {}", 
                    inlier_ratio * 100.0f, min_inlier_ratio * 100.0f,
                    candidate.query_keyframe_id, candidate.match_keyframe_id);
        return;
    }

    // T_current_l2l is the ICP correction: T_original^-1 * T_corrected
    // ICP optimizes curr_keyframe pose, returns correction transform
    
    // Get current poses (with drift)
    SE3f T_world_current = current_keyframe->get_pose();
    SE3f T_world_matched = matched_keyframe->get_pose();
    
    // Apply ICP correction: T_corrected = T_correction * T_original
    // ICP returns: T_correction = T_original^-1 * T_optimized
    // So: T_corrected = (T_original^-1 * T_optimized) * T_original = T_optimized
    SE3f T_current_corrected = T_world_current * T_current_l2l;
    
    // Calculate pose difference for logging (how much correction ICP suggests)
    SE3f pose_diff = T_world_current.Inverse() * T_current_corrected;
    float translation_diff = pose_diff.Translation().norm();
    float rotation_diff = pose_diff.Rotation().Log().norm() * 180.0f / M_PI;
    
    LOG_INFO("[Estimator] Loop closure ICP success {} <-> {}: Δt={:.3f}m, Δr={:.2f}°, inliers={:.1f}%",
                candidate.query_keyframe_id, candidate.match_keyframe_id,
                translation_diff, rotation_diff, inlier_ratio * 100.0f);
    
    // Compute relative pose constraint: from matched to current
    // Using GTSAM's between() logic: poseFrom.between(poseTo) = poseFrom^-1 * poseTo
    SE3f T_matched_to_current = T_world_matched.Inverse() * T_current_corrected;
    
    // Check if PGO is enabled
    if (!m_config.pgo_enable_pgo) {
        LOG_INFO("[Estimator] PGO disabled, skipping pose graph optimization");
        return;
    }
    
    // ========== Incremental ISAM2-based PGO (LIO-SAM pattern) ==========
    LOG_INFO("[PGO-ISAM2] Adding loop closure and optimizing incrementally");
    
    // Store pre-PGO poses for visualization (before optimization)
    std::map<int, SE3f> pre_pgo_poses;
    for (const auto& kf : m_keyframes) {
        pre_pgo_poses[kf->get_keyframe_id()] = kf->get_stored_pose();
    }
    
    // Store loop constraint for logging
    LoopConstraint loop_constraint;
    loop_constraint.from_keyframe_id = matched_keyframe->get_keyframe_id();
    loop_constraint.to_keyframe_id = current_keyframe->get_keyframe_id();
    loop_constraint.relative_pose = T_matched_to_current;
    loop_constraint.translation_noise = m_config.pgo_loop_translation_noise;
    loop_constraint.rotation_noise = m_config.pgo_loop_rotation_noise;
    m_loop_constraints.push_back(loop_constraint);
    
    LOG_INFO("[PGO-ISAM2] Loop closure: {} -> {} (total loops: {})",
                matched_keyframe->get_keyframe_id(), 
                current_keyframe->get_keyframe_id(),
                m_loop_constraints.size());
    
    // Run pose graph optimization with loop closure
    bool opt_success = m_pose_graph_optimizer->add_loop_and_optimize(
        matched_keyframe->get_keyframe_id(),
        current_keyframe->get_keyframe_id(),
        T_matched_to_current,
        m_config.pgo_loop_translation_noise,
        m_config.pgo_loop_rotation_noise
    );
    
    if (opt_success) {
        // Get optimized poses
        std::map<int, SE3f> optimized_poses = m_pose_graph_optimizer->get_all_optimized_poses();
        
        LOG_INFO("[PGO-ISAM2] ========== PGO Results ==========");
        LOG_INFO("[PGO-ISAM2] Total keyframes optimized: {}", optimized_poses.size());
        
        float max_translation_diff = 0.0f;
        float max_rotation_diff = 0.0f;
        float avg_translation_diff = 0.0f;
        float avg_rotation_diff = 0.0f;
        int count = 0;
        
        for (size_t i = 0; i < m_keyframes.size(); i++) {
            int kf_id = m_keyframes[i]->get_keyframe_id();
            auto it = optimized_poses.find(kf_id);
            
            if (it != optimized_poses.end()) {
                SE3f old_pose = pre_pgo_poses[kf_id];
                SE3f new_pose = it->second;
                
                float translation_diff = (new_pose.Translation() - old_pose.Translation()).norm();
                float rotation_diff = (new_pose.Rotation().Log() - old_pose.Rotation().Log()).norm() * 180.0f / M_PI;
                
                max_translation_diff = std::max(max_translation_diff, translation_diff);
                max_rotation_diff = std::max(max_rotation_diff, rotation_diff);
                avg_translation_diff += translation_diff;
                avg_rotation_diff += rotation_diff;
                count++;
            }
        }
        
        if (count > 0) {
            avg_translation_diff /= count;
            avg_rotation_diff /= count;
            
            LOG_INFO("[PGO-ISAM2] Average correction: Δt={:.3f}m, Δr={:.2f}°", avg_translation_diff, avg_rotation_diff);
            LOG_INFO("[PGO-ISAM2] Maximum correction: Δt={:.3f}m, Δr={:.2f}°", max_translation_diff, max_rotation_diff);
        }
        
        LOG_INFO("[PGO-ISAM2] =========================================");
        
        // Store optimized poses for visualization
        m_optimized_poses = optimized_poses;
        
        // Apply pose graph optimization results to all keyframes
        LOG_INFO("[PGO-ISAM2] Applying corrections to all keyframes...");
        apply_pose_graph_optimization();
        
        // Update cooldown
        m_last_successful_loop_keyframe_id = current_keyframe->get_keyframe_id();
        LOG_INFO("[PGO-ISAM2] Loop closure cooldown activated: next detection after keyframe {}",
                    m_last_successful_loop_keyframe_id + m_config.loop_min_keyframe_gap);
    } else {
        LOG_ERROR("[PGO-ISAM2] Pose graph optimization failed!");
    }
}

void Estimator::apply_pose_graph_optimization() {
    // Get all optimized poses from pose graph optimizer
    std::map<int, SE3f> optimized_poses = m_pose_graph_optimizer->get_all_optimized_poses();
    
    if (optimized_poses.empty()) {
        LOG_WARN("[Estimator] No optimized poses available from pose graph!");
        return;
    }
    
    LOG_INFO("[Estimator] Applying PGO results to {} keyframes", optimized_poses.size());

    auto last_keyframe = m_keyframes.back();

    SE3f last_keyframe_pose_before_opt = last_keyframe->get_pose();
    SE3f last_keyframe_pose_after_opt = optimized_poses[last_keyframe->get_keyframe_id()];
    SE3f total_correction = last_keyframe_pose_after_opt * last_keyframe_pose_before_opt.Inverse();
    
    // Update all keyframe poses (absolute poses only)
    // NOTE: We do NOT recalculate relative poses after PGO!
    // The PGO optimization already considers relative pose constraints,
    // so recalculating and normalizing them would distort the optimization result.
    for (auto& keyframe : m_keyframes) {
        int kf_id = keyframe->get_keyframe_id();
        
        auto it = optimized_poses.find(kf_id);
        if (it == optimized_poses.end()) {
            LOG_WARN("[Estimator] No optimized pose for keyframe {}", kf_id);
            continue;
        }
        
        SE3f old_pose = keyframe->get_pose();
        SE3f new_pose = it->second;
        
        float translation_diff = (new_pose.Translation() - old_pose.Translation()).norm();
        float rotation_diff = (new_pose.Rotation().Log() - old_pose.Rotation().Log()).norm() * 180.0f / M_PI;
        
        LOG_DEBUG("[Estimator] Keyframe {}: Δt={:.3f}m, Δr={:.2f}°", 
                     kf_id, translation_diff, rotation_diff);
        
        // Update keyframe pose (absolute pose)
        keyframe->set_pose(new_pose);
    }
    
    // Relative poses are kept as-is (not recalculated from absolute poses)
    // This preserves the optimization result from PGO

    // Apply correction to VoxelMap (transform centroids and re-hash)
    auto voxelmap_correction_start = std::chrono::high_resolution_clock::now();
    m_voxel_map->ApplyTransformAndRehash(total_correction.Matrix());
    auto voxelmap_correction_end = std::chrono::high_resolution_clock::now();
    double voxelmap_correction_ms = std::chrono::duration<double, std::milli>(voxelmap_correction_end - voxelmap_correction_start).count();
    
    LOG_INFO("[Estimator] VoxelMap correction applied in {:.2f}ms (L0={} L1={} voxels)", 
                 voxelmap_correction_ms, m_voxel_map->GetVoxelCount(), m_voxel_map->GetL1VoxelCount());

    // Update last keyframe's local map from VoxelMap (for visualization only)
    auto local_map_cloud = m_voxel_map->GetPointCloud();
    last_keyframe->set_local_map(local_map_cloud);
    // Note: KdTree no longer built - ICP uses VoxelMap directly
}

void Estimator::loop_pgo_thread_function() {
    while (m_thread_running) {
        int query_kf_id = -1;
        std::shared_ptr<database::LidarFrame> query_keyframe = nullptr;
        
        // Wait for query or termination signal
        {
            std::unique_lock<std::mutex> lock(m_query_mutex);
            m_query_cv.wait(lock, [this] { 
                return !m_loop_query_queue.empty() || !m_thread_running; 
            });
            
            if (!m_thread_running) break;
            
            // If PGO is in progress, skip processing (wait for next wake-up)
            if (m_pgo_in_progress) {
                LOG_DEBUG("[Background] PGO in progress, skipping queries");
                continue;
            }
            
            // Get most recent query and clear the rest (for real-time performance)
            query_kf_id = m_loop_query_queue.back();
            m_loop_query_queue.clear();
            LOG_DEBUG("[Background] Processing loop query for KF {}", query_kf_id);
        }
        
        // Find the keyframe (read-only access with lock)
        {
            std::lock_guard<std::mutex> lock(m_keyframes_mutex);
            for (const auto& kf : m_keyframes) {
                if (kf->get_keyframe_id() == query_kf_id) {
                    query_keyframe = kf;
                    break;
                }
            }
        }
        
        if (!query_keyframe) {
            LOG_WARN("[Background] Keyframe {} not found", query_kf_id);
            continue;
        }
        
        // Detect loop closure candidates
        auto loop_candidates = m_loop_detector->detect_loop_closures(query_keyframe);
        
        if (loop_candidates.empty()) {
            LOG_DEBUG("[Background] No loop candidates found for KF {}", query_kf_id);
            continue;
        }
        
        // Loop detected! Start PGO
        m_pgo_in_progress = true;
        
        // Process loop closure (ICP optimization + PGO)
        bool pgo_success = run_pgo_for_loop(query_keyframe, loop_candidates);
        
        m_pgo_in_progress = false;
        
        if (pgo_success) {
            
            // Clear accumulated queries during PGO (they're outdated now)
            {
                std::lock_guard<std::mutex> lock(m_query_mutex);
                m_loop_query_queue.clear();
            }
        }
    }
}

bool Estimator::run_pgo_for_loop(
    std::shared_ptr<database::LidarFrame> current_keyframe,
    const std::vector<LoopCandidate>& loop_candidates) 
{
    // Use only the best candidate (first one, already sorted by similarity score)
    const auto& candidate = loop_candidates[0];
    
    // Find the matched keyframe
    std::shared_ptr<database::LidarFrame> matched_keyframe = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        for (const auto& kf : m_keyframes) {
            if (static_cast<size_t>(kf->get_keyframe_id()) == candidate.match_keyframe_id) {
                matched_keyframe = kf;
                break;
            }
        }
    }

    if (!matched_keyframe)
    {
        LOG_WARN("[Background] Could not find matched keyframe {}",
                     candidate.match_keyframe_id);
        return false;
    }

    // Get feature clouds from both keyframes
    auto local_feature_matched = matched_keyframe->get_feature_cloud();
    auto local_feature_current = current_keyframe->get_feature_cloud();

    if (!local_feature_matched || !local_feature_current ||
        local_feature_matched->empty() || local_feature_current->empty())
    {
        LOG_WARN("[Background] Empty feature clouds for loop {} <-> {}",
                     candidate.query_keyframe_id, candidate.match_keyframe_id);
        return false;
    }

    // Perform ICP optimization for loop closure
    SE3f T_current_l2l;
    float inlier_ratio = 0.0f;

    bool icp_success = m_icp_optimizer->optimize_loop(
        current_keyframe,
        matched_keyframe,
        T_current_l2l,
        inlier_ratio
    );

    if (!icp_success) {
        LOG_WARN("[Background] Loop ICP failed {} <-> {}", 
                    candidate.query_keyframe_id, candidate.match_keyframe_id);
        return false;
    }
    
    // Validate loop closure using inlier ratio
    const float min_inlier_ratio = 0.3f;
    if (inlier_ratio < min_inlier_ratio) {
        LOG_WARN("[Background] Loop rejected: {:.1f}% inliers < {:.1f}%", 
                    inlier_ratio * 100.0f, min_inlier_ratio * 100.0f);
        return false;
    }

    // Get current poses
    SE3f T_world_current = current_keyframe->get_pose();
    SE3f T_world_matched = matched_keyframe->get_pose();
    
    // Apply ICP correction
    SE3f T_current_corrected = T_world_current * T_current_l2l;
    
    // Calculate pose difference
    SE3f pose_diff = T_world_current.Inverse() * T_current_corrected;
    float translation_diff = pose_diff.Translation().norm();
    float rotation_diff = pose_diff.Rotation().Log().norm() * 180.0f / M_PI;
    
    LOG_DEBUG("[Background] Loop detected {} <-> {}: Δt={:.2f}m, Δr={:.2f}°, {:.1f}% inliers",
                candidate.query_keyframe_id, candidate.match_keyframe_id,
                translation_diff, rotation_diff, inlier_ratio * 100.0f);
    
    // Compute relative pose constraint
    SE3f T_matched_to_current = T_world_matched.Inverse() * T_current_corrected;
    
    // Check if PGO is enabled
    if (!m_config.pgo_enable_pgo) {
        LOG_DEBUG("[Background] PGO disabled");
        return false;
    }
    
    // ========== Incremental ISAM2-based PGO (LIO-SAM pattern) ==========
    
    // Store loop constraint for logging
    LoopConstraint loop_constraint;
    loop_constraint.from_keyframe_id = matched_keyframe->get_keyframe_id();
    loop_constraint.to_keyframe_id = current_keyframe->get_keyframe_id();
    loop_constraint.relative_pose = T_matched_to_current;
    loop_constraint.translation_noise = m_config.pgo_loop_translation_noise;
    loop_constraint.rotation_noise = m_config.pgo_loop_rotation_noise;
    m_loop_constraints.push_back(loop_constraint);
    
    // Snapshot keyframe poses before optimization
    std::vector<int> kf_ids;
    std::vector<SE3f> kf_poses_before;
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        kf_ids.reserve(m_keyframes.size());
        kf_poses_before.reserve(m_keyframes.size());
        for (const auto& kf : m_keyframes) {
            kf_ids.push_back(kf->get_keyframe_id());
            kf_poses_before.push_back(kf->get_pose());
        }
    }
    
    // Run pose graph optimization with loop closure
    bool opt_success = m_pose_graph_optimizer->add_loop_and_optimize(
        matched_keyframe->get_keyframe_id(),
        current_keyframe->get_keyframe_id(),
        T_matched_to_current,
        m_config.pgo_loop_translation_noise,
        m_config.pgo_loop_rotation_noise
    );
    
    if (!opt_success) {
        LOG_ERROR("[Background] PGO failed!");
        return false;
    }
    
    // Get optimized poses and calculate statistics
    std::map<int, SE3f> optimized_poses = m_pose_graph_optimizer->get_all_optimized_poses();
    
    float avg_trans_diff = 0.0f;
    float avg_rot_diff = 0.0f;
    float max_trans_diff = 0.0f;
    float max_rot_diff = 0.0f;
    
    for (size_t i = 0; i < kf_ids.size(); ++i) {
        auto it = optimized_poses.find(kf_ids[i]);
        if (it != optimized_poses.end()) {
            SE3f old_pose = kf_poses_before[i];
            SE3f new_pose = it->second;
            
            float trans_diff = (new_pose.Translation() - old_pose.Translation()).norm();
            float rot_diff = (new_pose.Rotation().Log() - old_pose.Rotation().Log()).norm() * 180.0f / M_PI;
            
            avg_trans_diff += trans_diff;
            avg_rot_diff += rot_diff;
            max_trans_diff = std::max(max_trans_diff, trans_diff);
            max_rot_diff = std::max(max_rot_diff, rot_diff);
        }
    }
    
    if (!kf_ids.empty()) {
        avg_trans_diff /= kf_ids.size();
        avg_rot_diff /= kf_ids.size();
    }
    
    LOG_DEBUG("[Background] PGO completed: {} KFs, avg Δ({:.3f}m, {:.2f}°), max Δ({:.3f}m, {:.2f}°)",
                 kf_ids.size(), avg_trans_diff, avg_rot_diff, max_trans_diff, max_rot_diff);
    
    // Calculate correction transform for last keyframe
    int last_kf_id = kf_ids.back();
    SE3f last_kf_pose_before = kf_poses_before.back();
    SE3f last_kf_pose_after = optimized_poses[last_kf_id];
    SE3f last_kf_correction = last_kf_pose_after * last_kf_pose_before.Inverse();
    
    // Prepare PGO result for main thread
    PGOResult result;
    result.last_optimized_kf_id = last_kf_id;
    result.optimized_poses = std::move(optimized_poses);
    result.last_kf_correction = last_kf_correction;
    result.timestamp = std::chrono::steady_clock::now();
    
    // Put result in queue
    {
        std::lock_guard<std::mutex> lock(m_result_mutex);
        m_pending_result = std::move(result);
    }
    
    return true;
}

void Estimator::apply_pending_pgo_result_if_available() {
    // Check if there's a pending result (non-blocking)
    std::optional<PGOResult> result;
    {
        std::lock_guard<std::mutex> lock(m_result_mutex);
        if (m_pending_result.has_value()) {
            result = std::move(m_pending_result);
            m_pending_result.reset();
        }
    }
    
    if (!result) return;  // No pending result
    
    // Apply PGO result
    int last_optimized_id = result->last_optimized_kf_id;
    
    LOG_DEBUG("[Main] Applying PGO result ({} keyframes optimized)", 
                 result->optimized_poses.size());
    
    // Step 1: Update poses for keyframes included in PGO
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        for (auto& kf : m_keyframes) {
            int kf_id = kf->get_keyframe_id();
            
            if (kf_id <= last_optimized_id) {
                // Keyframe was included in PGO - update with optimized pose
                auto it = result->optimized_poses.find(kf_id);
                if (it != result->optimized_poses.end()) {
                    kf->set_pose(it->second);
                }
            } else {
                // Keyframe was added after PGO started - will be updated by propagation
                break;
            }
        }
    }
    
    // Step 2: Propagate poses to keyframes added after PGO
    propagate_poses_after_pgo(last_optimized_id);
    
    // Step 3: Apply correction to VoxelMap (critical for ICP to work correctly)
    m_voxel_map->ApplyTransformAndRehash(result->last_kf_correction.Matrix());
    
    // Step 4: Update last keyframe's local map from VoxelMap (for visualization)
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        if (!m_keyframes.empty()) {
            auto local_map_cloud = m_voxel_map->GetPointCloud();
            m_keyframes.back()->set_local_map(local_map_cloud);
        }
    }
    
    // Update cooldown
    m_last_successful_loop_keyframe_id = last_optimized_id;
}

void Estimator::propagate_poses_after_pgo(int last_optimized_kf_id) {
    std::lock_guard<std::mutex> lock(m_keyframes_mutex);
    
    // Find the last optimized keyframe
    std::shared_ptr<database::LidarFrame> last_optimized_kf = nullptr;
    SE3f accumulated_pose;
    bool found_start = false;
    
    for (auto& kf : m_keyframes) {
        if (kf->get_keyframe_id() == last_optimized_kf_id) {
            last_optimized_kf = kf;
            accumulated_pose = kf->get_pose();
            found_start = true;
            continue;
        }
        
        if (!found_start) continue;
        
        // Propagate pose using relative transform: new_pose = prev_pose * relative
        SE3f relative = kf->get_relative_pose();
        accumulated_pose = accumulated_pose * relative;
        
        kf->set_pose(accumulated_pose);
    }
    
    if (!found_start) {
        LOG_WARN("[Main] Could not find last optimized KF {} for propagation", 
                     last_optimized_kf_id);
    }
}

void Estimator::transform_current_keyframe_map(const SE3f& correction) {
    std::lock_guard<std::mutex> lock(m_keyframes_mutex);
    
    if (m_keyframes.empty()) return;
    
    // Transform only the most recent keyframe's map
    auto current_kf = m_keyframes.back();
    auto local_map = current_kf->get_local_map();
    
    if (!local_map || local_map->empty()) {
        LOG_DEBUG("[Main] Current keyframe has no local map to transform");
        return;
    }
    
    util::PointCloudPtr transformed_map = std::make_shared<util::PointCloud>();
    util::transform_point_cloud(local_map, transformed_map, correction.Matrix());
    
    current_kf->set_local_map(transformed_map);
    // Note: KdTree no longer built - ICP uses VoxelMap directly
}

bool Estimator::save_map_to_ply(const std::string& output_path, float voxel_size) {
    std::lock_guard<std::mutex> lock(m_keyframes_mutex);
    
    if (m_keyframes.empty()) {
        LOG_WARN("[Estimator] No keyframes to save");
        return false;
    }
    
    LOG_INFO("[Estimator] Building final map from {} keyframes...", m_keyframes.size());
    
    // Accumulate all keyframe feature clouds in world coordinates
    util::PointCloudPtr accumulated_map = std::make_shared<util::PointCloud>();
    
    for (const auto& kf : m_keyframes) {
        auto feature_cloud = kf->get_feature_cloud();
        if (!feature_cloud || feature_cloud->empty()) {
            continue;
        }
        
        // Transform feature cloud to world coordinates
        SE3f pose = kf->get_pose();
        util::PointCloudPtr transformed_cloud = std::make_shared<util::PointCloud>();
        util::transform_point_cloud(feature_cloud, transformed_cloud, pose.Matrix());
        
        // Accumulate
        *accumulated_map += *transformed_cloud;
    }
    
    if (accumulated_map->empty()) {
        LOG_ERROR("[Estimator] No points in accumulated map");
        return false;
    }
    
    LOG_INFO("[Estimator] Accumulated map: {} points", accumulated_map->size());
    
    // Downsample if voxel_size > 0
    util::PointCloudPtr final_map = accumulated_map;
    if (voxel_size > 0.0f) {
        util::VoxelGrid voxel_filter;
        voxel_filter.setLeafSize(voxel_size);
        voxel_filter.setInputCloud(accumulated_map);
        
        final_map = std::make_shared<util::PointCloud>();
        voxel_filter.filter(*final_map);
        
        LOG_INFO("[Estimator] Downsampled map: {} -> {} points (voxel_size={})", 
                     accumulated_map->size(), final_map->size(), voxel_size);
    }
    
    // Save as PLY binary format
    if (!util::save_point_cloud_ply(output_path, final_map)) {
        LOG_ERROR("[Estimator] Failed to save map to {}", output_path);
        return false;
    }
    
    LOG_INFO("[Estimator] Saved final map to {} ({} points)", output_path, final_map->size());
    return true;
}

bool Estimator::save_map_to_pcd(const std::string& output_path, float voxel_size) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Lock and prepare data
    auto lock_start = std::chrono::high_resolution_clock::now();
    std::deque<std::shared_ptr<database::LidarFrame>> keyframes_copy;
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        if (m_keyframes.empty()) {
            LOG_WARN("[Estimator] No keyframes to save");
            return false;
        }
        keyframes_copy = m_keyframes;  // Copy list (fast, small overhead)
    }
    auto lock_end = std::chrono::high_resolution_clock::now();
    double lock_ms = std::chrono::duration<double, std::milli>(lock_end - lock_start).count();

    LOG_INFO("[Estimator] Building final map (PCD) from {} keyframes...", keyframes_copy.size());

    // Step 2: Pre-allocate (measure separately)
    auto alloc_start = std::chrono::high_resolution_clock::now();
    size_t total_points = 0;
    for (const auto& kf : keyframes_copy) {
        auto feature_cloud = kf->get_feature_cloud();
        if (feature_cloud) total_points += feature_cloud->size();
    }
    
    util::PointCloudPtr final_map = std::make_shared<util::PointCloud>();
    final_map->reserve(total_points);
    auto alloc_end = std::chrono::high_resolution_clock::now();
    double alloc_ms = std::chrono::duration<double, std::milli>(alloc_end - alloc_start).count();
    LOG_INFO("[Estimator] Pre-allocated {} points (keyframes already downsampled)", total_points);
    
    // Step 3: Transform and accumulate (this is the hot loop - optimize aggressively)
    auto transform_start = std::chrono::high_resolution_clock::now();
    
    // Use += operator for batch accumulation (avoids element-wise operations)
    for (const auto& kf : keyframes_copy) {
        auto feature_cloud = kf->get_feature_cloud();
        if (!feature_cloud || feature_cloud->empty()) continue;

        const SE3f& pose = kf->get_pose();
        const Eigen::Matrix4f transform = pose.Matrix();
        
        // Extract rotation and translation for faster point transformation
        Eigen::Matrix3f R = transform.block<3, 3>(0, 0);
        Eigen::Vector3f t = transform.block<3, 1>(0, 3);
        
        // Transform cloud and accumulate using += operator
        util::PointCloudPtr transformed_cloud = std::make_shared<util::PointCloud>();
        transformed_cloud->reserve(feature_cloud->size());
        
        // Fast point-by-point transform with pre-extracted R and t
        for (size_t j = 0; j < feature_cloud->size(); ++j) {
            const auto& pt = (*feature_cloud)[j];
            util::PointType transformed_pt;
            Eigen::Vector3f p_in(pt.x, pt.y, pt.z);
            Eigen::Vector3f p_out = R * p_in + t;
            
            transformed_pt.x = p_out.x();
            transformed_pt.y = p_out.y();
            transformed_pt.z = p_out.z();
            transformed_cloud->push_back(transformed_pt);
        }
        
        // Batch accumulation (single operation, no element-wise resizes)
        *final_map += *transformed_cloud;
    }
    
    auto transform_end = std::chrono::high_resolution_clock::now();
    double transform_ms = std::chrono::duration<double, std::milli>(transform_end - transform_start).count();
    
    if (final_map->empty()) {
        LOG_ERROR("[Estimator] No points in accumulated map");
        return false;
    }

    // Step 4: Optional voxelization
    auto voxel_start = std::chrono::high_resolution_clock::now();
    if (voxel_size > 0.0f && voxel_size > m_config.voxel_size) {
        LOG_INFO("[Estimator] Applying additional voxelization ({}m > {}m)...",
                     voxel_size, m_config.voxel_size);
        util::VoxelGrid voxel_filter;
        voxel_filter.setLeafSize(voxel_size);
        
        util::PointCloudPtr downsampled = std::make_shared<util::PointCloud>();
        voxel_filter.setInputCloud(final_map);
        voxel_filter.filter(*downsampled);
        
        LOG_INFO("[Estimator] Downsampled: {} -> {} points", final_map->size(), downsampled->size());
        final_map = downsampled;
    }
    auto voxel_end = std::chrono::high_resolution_clock::now();
    double voxel_ms = std::chrono::duration<double, std::milli>(voxel_end - voxel_start).count();

    // Step 5: Write PCD file (with large buffer for faster I/O)
    auto write_start = std::chrono::high_resolution_clock::now();
    try {
        std::filesystem::path p(output_path);
        std::filesystem::create_directories(p.parent_path());

        std::ofstream ofs(output_path, std::ios::binary);
        if (!ofs.is_open()) {
            LOG_ERROR("[Estimator] Failed to open PCD file for writing: {}", output_path);
            return false;
        }

        // Increase buffer size for faster I/O (default is usually 4KB, we use 1MB)
        const size_t BUFFER_SIZE = 1024 * 1024;  // 1MB buffer
        std::vector<char> buffer(BUFFER_SIZE);
        ofs.rdbuf()->pubsetbuf(buffer.data(), BUFFER_SIZE);

        // PCD header
        ofs << "# .PCD v0.7 - Point Cloud Data file format\n";
        ofs << "VERSION 0.7\n";
        ofs << "FIELDS x y z\n";
        ofs << "SIZE 4 4 4\n";
        ofs << "TYPE F F F\n";
        ofs << "COUNT 1 1 1\n";
        ofs << "WIDTH " << final_map->size() << "\n";
        ofs << "HEIGHT 1\n";
        ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
        ofs << "POINTS " << final_map->size() << "\n";
        ofs << "DATA binary\n";

        // Pre-allocate binary data buffer (3 floats per point = 12 bytes)
        size_t binary_size = final_map->size() * 3 * sizeof(float);
        std::vector<float> binary_data;
        binary_data.reserve(final_map->size() * 3);
        
        // Batch write: copy all points to buffer first, then write once
        for (size_t i = 0; i < final_map->size(); ++i) {
            const auto& pt = (*final_map)[i];
            binary_data.push_back(pt.x);
            binary_data.push_back(pt.y);
            binary_data.push_back(pt.z);
        }
        
        // Single write operation (entire buffer at once)
        ofs.write(reinterpret_cast<const char*>(binary_data.data()), binary_size);
        ofs.close();
    } catch (const std::exception& e) {
        LOG_ERROR("[Estimator] Exception while saving PCD: {}", e.what());
        return false;
    }
    auto write_end = std::chrono::high_resolution_clock::now();
    double write_ms = std::chrono::duration<double, std::milli>(write_end - write_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    // Detailed timing breakdown
    LOG_INFO("[Estimator] ========== save_map_to_pcd() Timing ==========");
    LOG_INFO("[Estimator] Lock & Copy:      {:>8.2f} ms", lock_ms);
    LOG_INFO("[Estimator] Pre-allocate:     {:>8.2f} ms", alloc_ms);
    LOG_INFO("[Estimator] Transform & Acc:  {:>8.2f} ms ({}M points)", transform_ms, total_points / 1000000);
    LOG_INFO("[Estimator] Voxelization:     {:>8.2f} ms", voxel_ms);
    LOG_INFO("[Estimator] File Write:       {:>8.2f} ms ({} points)", write_ms, final_map->size());
    LOG_INFO("[Estimator] ────────────────────────────────");
    LOG_INFO("[Estimator] TOTAL:            {:>8.2f} ms", total_ms);
    LOG_INFO("[Estimator] =========================================");
    
    LOG_INFO("[Estimator] Saved final map to {} ({} points)", output_path, final_map->size());
    return true;
}

void Estimator::print_timing_statistics() const {
    if (m_timing_history.empty()) {
        return;
    }
    
    // Calculate statistics for last 100 frames (or all if less)
    size_t start_idx = m_timing_history.size() > 100 ? m_timing_history.size() - 100 : 0;
    size_t count = m_timing_history.size() - start_idx;
    
    double sum_preprocess = 0.0, sum_icp = 0.0, sum_map = 0.0, sum_total = 0.0;
    double max_preprocess = 0.0, max_icp = 0.0, max_map = 0.0, max_total = 0.0;
    double min_preprocess = 1e9, min_icp = 1e9, min_map = 1e9, min_total = 1e9;
    
    for (size_t i = start_idx; i < m_timing_history.size(); ++i) {
        const auto& t = m_timing_history[i];
        
        sum_preprocess += t.preprocessing_ms;
        sum_icp += t.icp_ms;
        sum_map += t.map_update_ms;
        sum_total += t.total_ms;
        
        max_preprocess = std::max(max_preprocess, t.preprocessing_ms);
        max_icp = std::max(max_icp, t.icp_ms);
        max_map = std::max(max_map, t.map_update_ms);
        max_total = std::max(max_total, t.total_ms);
        
        min_preprocess = std::min(min_preprocess, t.preprocessing_ms);
        min_icp = std::min(min_icp, t.icp_ms);
        min_map = std::min(min_map, t.map_update_ms);
        min_total = std::min(min_total, t.total_ms);
    }
    
    double avg_preprocess = sum_preprocess / count;
    double avg_icp = sum_icp / count;
    double avg_map = sum_map / count;
    double avg_total = sum_total / count;
    
    LOG_INFO("============================================================");
    LOG_INFO("[Timing Stats] Frame {} (last {} frames)", m_frame_count, count);
    LOG_INFO("------------------------------------------------------------");
    LOG_INFO("              |   Avg (ms)  |   Min (ms)  |   Max (ms)  ");
    LOG_INFO("------------------------------------------------------------");
    LOG_INFO(" Preprocess   | {:>10.2f}  | {:>10.2f}  | {:>10.2f}  ", avg_preprocess, min_preprocess, max_preprocess);
    LOG_INFO(" ICP          | {:>10.2f}  | {:>10.2f}  | {:>10.2f}  ", avg_icp, min_icp, max_icp);
    LOG_INFO(" Map Update   | {:>10.2f}  | {:>10.2f}  | {:>10.2f}  ", avg_map, min_map, max_map);
    LOG_INFO("------------------------------------------------------------");
    LOG_INFO(" Total        | {:>10.2f}  | {:>10.2f}  | {:>10.2f}  ", avg_total, min_total, max_total);
    LOG_INFO("============================================================");
}

} // namespace processing
} // namespace lidar_slam
