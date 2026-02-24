/**
 * @file      IterativeClosestPointOptimizer.cpp
 * @brief     Two-frame ICP optimizer implementation
 * @author    Seungwon Choi
 * @date      2025-10-04
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "IterativeClosestPointOptimizer.h"
#include "../util/MathUtils.h"
#include "../util/PointCloudUtils.h"
#include "../database/LidarFrame.h"
#include "../database/VoxelMap.h"
#include "util/LogUtils.h"
#include <chrono>
#include <numeric>
#include <algorithm>

namespace lidar_slam {
namespace optimization {

IterativeClosestPointOptimizer::IterativeClosestPointOptimizer(const ICPConfig& config)
    : m_config(config), m_adaptive_estimator(nullptr) {
    
    LOG_INFO("[IterativeClosestPointOptimizer] Initialized with max_iterations={}, max_correspondence_distance={}", 
                 m_config.max_iterations, m_config.max_correspondence_distance);
}

IterativeClosestPointOptimizer::IterativeClosestPointOptimizer(const ICPConfig& config, 
                                            std::shared_ptr<optimization::AdaptiveMEstimator> adaptive_estimator)
    : m_config(config), m_adaptive_estimator(adaptive_estimator) {
    
    LOG_INFO("[IterativeClosestPointOptimizer] Initialized with max_iterations={}, max_correspondence_distance={} and AdaptiveMEstimator", 
                 m_config.max_iterations, m_config.max_correspondence_distance);
}

bool IterativeClosestPointOptimizer::optimize_loop(std::shared_ptr<database::LidarFrame> curr_keyframe,
                                          std::shared_ptr<database::LidarFrame> matched_keyframe,
                                          SE3f &optimized_relative_transform,
                                          float& inlier_ratio)
{
    // Deep copy keyframes to avoid modifying original poses
    auto curr_keyframe_copy = std::make_shared<database::LidarFrame>(*curr_keyframe);
    auto matched_keyframe_copy = std::make_shared<database::LidarFrame>(*matched_keyframe);

    SE3f optimized_curr_pose = curr_keyframe_copy->get_pose();
    SE3f matched_pose = matched_keyframe_copy->get_pose();

    // Reset adaptive estimator
    if (m_adaptive_estimator) {
        m_adaptive_estimator->reset();
    }

    bool success = false;

    // Build kd tree of local map of matched keyframe
    auto local_feature_matched = get_frame_cloud(matched_keyframe_copy);
    PointCloudPtr transformed_matched_cloud = std::make_shared<PointCloud>();
    util::transform_point_cloud(local_feature_matched, transformed_matched_cloud, matched_keyframe_copy->get_pose().Matrix());
    matched_keyframe_copy->set_local_map(transformed_matched_cloud);
    matched_keyframe_copy->build_local_map_kdtree();

    // Residual normalization scale (calculated on first iteration)
    double residual_normalization_scale = 1.0;

    // Get loss type
    std::string loss_type = "huber";
    if (m_adaptive_estimator) {
        loss_type = m_adaptive_estimator->get_config().loss_type;
    }

    for (int icp_iter = 0; icp_iter < 100; ++icp_iter) {
        curr_keyframe_copy->set_pose(optimized_curr_pose);

        // Find correspondences
        DualFrameCorrespondences correspondences;
        find_correspondences_loop(matched_keyframe_copy, curr_keyframe_copy, correspondences);

        if (correspondences.size() < static_cast<size_t>(m_config.min_correspondence_points)) {
            LOG_WARN("[Loop ICP] Insufficient correspondences: {}", correspondences.size());
            break;
        }

        // Calculate residual normalization scale on first iteration
        if (icp_iter == 0 && !correspondences.residuals.empty()) {
            std::vector<double> residuals = correspondences.residuals;
            std::sort(residuals.begin(), residuals.end());
            double mean = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
            double variance = 0.0;
            for (double val : residuals) {
                variance += (val - mean) * (val - mean);
            }
            variance /= residuals.size();
            double std_dev = std::sqrt(variance);
            residual_normalization_scale = std_dev / 6.0;
        }

        // Calculate adaptive delta using AdaptiveMEstimator (PKO)
        double adaptive_delta = m_config.robust_loss_delta;
        if (m_adaptive_estimator && m_adaptive_estimator->get_config().use_adaptive_m_estimator) {
            std::vector<double> normalized_residuals;
            normalized_residuals.reserve(correspondences.residuals.size());
            for (double residual : correspondences.residuals) {
                double normalized_residual = residual / std::max(residual_normalization_scale, 1e-6);
                normalized_residuals.push_back(normalized_residual);
            }
            if (!normalized_residuals.empty()) {
                double scale_factor = m_adaptive_estimator->calculate_scale_factor(normalized_residuals);
                adaptive_delta = scale_factor;
            }
        }

        // ============================================
        // Gauss-Newton Point-to-Plane ICP (Dual Frame)
        // ============================================
        // For dual frame: we optimize curr_pose while matched_pose is fixed
        // Residual: r = n^T * (R_curr * p_curr + t_curr - (R_matched * p_matched + t_matched))
        //           = n^T * (p_curr_world - p_matched_world)
        // ============================================

        Eigen::Matrix3f R_curr = optimized_curr_pose.RotationMatrix();
        Eigen::Vector3f t_curr = optimized_curr_pose.Translation();
        Eigen::Matrix3f R_matched = matched_pose.RotationMatrix();
        Eigen::Vector3f t_matched = matched_pose.Translation();

        // Build normal equation: H * delta = -g
        Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> g = Eigen::Matrix<float, 6, 1>::Zero();

        for (size_t i = 0; i < correspondences.size(); ++i) {
            // points_last is in local coords of matched frame
            Eigen::Vector3f p_matched_local = correspondences.points_last[i].cast<float>();
            // points_curr is in local coords of current frame  
            Eigen::Vector3f p_curr_local = correspondences.points_curr[i].cast<float>();
            // normal is in world coords
            Eigen::Vector3f n = correspondences.normals_last[i].cast<float>();

            // Transform to world
            Eigen::Vector3f p_matched_world = R_matched * p_matched_local + t_matched;
            Eigen::Vector3f p_curr_world = R_curr * p_curr_local + t_curr;

            // Point-to-plane residual
            float residual = n.dot(p_curr_world - p_matched_world);

            // Normalize residual for robust weight calculation
            float normalized_residual = static_cast<float>(correspondences.residuals[i] / std::max(residual_normalization_scale, 1e-6));

            // Jacobian w.r.t. curr pose (right perturbation)
            // J = [n^T * R_curr, -n^T * R_curr * [p_curr_local]_x]
            Eigen::Matrix<float, 1, 6> J;
            J.block<1, 3>(0, 0) = n.transpose() * R_curr;

            Eigen::Matrix3f p_skew;
            p_skew <<     0, -p_curr_local.z(),  p_curr_local.y(),
                      p_curr_local.z(),     0, -p_curr_local.x(),
                     -p_curr_local.y(),  p_curr_local.x(),     0;

            J.block<1, 3>(0, 3) = -n.transpose() * R_curr * p_skew;

            // Robust weighting
            float weight = 1.0f;
            if (m_config.use_robust_loss) {
                float abs_normalized = std::abs(normalized_residual);
                float delta = static_cast<float>(adaptive_delta);

                if (loss_type == "cauchy") {
                    float ratio = abs_normalized / delta;
                    weight = 1.0f / (1.0f + ratio * ratio);
                } else {
                    if (abs_normalized > delta) {
                        weight = delta / abs_normalized;
                    }
                }
            }

            // Accumulate normal equation
            H += weight * J.transpose() * J;
            g += weight * residual * J.transpose();
        }

        // Solve H * delta = -g
        Eigen::Matrix<float, 6, 1> delta = H.ldlt().solve(-g);

        Eigen::Vector3f dt = delta.head<3>();
        Eigen::Vector3f dw = delta.tail<3>();

        // Apply 2D constraint if enabled for loop closure (only x, y, yaw)
        if (m_config.use_2d_constraint_for_loop_closure) {
            dt.z() = 0.0f;      // No z translation
            dw.x() = 0.0f;      // No roll
            dw.y() = 0.0f;      // No pitch
            // dw.z() remains (yaw only)
        }

        // Create SE3 from delta
        SE3f delta_transform;
        if (dw.norm() < 1e-10f) {
            delta_transform = SE3f(SO3f::Identity(), dt);
        } else {
            delta_transform = SE3f(SO3f::Exp(dw), dt);
        }

        // Update transform (right multiplication)
        optimized_curr_pose = optimized_curr_pose * delta_transform;

        // Check convergence
        float translation_delta = dt.norm();
        float rotation_delta = dw.norm();

        if (translation_delta < m_config.translation_tolerance && rotation_delta < m_config.rotation_tolerance) {
            LOG_DEBUG("[ICP] Loop closure converged at iteration {}", icp_iter + 1);
            optimized_relative_transform = curr_keyframe->get_pose().Inverse() * optimized_curr_pose;
            success = true;
            break;
        }
    }

    // Calculate inlier ratio for validation
    if (success) {
        curr_keyframe_copy->set_pose(optimized_curr_pose);

        int inlier_count = 0;
        int total_count = 0;
        auto curr_feature_cloud = get_frame_cloud(curr_keyframe_copy);
        for (size_t i = 0; i < curr_feature_cloud->size(); ++i) {
            const auto& point_local = curr_feature_cloud->at(i);

            Eigen::Vector3f point_local_eigen(point_local.x, point_local.y, point_local.z);
            Eigen::Vector3f point_world_eigen = curr_keyframe_copy->get_pose().Matrix().block<3, 1>(0, 3) +
                                                curr_keyframe_copy->get_pose().Matrix().block<3, 3>(0, 0) * point_local_eigen;

            util::Point3D point_world;
            point_world.x = point_world_eigen.x();
            point_world.y = point_world_eigen.y();
            point_world.z = point_world_eigen.z();

            std::vector<int> indices(1);
            std::vector<float> sqdist(1);
            matched_keyframe_copy->get_local_map_kdtree()->nearestKSearch(point_world, 1, indices, sqdist);

            if (std::sqrt(sqdist[0]) < 1.0f) {
                inlier_count++;
            }
            total_count++;
        }

        inlier_ratio = static_cast<float>(inlier_count) / static_cast<float>(total_count);
        LOG_DEBUG("[ICP] Loop closure inlier ratio: {:.3f}", inlier_ratio);

        if (inlier_ratio < 0.5f) {
            success = false;
        }
    }

    return success;
}



bool IterativeClosestPointOptimizer::optimize(map::VoxelMap* voxel_map,
                                     std::shared_ptr<database::LidarFrame> curr_frame,
                                     const SE3f& initial_transform,
                                     SE3f& optimized_transform) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Reset stats
    m_last_stats = OptimizationStats();
    
    // Initialize current transform estimate
    SE3f current_transform = initial_transform;
    optimized_transform = current_transform;
    
    // Reset adaptive estimator
    if (m_adaptive_estimator) {
        m_adaptive_estimator->reset();
    }
    
    double total_initial_cost = 0.0;
    double total_final_cost = 0.0;
    int total_iterations = 0;
    
    // Residual normalization scale (calculated on first iteration)
    double residual_normalization_scale = 1.0;

    // ICP iteration loop with Gauss-Newton optimization
    for (int icp_iter = 0; icp_iter < m_config.max_iterations; ++icp_iter) {
        
        // Update curr frame pose for correspondence finding
        curr_frame->set_pose(current_transform);
        
        // Find correspondences based on config
        DualFrameCorrespondences correspondences;
        size_t num_correspondences = 0;
        
        if (m_config.use_surfel_correspondence) {
            // Use VoxelMap surfels (O(1) lookup)
            num_correspondences = find_correspondences(voxel_map, curr_frame, correspondences);
        } else {
            // Use KDTree + plane fitting
            num_correspondences = find_correspondences_kdtree(voxel_map, curr_frame, correspondences);
        }
        
        if (num_correspondences < static_cast<size_t>(m_config.min_correspondence_points)) {
            LOG_WARN("[ICP] Insufficient correspondences: {} < {} at iteration {}", 
                        num_correspondences, m_config.min_correspondence_points, icp_iter + 1);
            return false;
        }
        
        // Calculate residual normalization scale on first iteration
        if (icp_iter == 0 && !correspondences.residuals.empty()) {
            std::vector<double> residuals = correspondences.residuals;
            std::sort(residuals.begin(), residuals.end());
            double mean = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
            double variance = 0.0;
            for (double val : residuals) {
                variance += (val - mean) * (val - mean);
            }
            variance /= residuals.size();
            double std_dev = std::sqrt(variance);
            residual_normalization_scale = std_dev / 6.0;
        }
        
        // Calculate adaptive delta using AdaptiveMEstimator (PKO)
        double adaptive_delta = m_config.robust_loss_delta;  // Default delta
        if (m_adaptive_estimator && m_adaptive_estimator->get_config().use_adaptive_m_estimator) {
            std::vector<double> normalized_residuals;
            normalized_residuals.reserve(correspondences.residuals.size());
            for (double residual : correspondences.residuals) {
                double normalized_residual = residual / std::max(residual_normalization_scale, 1e-6);
                normalized_residuals.push_back(normalized_residual);
            }
            
            if (!normalized_residuals.empty()) {
                double scale_factor = m_adaptive_estimator->calculate_scale_factor(normalized_residuals);
                adaptive_delta = scale_factor;
            }
        }
        
        // ============================================
        // Gauss-Newton Point-to-Plane ICP
        // ============================================
        // Minimize: sum_i w_i * || n_i^T * (R*p_i + t - q_i) ||^2
        // 
        // Residual: r_i = n_i^T * (R*p_i + t - q_i)
        // Jacobian: J_i = [n_i^T, -n_i^T * R * [p_i]_x]  (1x6)
        //
        // Normal equation: (J^T * W * J) * delta = -J^T * W * r
        // ============================================
        
        Eigen::Matrix3f R = current_transform.RotationMatrix();
        Eigen::Vector3f t = current_transform.Translation();
        
        // Build normal equation: H * delta = -g
        Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> g = Eigen::Matrix<float, 6, 1>::Zero();
        float total_cost = 0.0f;
        
        // Get loss type from adaptive estimator
        std::string loss_type = "huber";
        if (m_adaptive_estimator) {
            loss_type = m_adaptive_estimator->get_config().loss_type;
        }
        
        for (size_t i = 0; i < correspondences.size(); ++i) {
            // curr point in local coords (to be transformed)
            Eigen::Vector3f p = correspondences.points_curr[i].cast<float>();
            // map point in world coords (target)
            Eigen::Vector3f q = correspondences.points_last[i].cast<float>();
            // normal at target point (world coords)
            Eigen::Vector3f n = correspondences.normals_last[i].cast<float>();
            
            // Transform source point to world
            Eigen::Vector3f p_world = R * p + t;
            
            // Point-to-plane residual: n^T * (p_world - q)
            float residual = n.dot(p_world - q);
            
            // Normalize residual for robust weight calculation
            float normalized_residual = static_cast<float>(correspondences.residuals[i] / std::max(residual_normalization_scale, 1e-6));
            
            // Jacobian: J = [n^T * R, -n^T * R * [p]_x] (right perturbation)
            Eigen::Matrix<float, 1, 6> J;
            J.block<1, 3>(0, 0) = n.transpose() * R;  // d/dt (right perturbation)
            
            // Skew-symmetric of p
            Eigen::Matrix3f p_skew;
            p_skew <<     0, -p.z(),  p.y(),
                      p.z(),     0, -p.x(),
                     -p.y(),  p.x(),     0;
            
            J.block<1, 3>(0, 3) = -n.transpose() * R * p_skew;  // d/dw
            
            // Robust weighting based on normalized residual
            float weight = 1.0f;
            if (m_config.use_robust_loss) {
                float abs_normalized = std::abs(normalized_residual);
                float delta = static_cast<float>(adaptive_delta);
                
                if (loss_type == "cauchy") {
                    // Cauchy weight: w = 1 / (1 + (r/delta)^2)
                    float ratio = abs_normalized / delta;
                    weight = 1.0f / (1.0f + ratio * ratio);
                } else {
                    // Huber weight
                    if (abs_normalized > delta) {
                        weight = delta / abs_normalized;
                    }
                }
            }
            
            // Accumulate normal equation (using original residual, not normalized)
            H += weight * J.transpose() * J;
            g += weight * residual * J.transpose();
            total_cost += weight * residual * residual;
        }
        
        if (icp_iter == 0) {
            total_initial_cost = total_cost;
        }
        total_final_cost = total_cost;
        
        // Solve H * delta = -g using Cholesky decomposition
        Eigen::Matrix<float, 6, 1> delta = H.ldlt().solve(-g);
        
        // Apply update: T_new = T_old * exp(delta)
        // delta = [dt, dw] where dt is translation, dw is rotation (axis-angle)
        Eigen::Vector3f dt = delta.head<3>();
        Eigen::Vector3f dw = delta.tail<3>();
        
        // Apply 2D constraint if enabled (only x, y, yaw)
        if (m_config.use_2d_constraint) {
            dt.z() = 0.0f;      // No z translation
            dw.x() = 0.0f;      // No roll
            dw.y() = 0.0f;      // No pitch
            // dw.z() remains (yaw only)
        }
        
        // Create SE3 from delta
        SE3f delta_transform;
        if (dw.norm() < 1e-10f) {
            delta_transform = SE3f(SO3f::Identity(), dt);
        } else {
            delta_transform = SE3f(SO3f::Exp(dw), dt);
        }
        
        // Update transform (right multiplication)
        current_transform = current_transform * delta_transform;
        
        // Check convergence
        float translation_delta = dt.norm();
        float rotation_delta = dw.norm();
        
        total_iterations++;
        m_last_stats.num_correspondences = num_correspondences;
        
        bool converged = (translation_delta < m_config.translation_tolerance) && 
                        (rotation_delta < m_config.rotation_tolerance);
        
        if (converged) {
            break;
        }
    }
    
    // Set final results
    optimized_transform = current_transform;
    m_last_stats.num_iterations = total_iterations;
    m_last_stats.initial_cost = total_initial_cost;
    m_last_stats.final_cost = total_final_cost;
    m_last_stats.converged = true;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    m_last_stats.optimization_time_ms = duration.count();
    
    return true;
}

size_t IterativeClosestPointOptimizer::find_correspondences_loop(std::shared_ptr<database::LidarFrame> last_keyframe,
                                                        std::shared_ptr<database::LidarFrame> curr_keyframe,
                                                        DualFrameCorrespondences &correspondences)
{
    correspondences.clear();
    PointCloudPtr local_map_last = last_keyframe->get_local_map(); // Local map of last keyframe (world coordinates)

    auto local_feature_curr = get_frame_cloud(curr_keyframe);     // Current frame feature cloud (local coordinates)

    if(!local_feature_curr || !local_map_last || local_feature_curr->empty() || local_map_last->empty()) {
        LOG_WARN("[IterativeClosestPointOptimizer] Empty point clouds - curr_map: {}, matched_map: {}",
                     local_feature_curr ? local_feature_curr->size() : 0,
                     local_map_last ? local_map_last->size() : 0);
        return 0;
    }

    auto kdtree_last_ptr = last_keyframe->get_local_map_kdtree();

    if(!kdtree_last_ptr) {
        LOG_ERROR("[IterativeClosestPointOptimizer] Last keyframe has no KdTree - this should not happen!");
        return 0;
    }



    const int K = 5;  // Number of neighbors for plane fitting

    // Find correspondences: query CURR points, find neighbors in LAST cloud

    Eigen::Matrix4f T_wl_last = last_keyframe->get_pose().Matrix(); // Last keyframe pose in world coordinates
    Eigen::Matrix4f T_lw_last = T_wl_last.inverse(); // Inverse transform
    Eigen::Matrix4f T_wl_curr = curr_keyframe->get_pose().Matrix(); // Current keyframe pose in world coordinates
    Eigen::Matrix4f T_lw_curr = T_wl_curr.inverse(); // Inverse transform

    for (size_t idx = 0; idx < local_feature_curr->size(); ++idx)
    {
        const auto& curr_point_local = local_feature_curr->at(idx);

        // Transform current point to world coordinates
        Eigen::Vector4f curr_point_world_h = T_wl_curr * Eigen::Vector4f(curr_point_local.x, curr_point_local.y, curr_point_local.z, 1.0f);
        Eigen::Vector3f curr_point_world(curr_point_world_h.x(), curr_point_world_h.y(), curr_point_world_h.z());

        // Find K nearest neighbors in LAST cloud
        std::vector<int> neighbor_indices(K);
        std::vector<float> neighbor_distances(K);
        util::Point3D query_point;
        query_point.x = curr_point_world.x();
        query_point.y = curr_point_world.y();
        query_point.z = curr_point_world.z();
        int found_neighbors = kdtree_last_ptr->nearestKSearch(query_point, K, neighbor_indices, neighbor_distances);
        if (found_neighbors < 5) {
            continue;
        }
        // Select points for plane fitting from LAST cloud
        std::vector<Eigen::Vector3d> selected_points_world;
        std::vector<Eigen::Vector3d> selected_points_local;
        bool non_collinear_found = false;

        for (int k = 0; k < found_neighbors && selected_points_world.size() < 5; ++k) {
            int neighbor_idx = neighbor_indices[k];

            // Local map points are already in world coordinates
            Eigen::Vector3d pt_world(local_map_last->at(neighbor_idx).x,
                                   local_map_last->at(neighbor_idx).y,
                                   local_map_last->at(neighbor_idx).z);

            // For local coordinates, we use the same world coordinates since local map is in world frame
            Eigen::Vector3d pt_local = T_lw_last.block<3,3>(0,0).cast<double>() * pt_world.cast<double>() + T_lw_last.block<3,1>(0,3).cast<double>();

            selected_points_world.push_back(pt_world);
            selected_points_local.push_back(pt_local);
        }

        // Check for non-collinear points
        if (selected_points_world.size() >= 3) {
            if (is_collinear(selected_points_world[0], selected_points_world[1], selected_points_world[2], 0.5)) {
                continue;
            }
            non_collinear_found = true;
        }
        if (!non_collinear_found) {
            continue;
        }
        
        // Fit plane to selected points using SVD
        if (selected_points_world.size() < 3) {
            continue;
        }
        
        // Compute centroid
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const auto& pt : selected_points_world) {
            centroid += pt;
        }
        centroid /= selected_points_world.size();
        
        // Build matrix for SVD
        Eigen::MatrixXd A(selected_points_world.size(), 3);
        for (size_t i = 0; i < selected_points_world.size(); ++i) {
            A.row(i) = (selected_points_world[i] - centroid).transpose();
        }
        
        // Compute SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::Vector3d plane_normal = svd.matrixV().col(2);  // Last column is normal
        double plane_d = -plane_normal.dot(centroid);
        // Compute point-to-plane distance (residual)
        Eigen::Vector3d curr_point_world_d(curr_point_world.x(), curr_point_world.y(), curr_point_world.z());
        double distance = plane_normal.dot(curr_point_world_d) + plane_d;
        distance = std::abs(distance);  
        // No distance thresholding for loop closure
        // Store correspondence
        correspondences.points_last.push_back(selected_points_local[0]); // Use one of the local points
        correspondences.points_curr.push_back(Eigen::Vector3d(curr_point_local.x, curr_point_local.y, curr_point_local.z));
        correspondences.normals_last.push_back(plane_normal);
        correspondences.residuals.push_back(distance);  

    }

    return correspondences.size();
}

size_t IterativeClosestPointOptimizer::find_correspondences(map::VoxelMap* voxel_map,
                                                  std::shared_ptr<database::LidarFrame> curr_frame,
                                                  DualFrameCorrespondences& correspondences) {
    
    correspondences.clear();
    
    if (!voxel_map || voxel_map->empty()) {
        LOG_WARN("[IterativeClosestPointOptimizer] VoxelMap is empty!");
        return 0;
    }
    
    auto curr_cloud = get_frame_cloud(curr_frame);     // Current frame feature cloud (local coordinates)
    
    if (!curr_cloud || curr_cloud->empty()) {
        LOG_WARN("[IterativeClosestPointOptimizer] Empty curr_cloud");
        return 0;
    }
    
    // Transform current frame cloud to world coordinates using current pose estimate
    auto curr_pose = curr_frame->get_pose();   // Current estimate
    
    PointCloudPtr curr_world(new PointCloud());
    util::transform_point_cloud(curr_cloud, curr_world, curr_pose.Matrix());
    
    // Find correspondences: query CURR points, get precomputed surfels from VoxelMap
    for (size_t idx = 0; idx < curr_world->size(); ++idx) {
        
        const auto& curr_point_world = curr_world->at(idx);
        Eigen::Vector3f query_pt(curr_point_world.x, curr_point_world.y, curr_point_world.z);
        
        // Use precomputed surfel from VoxelMap (O(1) lookup)
        Eigen::Vector3f normal_f, plane_point_f;
        if (!voxel_map->GetSurfelAtPoint(query_pt, normal_f, plane_point_f)) {
            continue;  // No valid surfel in this L1 voxel
        }
        
        Eigen::Vector3d normal = normal_f.cast<double>();
        Eigen::Vector3d plane_point = plane_point_f.cast<double>();
        
        // Calculate residual
        Eigen::Vector3d curr_point_world_d(curr_point_world.x, curr_point_world.y, curr_point_world.z);
        double residual = std::abs(normal.dot(curr_point_world_d - plane_point));
        
        if (residual > m_config.max_correspondence_distance) {
            continue;
        }
        
        // Store correspondence
        // plane_point is in world coordinates (will be treated as "local" with identity source pose)
        Eigen::Vector3d curr_point_local(curr_cloud->at(idx).x, curr_cloud->at(idx).y, curr_cloud->at(idx).z);
        
        correspondences.points_last.push_back(plane_point);  // World coords (identity source pose)
        correspondences.points_curr.push_back(curr_point_local);
        correspondences.normals_last.push_back(normal);  // Normal from map (world coordinates)
        correspondences.residuals.push_back(residual);
    }
    
    return correspondences.size();
}

size_t IterativeClosestPointOptimizer::find_correspondences_kdtree(map::VoxelMap* voxel_map,
                                                  std::shared_ptr<database::LidarFrame> curr_frame,
                                                  DualFrameCorrespondences& correspondences) {
    
    correspondences.clear();
    
    if (!voxel_map || voxel_map->empty()) {
        LOG_WARN("[IterativeClosestPointOptimizer] VoxelMap is empty!");
        return 0;
    }
    
    // Check if KDTree is built
    auto kdtree = voxel_map->GetKdTree();
    if (!kdtree) {
        LOG_WARN("[IterativeClosestPointOptimizer] VoxelMap KDTree not built! Call RebuildKdTree() first.");
        return 0;
    }
    
    auto curr_cloud = get_frame_cloud(curr_frame);     // Current frame feature cloud (local coordinates)
    
    if (!curr_cloud || curr_cloud->empty()) {
        LOG_WARN("[IterativeClosestPointOptimizer] Empty curr_cloud");
        return 0;
    }
    
    // Transform current frame cloud to world coordinates using current pose estimate
    auto curr_pose = curr_frame->get_pose();   // Current estimate
    
    PointCloudPtr curr_world(new PointCloud());
    util::transform_point_cloud(curr_cloud, curr_world, curr_pose.Matrix());
    
    // Get map point cloud for neighbor access
    auto map_cloud = voxel_map->GetPointCloud();
    if (!map_cloud || map_cloud->empty()) {
        LOG_WARN("[IterativeClosestPointOptimizer] Empty map cloud");
        return 0;
    }
    
    const int K = 5;  // Number of neighbors for plane fitting
    
    // Find correspondences: query CURR points, find neighbors in map using KDTree
    for (size_t idx = 0; idx < curr_world->size(); ++idx) {
        
        const auto& curr_point_world = curr_world->at(idx);
        util::Point3D query_point;
        query_point.x = curr_point_world.x;
        query_point.y = curr_point_world.y;
        query_point.z = curr_point_world.z;
        
        // Find K nearest neighbors
        std::vector<int> neighbor_indices(K);
        std::vector<float> neighbor_distances(K);
        int found_neighbors = kdtree->nearestKSearch(query_point, K, neighbor_indices, neighbor_distances);
        
        if (found_neighbors < 5) {
            continue;
        }
        
        // Select points for plane fitting from map cloud
        std::vector<Eigen::Vector3d> selected_points;
        selected_points.reserve(K);
        
        for (int k = 0; k < found_neighbors && selected_points.size() < 5; ++k) {
            int neighbor_idx = neighbor_indices[k];
            if (neighbor_idx < 0 || static_cast<size_t>(neighbor_idx) >= map_cloud->size()) {
                continue;
            }
            
            Eigen::Vector3d pt(map_cloud->at(neighbor_idx).x,
                               map_cloud->at(neighbor_idx).y,
                               map_cloud->at(neighbor_idx).z);
            selected_points.push_back(pt);
        }
        
        if (selected_points.size() < 3) {
            continue;
        }
        
        // Check for non-collinear points
        if (is_collinear(selected_points[0], selected_points[1], selected_points[2], 0.5)) {
            continue;
        }
        
        // Fit plane to selected points using SVD
        // Compute centroid
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const auto& pt : selected_points) {
            centroid += pt;
        }
        centroid /= selected_points.size();
        
        // Build matrix for SVD
        Eigen::MatrixXd A(selected_points.size(), 3);
        for (size_t i = 0; i < selected_points.size(); ++i) {
            A.row(i) = (selected_points[i] - centroid).transpose();
        }
        
        // Compute SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::Vector3d plane_normal = svd.matrixV().col(2);  // Last column is normal
        double plane_d = -plane_normal.dot(centroid);
        
        // Compute point-to-plane distance (residual)
        Eigen::Vector3d curr_point_world_d(curr_point_world.x, curr_point_world.y, curr_point_world.z);
        double distance = std::abs(plane_normal.dot(curr_point_world_d) + plane_d);
        
        if (distance > m_config.max_correspondence_distance) {
            continue;
        }
        
        // Store correspondence
        Eigen::Vector3d curr_point_local(curr_cloud->at(idx).x, curr_cloud->at(idx).y, curr_cloud->at(idx).z);
        
        correspondences.points_last.push_back(centroid);  // Use centroid as target point
        correspondences.points_curr.push_back(curr_point_local);
        correspondences.normals_last.push_back(plane_normal);
        correspondences.residuals.push_back(distance);
    }
    
    return correspondences.size();
}

PointCloudConstPtr IterativeClosestPointOptimizer::get_frame_cloud(std::shared_ptr<database::LidarFrame> frame) {
    
    // Try to get feature cloud first, fall back to processed cloud
    auto feature_cloud = frame->get_feature_cloud();
    if (feature_cloud && !feature_cloud->empty()) {
        return feature_cloud;
    }
    
    auto processed_cloud = frame->get_processed_cloud();
    if (processed_cloud && !processed_cloud->empty()) {
        return processed_cloud;
    }
    
    return nullptr;
}

bool IterativeClosestPointOptimizer::is_collinear(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, 
                                         const Eigen::Vector3d& p3, double threshold) {
    Eigen::Vector3d v1 = (p2 - p1).normalized();
    Eigen::Vector3d v2 = (p3 - p1).normalized();
    
    double cross_norm = v1.cross(v2).norm();
    return cross_norm < threshold;
}

} // namespace processing
} // namespace lidar_slam