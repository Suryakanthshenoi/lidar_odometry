/**
 * @file      IterativeClosestPointOptimizer.h
 * @brief     Two-frame ICP optimizer using Gauss-Newton
 * @author    Seungwon Choi
 * @date      2025-10-04
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */


#pragma once

#include "../util/MathUtils.h"
#include "AdaptiveMEstimator.h"

#include <memory>
#include <vector>

// Forward declarations
namespace lidar_slam {
namespace database {
class LidarFrame;
}
namespace map {
class VoxelMap;
}
namespace util {
class PointCloud;
}
}

// Type alias for PointCloud pointer (defined in PointCloudUtils.h)
namespace lidar_slam {
namespace util {
using PointCloudPtr = std::shared_ptr<PointCloud>;
using PointCloudConstPtr = std::shared_ptr<const PointCloud>;
}
}

namespace lidar_slam {
namespace optimization {

// Import types from util namespace
using namespace lidar_slam::util;

// ============================================================================
// ICP Types (previously in ICPUtils.h)
// ============================================================================

/**
 * @brief Configuration for ICP algorithm
 */
struct ICPConfig {
    // Convergence criteria
    int max_iterations = 50;
    double translation_tolerance = 1e-6;  // meters
    double rotation_tolerance = 1e-6;     // radians
    
    // Correspondence parameters
    double max_correspondence_distance = 1.0;  // meters
    int min_correspondence_points = 10;
    
    // Outlier rejection
    double outlier_rejection_ratio = 0.9;  // Keep top 90% of correspondences
    bool use_robust_loss = true;
    double robust_loss_delta = 0.1;  // Huber loss delta
    
    // Performance
    bool use_kdtree = true;
    int max_kdtree_neighbors = 1;
    
    // Correspondence method
    bool use_surfel_correspondence = true;  // true: O(1) surfel lookup, false: KDTree + plane fitting
    
    // 2D constraint (x, y, yaw only)
    bool use_2d_constraint = false;  // Constrain ICP to 2D motion (x, y, yaw) to prevent z-axis drift
    bool use_2d_constraint_for_loop_closure = false;  // Apply 2D constraint to loop closure (usually false to handle height drift)
};

/**
 * @brief Point-to-plane correspondence for ICP
 */
struct ICPPointCorrespondence {
    Eigen::Vector3f source_point;
    Eigen::Vector3f target_point;
    Eigen::Vector3f plane_normal;    // Normal vector of target plane
    double distance = 0.0;
    double weight = 1.0;
    bool is_valid = false;
    
    ICPPointCorrespondence() = default;
    ICPPointCorrespondence(const Eigen::Vector3f& src, const Eigen::Vector3f& tgt, double dist)
        : source_point(src), target_point(tgt), distance(dist), weight(1.0), is_valid(true) {}
    ICPPointCorrespondence(const Eigen::Vector3f& src, const Eigen::Vector3f& tgt, const Eigen::Vector3f& normal, double dist)
        : source_point(src), target_point(tgt), plane_normal(normal), distance(dist), weight(1.0), is_valid(true) {}
};

using ICPCorrespondenceVector = std::vector<ICPPointCorrespondence>;

/**
 * @brief ICP statistics for monitoring convergence
 */
struct ICPStatistics {
    int iterations_used = 0;
    double final_cost = 0.0;
    double initial_cost = 0.0;
    size_t correspondences_count = 0;
    size_t inlier_count = 0;
    double match_ratio = 0.0;
    bool converged = false;
    
    void reset() {
        iterations_used = 0;
        final_cost = 0.0;
        initial_cost = 0.0;
        correspondences_count = 0;
        inlier_count = 0;
        match_ratio = 0.0;
        converged = false;
    }
};

// ============================================================================
// Correspondence Data Structures
// ============================================================================

/**
 * @brief Two-frame correspondence data
 */
struct DualFrameCorrespondences {
    std::vector<Eigen::Vector3d> points_last; ///< Points in last frame (local coordinates)
    std::vector<Eigen::Vector3d> points_curr; ///< Points in curr frame (local coordinates) 
    std::vector<Eigen::Vector3d> normals_last; ///< Normal vectors at last frame points (world coordinates)
    std::vector<double> residuals;         ///< Raw residuals for robust estimation
    
    void clear() {
        points_last.clear();
        points_curr.clear();
        normals_last.clear();
        residuals.clear();
    }
    
    size_t size() const {
        return points_last.size();
    }
};

/**
 * @brief Simple two-frame ICP optimizer
 * 
 * This class implements point-to-plane ICP between exactly two frames.
 * Much simpler than MultiFrameOptimizer for debugging and basic use cases.
 */
class IterativeClosestPointOptimizer {
public:
    /**
     * @brief Constructor
     * @param config ICP configuration
     */
    explicit IterativeClosestPointOptimizer(const ICPConfig& config = ICPConfig());
    
    /**
     * @brief Constructor with AdaptiveMEstimator
     * @param config ICP configuration
     * @param adaptive_estimator Pointer to AdaptiveMEstimator for robust optimization
     */
    IterativeClosestPointOptimizer(const ICPConfig& config, 
                         std::shared_ptr<optimization::AdaptiveMEstimator> adaptive_estimator);
    
    /**
     * @brief Destructor
     */
    ~IterativeClosestPointOptimizer() = default;
    
    /**
     * @brief Optimize relative pose between two frames using VoxelMap
     * 
     * @param voxel_map Pointer to VoxelMap with precomputed surfels
     * @param curr_frame Current frame (optimized)
     * @param initial_transform Initial relative transform guess (curr relative to keyframe)
     * @param optimized_transform Output optimized relative transform
     * @return True if optimization succeeded
     */
    bool optimize(map::VoxelMap* voxel_map,
                 std::shared_ptr<database::LidarFrame> curr_frame,
                 const SE3f& initial_transform,
                 SE3f& optimized_transform);

    /**
     * @brief Optimize relative pose between two keyframes for loop closure
     * @param curr_keframe Current keyframe (has fresh kdtree built)
     * @param matched_keyframe Matched keyframe from loop closure detection
     * @param optimized_relative_transform Output optimized relative transform (curr to matched)
     * @return True if optimization succeeded
     */
    
    bool optimize_loop(std::shared_ptr<database::LidarFrame> curr_keframe,
                 std::shared_ptr<database::LidarFrame> matched_keyframe,
                 SE3f& optimized_relative_transform,
                 float& inlier_ratio);
    
    /**
     * @brief Get optimization statistics
     */
    struct OptimizationStats {
        size_t num_correspondences = 0;
        size_t num_iterations = 0;
        double initial_cost = 0.0;
        double final_cost = 0.0;
        double optimization_time_ms = 0.0;
        bool converged = false;
    };
    
    /**
     * @brief Get last optimization statistics
     */
    const OptimizationStats& get_last_stats() const { return m_last_stats; }
    
    /**
     * @brief Update configuration
     */
    void update_config(const ICPConfig& config) { m_config = config; }
    
    /**
     * @brief Get current configuration
     */
    const ICPConfig& get_config() const { return m_config; }

private:
    /**
     * @brief Find correspondences using VoxelMap surfels (O(1) lookup)
     */
    size_t find_correspondences(map::VoxelMap* voxel_map,
                               std::shared_ptr<database::LidarFrame> curr_frame,
                               DualFrameCorrespondences& correspondences);
    
    /**
     * @brief Find correspondences using VoxelMap KDTree + plane fitting
     */
    size_t find_correspondences_kdtree(map::VoxelMap* voxel_map,
                                       std::shared_ptr<database::LidarFrame> curr_frame,
                                       DualFrameCorrespondences& correspondences);
    
    /**
     * @brief Find correspondences between two keyframes for loop closure
     */
    size_t find_correspondences_loop(std::shared_ptr<database::LidarFrame> last_keyframe,
                                     std::shared_ptr<database::LidarFrame> curr_keyframe,
                                     DualFrameCorrespondences &correspondences);

    /**
     * @brief Extract point cloud for correspondence finding
     */
    PointCloudConstPtr get_frame_cloud(std::shared_ptr<database::LidarFrame> frame);
    
    /**
     * @brief Check if points are collinear
     */
    bool is_collinear(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, 
                     const Eigen::Vector3d& p3, double threshold = 0.5);

    // Member variables
    ICPConfig m_config;
    OptimizationStats m_last_stats;
    std::shared_ptr<optimization::AdaptiveMEstimator> m_adaptive_estimator;
};

} // namespace processing
} // namespace lidar_slam