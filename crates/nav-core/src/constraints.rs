#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments, unreachable_patterns)]
//! TebConfig: all constraint/configuration structs for the TEB planner.
//!
//! Ported from `teb_config.h` and `teb_config.cpp` with exact default values.

/// The complete TEB configuration.
///
/// Ported 1:1 from `TebConfig` in the C++ implementation.
/// Default values match `teb_config.cpp` exactly.
#[derive(Debug, Clone)]
pub struct TebConfig {
    /// Trajectory-related parameters.
    pub trajectory: TrajectoryConfig,
    /// Robot kinematics parameters.
    pub robot: RobotConfig,
    /// Goal tolerance parameters.
    pub goal_tolerance: GoalToleranceConfig,
    /// Obstacle avoidance parameters.
    pub obstacles: ObstacleConfig,
    /// Optimization weights and settings.
    pub optim: OptimizationConfig,
    /// Homotopy class planning parameters.
    pub hcp: HcpConfig,
    /// Recovery behavior parameters.
    pub recovery: RecoveryConfig,
    /// Minimum turning radius [m].
    pub min_turning_radius: f64,
    /// Wheelbase distance for carlike robots [m].
    pub wheelbase: f64,
    /// Use steering angle instead of rotational velocity.
    pub cmd_angle_instead_rotvel: bool,
}

impl TebConfig {
    /// Create a new TebConfig with default values from the C++ implementation.
    pub fn default_() -> Self {
        Self {
            trajectory: TrajectoryConfig::default_(),
            robot: RobotConfig::default_(),
            goal_tolerance: GoalToleranceConfig::default_(),
            obstacles: ObstacleConfig::default_(),
            optim: OptimizationConfig::default_(),
            hcp: HcpConfig::default_(),
            recovery: RecoveryConfig::default_(),
            min_turning_radius: 0.0,
            wheelbase: 1.0,
            cmd_angle_instead_rotvel: false,
        }
    }

    /// Create a config for a differential-drive robot.
    pub fn for_differential_drive() -> Self {
        Self::default_()
    }

    /// Create a config for an ackermann/carlike robot.
    pub fn for_ackermann_drive() -> Self {
        let mut cfg = Self::default_();
        cfg.min_turning_radius = 0.5;
        cfg
    }
}

impl Default for TebConfig {
    fn default() -> Self {
        Self::default_()
    }
}

// ── Trajectory Configuration ───────────────────────────────────────

/// Trajectory-related parameters.
/// Ported from `TebConfig::Trajectory`.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryConfig {
    /// Enable automatic resizing w.r.t. temporal resolution.
    pub teb_autosize: bool,
    /// Desired temporal resolution [s].
    pub dt_ref: f64,
    /// Hysteresis for auto-resize (10% of dt_ref).
    pub dt_hysteresis: f64,
    /// Minimum number of samples.
    pub min_samples: usize,
    /// Maximum number of samples.
    pub max_samples: usize,
    /// Overwrite orientation of local subgoals from global planner.
    pub global_plan_overwrite_orientation: bool,
    /// Allow initialization with backwards motion.
    pub allow_init_with_backwards_motion: bool,
    /// Min separation between via-points [-1 = disabled].
    pub global_plan_viapoint_sep: f64,
    /// Enforce order of via-points.
    pub via_points_ordered: bool,
    /// Maximum length of global plan subset [m].
    pub max_global_plan_lookahead_dist: f64,
    /// Distance for pruning global plan.
    pub global_plan_prune_distance: f64,
    /// Use exact arc length in computations.
    pub exact_arc_length: bool,
    /// Reinit threshold for goal position [m].
    pub force_reinit_new_goal_dist: f64,
    /// Reinit threshold for goal orientation [rad].
    pub force_reinit_new_goal_angular: f64,
    /// Poses for feasibility check [-1 = all].
    pub feasibility_check_no_poses: i32,
    /// Max distance for feasibility check [m].
    pub feasibility_check_lookahead_distance: f64,
    /// Publish feedback.
    pub publish_feedback: bool,
    /// Min angular resolution for collision check [rad].
    pub min_resolution_collision_check_angular: f64,
    /// Index of pose for velocity command extraction.
    pub control_look_ahead_poses: usize,
    /// Prevent look-ahead near goal.
    pub prevent_look_ahead_poses_near_goal: usize,
}

impl TrajectoryConfig {
    fn default_() -> Self {
        Self {
            teb_autosize: true,
            dt_ref: 0.3,
            dt_hysteresis: 0.1,
            min_samples: 3,
            max_samples: 500,
            global_plan_overwrite_orientation: true,
            allow_init_with_backwards_motion: false,
            global_plan_viapoint_sep: -1.0,
            via_points_ordered: false,
            max_global_plan_lookahead_dist: 1.0,
            global_plan_prune_distance: 1.0,
            exact_arc_length: false,
            force_reinit_new_goal_dist: 1.0,
            force_reinit_new_goal_angular: std::f64::consts::PI / 2.0,
            feasibility_check_no_poses: 5,
            feasibility_check_lookahead_distance: -1.0,
            publish_feedback: false,
            min_resolution_collision_check_angular: std::f64::consts::PI,
            control_look_ahead_poses: 1,
            prevent_look_ahead_poses_near_goal: 0,
        }
    }
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self::default_()
    }
}

// ── Robot Configuration ────────────────────────────────────────────

/// Robot kinematic configuration.
#[derive(Debug, Clone, Copy)]
pub struct RobotConfig {
    /// Maximum translational velocity [m/s].
    pub max_vel_x: f64,
    /// Maximum backwards velocity [m/s].
    pub max_vel_x_backwards: f64,
    /// Maximum strafing velocity [m/s].
    pub max_vel_y: f64,
    /// Maximum translational velocity for omni robots [m/s].
    pub max_vel_trans: f64,
    /// Maximum angular velocity [rad/s].
    pub max_vel_theta: f64,
    /// Maximum translational acceleration [m/s²].
    pub acc_lim_x: f64,
    /// Maximum strafing acceleration [m/s²].
    pub acc_lim_y: f64,
    /// Maximum angular acceleration [rad/s²].
    pub acc_lim_theta: f64,
    /// Whether to use exact arc-length computation.
    pub exact_arc_length: bool,
    /// Minimum turning radius [m].
    pub min_turning_radius: f64,
}

impl RobotConfig {
    fn default_() -> Self {
        Self {
            max_vel_x: 0.4,
            max_vel_x_backwards: 0.2,
            max_vel_y: 0.0,
            max_vel_trans: 0.0,
            max_vel_theta: 0.3,
            acc_lim_x: 0.5,
            acc_lim_y: 0.5,
            acc_lim_theta: 0.5,
            exact_arc_length: false,
            min_turning_radius: 0.0,
        }
    }
}

impl Default for RobotConfig {
    fn default() -> Self {
        Self::default_()
    }
}

// ── Goal Tolerance Configuration ───────────────────────────────────

/// Goal tolerance parameters.
#[derive(Debug, Clone, Copy)]
pub struct GoalToleranceConfig {
    /// Allowed final orientation error [rad].
    pub yaw_goal_tolerance: f64,
    /// Allowed final euclidean distance [m].
    pub xy_goal_tolerance: f64,
    /// Allow nonzero final velocity.
    pub free_goal_vel: bool,
    /// Robot considered stopped (translation) [m/s].
    pub trans_stopped_vel: f64,
    /// Robot considered stopped (rotation) [rad/s].
    pub theta_stopped_vel: f64,
    /// Must complete entire global plan.
    pub complete_global_plan: bool,
}

impl GoalToleranceConfig {
    fn default_() -> Self {
        Self {
            yaw_goal_tolerance: 0.2,
            xy_goal_tolerance: 0.2,
            free_goal_vel: false,
            trans_stopped_vel: 0.1,
            theta_stopped_vel: 0.1,
            complete_global_plan: true,
        }
    }
}

impl Default for GoalToleranceConfig {
    fn default() -> Self {
        Self::default_()
    }
}

// ── Obstacle Configuration ─────────────────────────────────────────

/// Obstacle avoidance parameters.
#[derive(Debug, Clone)]
pub struct ObstacleConfig {
    /// Minimum separation from obstacles [m].
    pub min_obstacle_dist: f64,
    /// Buffer zone with non-zero penalty [m].
    pub inflation_dist: f64,
    /// Buffer for dynamic obstacles [m].
    pub dynamic_obstacle_inflation_dist: f64,
    /// Consider obstacle movement.
    pub include_dynamic_obstacles: bool,
    /// Consider costmap obstacles.
    pub include_costmap_obstacles: bool,
    /// Limit obstacles behind robot [m].
    pub costmap_obstacles_behind_robot_dist: f64,
    /// Neighboring poses affected by obstacle.
    pub obstacle_poses_affected: usize,
    /// Use legacy obstacle association.
    pub legacy_obstacle_association: bool,
    /// Force-inclusion factor (× min_obstacle_dist).
    pub obstacle_association_force_inclusion_factor: f64,
    /// Cut-off factor (× min_obstacle_dist).
    pub obstacle_association_cutoff_factor: f64,
    /// Costmap converter plugin name.
    pub costmap_converter_plugin: String,
    /// Costmap converter in separate thread.
    pub costmap_converter_spin_thread: bool,
    /// Costmap converter rate.
    pub costmap_converter_rate: usize,
    /// Velocity upper bound ratio near obstacles.
    pub obstacle_proximity_ratio_max_vel: f64,
    /// Distance below which velocity lower [m].
    pub obstacle_proximity_lower_bound: f64,
    /// Distance above which full velocity allowed [m].
    pub obstacle_proximity_upper_bound: f64,
}

impl ObstacleConfig {
    fn default_() -> Self {
        Self {
            min_obstacle_dist: 0.5,
            inflation_dist: 0.6,
            dynamic_obstacle_inflation_dist: 0.6,
            include_dynamic_obstacles: true,
            include_costmap_obstacles: true,
            costmap_obstacles_behind_robot_dist: 1.5,
            obstacle_poses_affected: 25,
            legacy_obstacle_association: false,
            obstacle_association_force_inclusion_factor: 1.5,
            obstacle_association_cutoff_factor: 5.0,
            costmap_converter_plugin: String::new(),
            costmap_converter_spin_thread: true,
            costmap_converter_rate: 5,
            obstacle_proximity_ratio_max_vel: 1.0,
            obstacle_proximity_lower_bound: 0.0,
            obstacle_proximity_upper_bound: 0.5,
        }
    }
}

impl Default for ObstacleConfig {
    fn default() -> Self {
        Self::default_()
    }
}

// ── Optimization Configuration ─────────────────────────────────────

/// Optimization weights and settings.
#[derive(Debug, Clone, Copy)]
pub struct OptimizationConfig {
    /// Solver iterations per outer loop.
    pub no_inner_iterations: usize,
    /// Outer loop iterations.
    pub no_outer_iterations: usize,
    /// Activate optimization.
    pub optimization_activate: bool,
    /// Verbose optimization output.
    pub optimization_verbose: bool,
    /// Safety margin for penalty functions.
    pub penalty_epsilon: f64,
    /// Weight: max translational velocity.
    pub weight_max_vel_x: f64,
    /// Weight: max strafing velocity.
    pub weight_max_vel_y: f64,
    /// Weight: max angular velocity.
    pub weight_max_vel_theta: f64,
    /// Weight: max translational acceleration.
    pub weight_acc_lim_x: f64,
    /// Weight: max strafing acceleration.
    pub weight_acc_lim_y: f64,
    /// Weight: max angular acceleration.
    pub weight_acc_lim_theta: f64,
    /// Weight: non-holonomic kinematics.
    pub weight_kinematics_nh: f64,
    /// Weight: forward drive direction.
    pub weight_kinematics_forward_drive: f64,
    /// Weight: minimum turning radius.
    pub weight_kinematics_turning_radius: f64,
    /// Weight: optimal transition time.
    pub weight_optimaltime: f64,
    /// Weight: shortest path.
    pub weight_shortest_path: f64,
    /// Weight: obstacle avoidance.
    pub weight_obstacle: f64,
    /// Weight: inflation penalty.
    pub weight_inflation: f64,
    /// Weight: dynamic obstacle avoidance.
    pub weight_dynamic_obstacle: f64,
    /// Weight: dynamic obstacle inflation.
    pub weight_dynamic_obstacle_inflation: f64,
    /// Weight: velocity-obstacle ratio.
    pub weight_velocity_obstacle_ratio: f64,
    /// Weight: via-point.
    pub weight_viapoint: f64,
    /// Weight: prefer turning direction.
    pub weight_prefer_rotdir: f64,
    /// Weight adaptation factor per outer iteration.
    pub weight_adapt_factor: f64,
    /// Nonlinear obstacle cost exponent (1.0 = linear).
    pub obstacle_cost_exponent: f64,
}

impl OptimizationConfig {
    fn default_() -> Self {
        Self {
            no_inner_iterations: 5,
            no_outer_iterations: 4,
            optimization_activate: true,
            optimization_verbose: false,
            penalty_epsilon: 0.05,
            weight_max_vel_x: 2.0,
            weight_max_vel_y: 2.0,
            weight_max_vel_theta: 1.0,
            weight_acc_lim_x: 1.0,
            weight_acc_lim_y: 1.0,
            weight_acc_lim_theta: 1.0,
            weight_kinematics_nh: 1000.0,
            weight_kinematics_forward_drive: 1.0,
            weight_kinematics_turning_radius: 1.0,
            weight_optimaltime: 1.0,
            weight_shortest_path: 0.0,
            weight_obstacle: 50.0,
            weight_inflation: 0.1,
            weight_dynamic_obstacle: 50.0,
            weight_dynamic_obstacle_inflation: 0.1,
            weight_velocity_obstacle_ratio: 0.0,
            weight_viapoint: 1.0,
            weight_prefer_rotdir: 50.0,
            weight_adapt_factor: 2.0,
            obstacle_cost_exponent: 1.0,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self::default_()
    }
}

// ── Homotopy Class Planning Configuration ──────────────────────────

/// Homotopy class planning parameters.
#[derive(Debug, Clone, Copy)]
pub struct HcpConfig {
    /// Enable homotopy class planning.
    pub enable_homotopy_class_planning: bool,
    /// Enable multithreading.
    pub enable_multithreading: bool,
    /// Simple left-right exploration.
    pub simple_exploration: bool,
    /// Max homotopy classes.
    pub max_number_classes: usize,
    /// Max plans in current class.
    pub max_number_plans_in_current_class: usize,
    /// Cost hysteresis for selection.
    pub selection_cost_hysteresis: f64,
    /// Prefer initial plan cost reduction.
    pub selection_prefer_initial_plan: f64,
    /// Extra obstacle cost scaling for selection.
    pub selection_obst_cost_scale: f64,
    /// Extra via-point cost scaling for selection.
    pub selection_viapoint_cost_scale: f64,
    /// Use total transition time for selection.
    pub selection_alternative_time_cost: bool,
    /// Dropping probability for non-best trajectories.
    pub selection_dropping_probability: f64,
    /// Duration before class switch allowed [s].
    pub switching_blocking_period: f64,
    /// Roadmap graph samples.
    pub roadmap_graph_no_samples: usize,
    /// Roadmap graph width [m].
    pub roadmap_graph_area_width: f64,
    /// Roadmap graph length scale.
    pub roadmap_graph_area_length_scale: f64,
    /// H-signature prescaler.
    pub h_signature_prescaler: f64,
    /// H-signature equality threshold.
    pub h_signature_threshold: f64,
    /// Keypoint offset for exploration [m].
    pub obstacle_keypoint_offset: f64,
    /// Obstacle heading threshold.
    pub obstacle_heading_threshold: f64,
    /// All candidates attach to via-points.
    pub viapoints_all_candidates: bool,
    /// Visualize homotopy graph.
    pub visualize_hc_graph: bool,
    /// 3D viz time-as-z scale.
    pub visualize_with_time_as_z_axis_scale: f64,
    /// Delete backwards detours.
    pub delete_detours_backwards: bool,
    /// Detour orientation tolerance [rad].
    pub detours_orientation_tolerance: f64,
    /// Start orientation vector length [m].
    pub length_start_orientation_vector: f64,
    /// Max ratio detour/best duration.
    pub max_ratio_detours_duration_best_duration: f64,
}

impl HcpConfig {
    fn default_() -> Self {
        Self {
            enable_homotopy_class_planning: true,
            enable_multithreading: true,
            simple_exploration: false,
            max_number_classes: 5,
            max_number_plans_in_current_class: 1,
            selection_cost_hysteresis: 1.0,
            selection_prefer_initial_plan: 0.95,
            selection_obst_cost_scale: 100.0,
            selection_viapoint_cost_scale: 1.0,
            selection_alternative_time_cost: false,
            selection_dropping_probability: 0.0,
            switching_blocking_period: 0.0,
            roadmap_graph_no_samples: 15,
            roadmap_graph_area_width: 6.0,
            roadmap_graph_area_length_scale: 1.0,
            h_signature_prescaler: 1.0,
            h_signature_threshold: 0.1,
            obstacle_keypoint_offset: 0.1,
            obstacle_heading_threshold: 0.45,
            viapoints_all_candidates: true,
            visualize_hc_graph: false,
            visualize_with_time_as_z_axis_scale: 0.0,
            delete_detours_backwards: true,
            detours_orientation_tolerance: std::f64::consts::PI / 2.0,
            length_start_orientation_vector: 0.4,
            max_ratio_detours_duration_best_duration: 3.0,
        }
    }
}

impl Default for HcpConfig {
    fn default() -> Self {
        Self::default_()
    }
}

// ── Recovery Configuration ─────────────────────────────────────────

/// Recovery and backup parameters.
#[derive(Debug, Clone, Copy)]
pub struct RecoveryConfig {
    /// Temporary 50% horizon shrinkage.
    pub shrink_horizon_backup: bool,
    /// Min duration for reduced horizon [iterations].
    pub shrink_horizon_min_duration: f64,
    /// Detect and resolve oscillations.
    pub oscillation_recovery: bool,
    /// Oscillation linear velocity threshold.
    pub oscillation_v_eps: f64,
    /// Oscillation angular velocity threshold.
    pub oscillation_omega_eps: f64,
    /// Min recovery duration [iterations].
    pub oscillation_recovery_min_duration: f64,
    /// Oscillation filter duration [iterations].
    pub oscillation_filter_duration: f64,
    /// Enable divergence detection.
    pub divergence_detection_enable: bool,
    /// Max Mahalanobis distance for divergence.
    pub divergence_detection_max_chi_squared: usize,
}

impl RecoveryConfig {
    fn default_() -> Self {
        Self {
            shrink_horizon_backup: true,
            shrink_horizon_min_duration: 10.0,
            oscillation_recovery: true,
            oscillation_v_eps: 0.1,
            oscillation_omega_eps: 0.1,
            oscillation_recovery_min_duration: 10.0,
            oscillation_filter_duration: 10.0,
            divergence_detection_enable: false,
            divergence_detection_max_chi_squared: 38,
        }
    }
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self::default_()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_teb_config_defaults() {
        let cfg = TebConfig::default_();
        assert_eq!(cfg.trajectory.dt_ref, 0.3);
        assert_eq!(cfg.trajectory.min_samples, 3);
        assert_eq!(cfg.robot.max_vel_x, 0.4);
        assert_eq!(cfg.robot.max_vel_theta, 0.3);
        assert_eq!(cfg.obstacles.min_obstacle_dist, 0.5);
        assert_eq!(cfg.optim.weight_obstacle, 50.0);
        assert_eq!(cfg.goal_tolerance.xy_goal_tolerance, 0.2);
    }

    #[test]
    fn test_ackermann_config() {
        let cfg = TebConfig::for_ackermann_drive();
        assert_eq!(cfg.min_turning_radius, 0.5);
    }

    #[test]
    fn test_optimization_weights() {
        let cfg = TebConfig::default_();
        assert_eq!(cfg.optim.weight_max_vel_x, 2.0);
        assert_eq!(cfg.optim.weight_optimaltime, 1.0);
        assert_eq!(cfg.optim.weight_shortest_path, 0.0);
    }

    #[test]
    fn test_obstacle_config_defaults() {
        let cfg = ObstacleConfig::default_();
        assert_eq!(cfg.min_obstacle_dist, 0.5);
        assert_eq!(cfg.inflation_dist, 0.6);
    }
}
