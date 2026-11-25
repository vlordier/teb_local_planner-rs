//! Bevy simulation systems for TEB planning.
//!
//! Provides a 2D physics simulation loop where:
//! - Robots integrate pose using kinematic vehicle models
//! - TEB planning runs each tick to produce velocity commands
//! - Dynamic obstacles move and the costmap updates from scene data
//!
//! Adapted from concepts in `test_optim_node.cpp`.

use bevy::prelude::*;
use nav_core::{DifferentialModel, PoseSE2, VehicleModel};
use nav_local::teb::PoseSe2;

// ─── Velocity / command types ────────────────────────────────────

/// A velocity command in SE2 (twist-like).
///
/// Equivalent to ROS `geometry_msgs/Twist` projected to 2D.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Twist {
    /// Forward velocity [m/s].
    pub vx: f64,
    /// Strafe velocity [m/s] (nonzero for holonomic robots).
    pub vy: f64,
    /// Angular velocity [rad/s].
    pub omega: f64,
}

impl Twist {
    /// Zero velocity.
    #[inline]
    pub fn zero() -> Self {
        Self {
            vx: 0.0,
            vy: 0.0,
            omega: 0.0,
        }
    }

    /// Euclidean speed.
    #[inline]
    pub fn speed(&self) -> f64 {
        (self.vx * self.vx + self.vy * self.vy).sqrt()
    }
}

impl Default for Twist {
    fn default() -> Self {
        Self::zero()
    }
}

/// Velocity command target for a simulated robot.
#[derive(Component, Debug, Clone, Copy)]
pub struct VelocityCommand {
    pub twist: Twist,
}

// ─── Simulation components ───────────────────────────────────────

/// A robot entity being simulated in the Bevy world.
#[derive(Component)]
pub struct SimulatedRobot {
    /// Current pose in SE2 (local frame).
    pub pose: PoseSe2,
    /// Current twist estimate.
    pub twist: Twist,
    /// Kinematic constraints for integration.
    pub model: Box<dyn VehicleModel>,
}

/// Planning target with goal and optional via-points.
#[derive(Component, Clone)]
pub struct PlanningTarget {
    /// Goal pose the robot should reach.
    pub goal: PoseSe2,
    /// Via-points the planner should visit en route.
    pub via_points: Vec<PoseSe2>,
}

/// Sensor data: obstacles and costmap.
#[derive(Component, Clone)]
pub struct SensorData {
    /// Detected dynamic obstacles (world frame).
    pub obstacles: Vec<DynamicObstacle>,
    /// 2D occupancy grid covering the explored area.
    pub costmap: Costmap,
}

/// A moving obstacle detected by sensors.
#[derive(Clone)]
pub struct DynamicObstacle {
    /// Pose in world frame.
    pub pose: PoseSe2,
    /// Velocity in world frame.
    pub twist: Twist,
    /// Footprint polygon (robot-relative).
    pub footprint: Vec<[f64; 2]>,
}

/// 2D costmap for occupancy grid.
///
/// Each cell stores a continuous cost value (e.g. distance-field based).
#[derive(Clone)]
pub struct Costmap {
    /// Grid width in cells.
    pub width: u32,
    /// Grid height in cells.
    pub height: u32,
    /// Cell resolution [m/cell].
    pub resolution: f64,
    /// Origin of the costmap in world coords.
    pub origin_x: f64,
    pub origin_y: f64,
    /// Cell costs (row-major).
    pub data: Vec<f64>, // 0 = free, 255 = occupied
}

impl Costmap {
    /// Create a new empty costmap.
    pub fn new(width: u32, height: u32, resolution: f64, origin_x: f64, origin_y: f64) -> Self {
        let data = vec![0.0; (width * height) as usize];
        Self {
            width,
            height,
            resolution,
            origin_x,
            origin_y,
            data,
        }
    }

    /// Check if a world coordinate is occupied.
    pub fn is_occupied(&self, x: f64, y: f64) -> bool {
        let col = ((x - self.origin_x) / self.resolution) as usize;
        let row = ((y - self.origin_y) / self.resolution) as usize;
        if col >= self.width as usize || row >= self.height as usize {
            return true;
        }
        let idx = row * self.width as usize + col;
        self.data[idx] >= 255.0
    }
}

impl Default for Costmap {
    fn default() -> Self {
        Self::new(100, 100, 0.05, 0.0, 0.0)
    }
}

// ─── Stub planner type ───────────────────────────────────────────

/// Placeholder TEB planner wrapper.
///
/// In production this wraps `nav_local::teb::TimedElasticBand` and the
/// LM optimizer. Here it exposes the API expected by
/// [`run_teb_planning`].
#[derive(Component)]
pub struct TEBPlanner {
    /// Planner configuration.
    pub config: TEBConfig,
}

/// Stub TEB configuration mirroring [`nav_core::TebConfig`].
///
/// Re-exports the real nav-core config under the name used by the
/// user-facing API.
pub type TEBConfig = nav_core::TebConfig;

impl TEBPlanner {
    /// Plan a trajectory from `start` to `goal`, avoiding obstacles.
    ///
    /// Returns a [`Twist`] command if planning succeeds, or `None`
    /// if the planner is blocked (e.g. by obstacles).
    pub fn plan(
        &mut self,
        _start: &PoseSe2,
        _goal: &PoseSe2,
        _costmap: &Costmap,
        _obstacles: &[DynamicObstacle],
    ) -> Option<Twist> {
        // Stub: return a simple forward command.
        // In production this would call the LM optimizer on a TimedElasticBand.
        Some(Twist {
            vx: 0.5,
            vy: 0.0,
            omega: 0.0,
        })
    }
}

// ─── Systems ─────────────────────────────────────────────────────

/// Integrate robot pose using kinematic vehicle model.
///
/// Reads [`VelocityCommand`] components and updates [`SimulatedRobot`]
/// with simple Euler integration.
pub fn simulate_robot(
    time: Res<Time>,
    mut query: Query<(&mut SimulatedRobot, Option<&VelocityCommand>)>,
) {
    let dt = time.delta_secs();
    for (mut robot, cmd) in &mut query {
        // Apply velocity command
        if let Some(vcmd) = cmd {
            robot.twist = vcmd.twist;
        }

        // Forward integrate: x' = cos(yaw)*vx, y' = sin(yaw)*vx, yaw' = omega
        let theta = robot.pose.theta();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        *robot.pose.x_mut() += cos_theta * robot.twist.vx * dt;
        *robot.pose.y_mut() += sin_theta * robot.twist.vx * dt;
        *robot.pose.theta_mut() += robot.twist.omega * dt;

        // Normalize theta to [-pi, pi]
        let mut t = *robot.pose.theta_mut();
        while t > std::f64::consts::PI {
            t -= 2.0 * std::f64::consts::PI;
        }
        while t < -std::f64::consts::PI {
            t += 2.0 * std::f64::consts::PI;
        }
        *robot.pose.theta_mut() = t;
    }
}

/// Run TEB planning for each simulated robot.
///
/// Queries for entities carrying [`SimulatedRobot`] and
/// [`PlanningTarget`]. Creates a [`TEBPlanner`] per robot if one
/// doesn't already exist, then runs planning and writes the result
/// into a [`VelocityCommand`].
pub fn run_teb_planning(
    mut commands: Commands,
    mut query: Query<(
        Entity,
        &mut SimulatedRobot,
        &PlanningTarget,
        &SensorData,
        Option<&mut VelocityCommand>,
    )>,
) {
    for (entity, mut robot, target, sensor, maybe_cmd) in query.iter_mut() {
        // Create a planner if one doesn't exist
        let planner_entity = commands
            .spawn(TEBPlanner {
                config: TEBConfig::default(),
            })
            .id();

        // Run planning
        let twist = {
            let mut planner = commands
                .entity(planner_entity)
                .get_mut::<TEBPlanner>()
                .unwrap();
            planner.plan(
                &robot.pose,
                &target.goal,
                &sensor.costmap,
                &sensor.obstacles,
            )
        };

        if let Some(twist) = twist {
            robot.twist = twist;
            // Write command (or create one if missing)
            if let Some(mut cmd) = maybe_cmd {
                cmd.twist = twist;
            } else {
                commands.entity(entity).insert(VelocityCommand { twist });
            }
        }

        // Despawn the temporary planner entity
        commands.entity(planner_entity).despawn();
    }
}

/// Update dynamic obstacle positions and the costmap from scene.
pub fn update_sensor_data(time: Res<Time>, mut query: Query<&mut SensorData>) {
    let dt = time.delta_secs();
    for mut sensor in &mut query {
        for obs in &mut sensor.obstacles {
            *obs.pose.x_mut() += obs.twist.vx * dt;
            *obs.pose.y_mut() += obs.twist.vy * dt;
            let mut t = obs.pose.theta();
            t += obs.twist.omega * dt;
            // Normalize theta
            while t > std::f64::consts::PI {
                t -= 2.0 * std::f64::consts::PI;
            }
            while t < -std::f64::consts::PI {
                t += 2.0 * std::f64::consts::PI;
            }
            *obs.pose.theta_mut() = t;
        }
    }
}

/// Update costmap from scene objects.
///
/// Queries entities with [`GlobalTransform`] and marks their positions
/// on the costmap as occupied.
pub fn update_costmap(
    query: Query<(&GlobalTransform, &Transform)>,
    mut costmap: Query<&mut Costmap>,
) {
    // Clear costmap
    for mut cm in costmap.iter_mut() {
        cm.data.iter_mut().for_each(|v| *v = 0.0);
    }

    // Mark occupied cells from scene objects
    for (global, local) in query.iter() {
        if global.translation().z > 0.001 {
            continue; // Only 2D objects
        }
        let x = global.translation().x;
        let y = global.translation().y;
        for mut cm in costmap.iter_mut() {
            // Mark a circular area around the object
            let r_cells = (0.3 / cm.resolution) as usize;
            for dr in -r_cells..=r_cells {
                for dc in -r_cells..=r_cells {
                    let dx = dc as f64 * cm.resolution;
                    let dy = dr as f64 * cm.resolution;
                    if dx * dx + dy * dy <= 0.09 {
                        let col = ((x - cm.origin_x + dx) / cm.resolution) as usize;
                        let row = ((y - cm.origin_y + dy) / cm.resolution) as usize;
                        if col < cm.width as usize && row < cm.height as usize {
                            let idx = row * cm.width as usize + col;
                            cm.data[idx] = 255.0;
                        }
                    }
                }
            }
        }
    }
}
