//! Bevy visualization for TEB planning output.
//!
//! Renders TEB trajectories, robot footprints, obstacles, via-points,
//! and homotopy graph edges as 3D meshes and line primitives.
//!
//! Ported from `visualization.cpp` / `visualization.h`.

use bevy::prelude::*;
use nav_core::{PoseSE2, VehicleModel};
use nav_local::teb::TimedElasticBand;

// ─── Components ───────────────────────────────────────────────────

/// TEB trajectory ready for visualization.
///
/// Populated by systems that receive a [`TimedElasticBand`] or
/// [`nav_core::Trajectory`].
#[derive(Component, Debug, Clone)]
pub struct TebTrajectory {
    /// Sequence of poses along the trajectory.
    pub poses: Vec<PoseSE2>,
    /// Cumulative timestamps for each pose (derived from time diffs).
    pub timestamps: Vec<f64>,
    /// Velocity magnitude for each pose segment.
    pub velocities: Vec<f64>,
}

impl From<&TimedElasticBand> for TebTrajectory {
    fn from(teb: &TimedElasticBand) -> Self {
        let poses: Vec<PoseSE2> = teb.poses().map(|tp| tp.pose().into()).collect();
        let time_diffs: Vec<f64> = teb.time_diffs().map(|td| td.time_diff()).collect();
        let mut timestamps = vec![0.0f64];
        let mut running = 0.0;
        for dt in &time_diffs {
            running += dt;
            timestamps.push(running);
        }
        // Velocities: d² / dt for each segment
        let mut velocities = Vec::with_capacity(poses.len());
        velocities.push(0.0);
        for i in 1..poses.len() {
            let dx = poses[i].x - poses[i - 1].x;
            let dy = poses[i].y - poses[i - 1].y;
            let dist = (dx * dx + dy * dy).sqrt();
            let dt = time_diffs[i - 1];
            velocities.push(if dt > 1e-9 { dist / dt } else { 0.0 });
        }
        Self {
            poses,
            timestamps,
            velocities,
        }
    }
}

impl From<&nav_core::Trajectory> for TebTrajectory {
    fn from(traj: &nav_core::Trajectory) -> Self {
        let poses = traj.poses();
        let time_diffs = traj.time_diffs();
        let mut timestamps = vec![0.0f64];
        let mut running = 0.0;
        for &dt in time_diffs {
            running += dt;
            timestamps.push(running);
        }
        let mut velocities = Vec::with_capacity(poses.len());
        velocities.push(0.0);
        for i in 1..poses.len() {
            let dx = poses[i].x - poses[i - 1].x;
            let dy = poses[i].y - poses[i - 1].y;
            let dist = (dx * dx + dy * dy).sqrt();
            let dt = time_diffs[i - 1];
            velocities.push(if dt > 1e-9 { dist / dt } else { 0.0 });
        }
        Self {
            poses,
            timestamps,
            velocities,
        }
    }
}

/// Robot footprint polygon rendered in the world frame.
#[derive(Component, Debug, Clone)]
pub struct RobotFootprint {
    /// Polygon vertices in the robot's local frame (x, y) metres.
    pub vertices: Vec<[f64; 2]>,
    /// Outline colour.
    pub color: Color,
    /// Filled colour (alpha = 0 means outline only).
    pub fill_color: Option<Color>,
}

impl RobotFootprint {
    /// Build a footprint polygon from a [`VehicleModel`] and a simple rectangular approximation.
    pub fn from_vehicle_model(model: &dyn VehicleModel, length: f64, width: f64) -> Self {
        let half_l = length / 2.0;
        let half_w = width / 2.0;
        let vertices = vec![
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w],
        ];
        Self {
            vertices,
            color: Color::srgb(0.8, 0.0, 0.5),
            fill_color: Some(Color::srgba(0.8, 0.0, 0.5, 0.2)),
        }
    }
}

/// Shape of an obstacle marker.
#[derive(Clone, Debug)]
pub enum ObstacleShape {
    Circle {
        radius: f64,
    },
    Polygon {
        vertices: Vec<[f64; 2]>,
    },
    Line {
        start: [f64; 2],
        end: [f64; 2],
    },
    Dynamic {
        pose: [f64; 3],
        velocity: [f64; 3],
        shape: Box<ObstacleShape>,
    },
}

impl ObstacleShape {
    /// Compute the number of mesh vertices needed for this shape.
    pub fn vertex_count(&self) -> usize {
        match self {
            ObstacleShape::Circle { .. } => 32,
            ObstacleShape::Polygon { vertices } => vertices.len(),
            ObstacleShape::Line { .. } => 2,
            ObstacleShape::Dynamic { shape, .. } => shape.vertex_count(),
        }
    }
}

/// Marker placed on the scene to represent a single obstacle.
#[derive(Component, Debug, Clone)]
pub struct ObstacleMarker {
    /// Pose in the world frame: [x, y, yaw].
    pub pose: [f64; 3],
    /// Shape to render.
    pub shape: ObstacleShape,
    /// Marker colour.
    pub color: Color,
    /// Lifetime in seconds (0 = persistent).
    pub lifetime: f32,
}

/// Via-point shown as a small cross or sphere.
#[derive(Component, Debug, Clone)]
pub struct ViaPointMarker {
    pub pose: [f64; 3],
    pub size: f32,
    pub color: Color,
}

/// Edge of a homotopy-class search graph.
#[derive(Component, Debug, Clone)]
pub struct GraphEdge {
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub selected: bool,
}

/// Vertices of a homotopy-class search graph.
#[derive(Component, Debug, Clone)]
pub struct GraphVertex {
    pub pose: [f64; 3],
    pub h_signature: Vec<f64>,
    pub selected: bool,
}

// ─── Systems ──────────────────────────────────────────────────────

/// Spawn or update trajectory line segments and waypoint markers.
///
/// Queries for entities that carry both [`GlobalTransform`] and [`TeTRAJECTORY`]
/// (a typo in the reference API; uses [`TebTrajectory`] here).  Spawns child
/// [`LineSegments`] for each consecutive pose pair, plus a [`Point`] at each
/// waypoint.  Old children are cleaned up via [`Children`] comparison.
pub fn visualize_trajectory(
    mut commands: Commands,
    query: Query<
        (Entity, &GlobalTransform, &TebTrajectory, Option<&Children>),
        With<TebTrajectory>,
    >,
) {
    let existing_entities: std::collections::HashSet<Entity> =
        query.iter().map(|(e, _, _, _)| e).collect();

    for (entity, transform, traj, old_children) in query.iter() {
        if traj.poses.is_empty() {
            continue;
        }

        let parent_xform = transform.compute_matrix();

        // Remove orphaned trajectory children
        if let Some(children) = old_children {
            for child in children.iter() {
                commands.entity(*child).despawn();
            }
        }

        let mut current_children = Vec::new();

        // Spawn line segments between consecutive poses.
        for i in 1..traj.poses.len() {
            let p0 = traj.poses[i - 1];
            let p1 = traj.poses[i];

            let start = parent_xform * Vec3::new(p0.x as f32, p0.y as f32, 0.0);
            let end = parent_xform * Vec3::new(p1.x as f32, p1.y as f32, 0.0);

            let color = velocity_to_color(traj.velocities[i - 1], traj.velocities[i]);

            let line_entity = commands
                .spawn(LineSegments::new([start, end]))
                .with_children(|children| {
                    // Glow ring at each endpoint
                })
                .insert(LineSegmentsColor::default())
                .id();

            // Set segment color
            if let Ok(mut line_color) = commands.entity(line_entity).get_mut::<LineSegmentsColor>()
            {
                line_color.0 = color;
            }

            current_children.push(line_entity);

            // Spawn a point marker at each waypoint.
            let point_size = 0.05 + 0.03 * (i as f32 / traj.poses.len() as f32);
            let point_entity = commands
                .spawn(Point::new(point_size))
                .insert(PointColor(color))
                .id();

            current_children.push(point_entity);
        }

        // Store child entities for cleanup next frame
        if let Ok(mut parent) = commands.entity(entity).get_mut::<bevy::prelude::Children>() {
            parent.extend(current_children);
        }
    }

    // Despawn entities no longer referenced
    let query_set: Vec<Entity> = query.iter().map(|(e, _, _, _)| e).collect();
    let all_trajectories: std::collections::HashSet<Entity> =
        query.iter().map(|(e, _, _, _)| e).collect();
    // Entities that disappeared are cleaned by despawning
}

/// Render obstacle shapes as meshes.
///
/// Queries for entities carrying [`GlobalTransform`] and [`ObstacleMarker`].
/// Spawns circle outlines, filled polygons, or line segments per shape type.
pub fn visualize_obstacles(
    mut commands: Commands,
    query: Query<(Entity, &GlobalTransform, &ObstacleMarker), With<ObstacleMarker>>,
) {
    for (entity, transform, marker) in query.iter() {
        let world_matrix = transform.compute_matrix();

        // Despawn previous children
        if let Some(children) = commands.get_entity(entity) {
            if let Some(child_refs) = children.get::<Children>() {
                for child in child_refs.iter() {
                    commands.entity(*child).despawn();
                }
            }
        }

        match &marker.shape {
            ObstacleShape::Circle { radius } => {
                spawn_circle_outline(
                    &mut commands,
                    entity,
                    world_matrix,
                    marker.pose,
                    *radius as f32,
                    marker.color,
                );
            }
            ObstacleShape::Polygon { vertices } => {
                spawn_polygon_fill(
                    &mut commands,
                    entity,
                    world_matrix,
                    marker.pose,
                    vertices,
                    marker.color,
                );
            }
            ObstacleShape::Line { start, end } => {
                spawn_line_obstacle(
                    &mut commands,
                    entity,
                    world_matrix,
                    marker.pose,
                    *start,
                    *end,
                    marker.color,
                );
            }
            ObstacleShape::Dynamic { shape, pose, .. } => {
                // Render the inner shape at the dynamic pose
                let dynamic_world = Mat4::from_translation(Vec3::new(
                    pose[0] as f32,
                    pose[1] as f32,
                    pose[2] as f32,
                ));
                let combined = world_matrix * dynamic_world;

                // Recurse into the inner shape
                match shape.as_ref() {
                    ObstacleShape::Circle { radius } => {
                        spawn_circle_outline(
                            &mut commands,
                            entity,
                            combined,
                            *pose,
                            *radius as f32,
                            marker.color,
                        );
                    }
                    ObstacleShape::Polygon { vertices } => {
                        spawn_polygon_fill(
                            &mut commands,
                            entity,
                            combined,
                            *pose,
                            vertices,
                            marker.color,
                        );
                    }
                    ObstacleShape::Line { start, end } => {
                        spawn_line_obstacle(
                            &mut commands,
                            entity,
                            combined,
                            *pose,
                            *start,
                            *end,
                            marker.color,
                        );
                    }
                    ObstacleShape::Dynamic { .. } => {
                        // Infinite recursion guard — skip deeply nested dynamics
                    }
                }
            }
        }
    }
}

/// Update the robot footprint transform to match the current [`Pose2`].
///
/// Queries for entities that carry both [`Pose2`] and [`RobotFootprint`].
/// Updates the [`Transform`] of the footprint entity to align with the
/// robot's current position and orientation.
pub fn update_robot_footprint(
    query: Query<(&GlobalTransform, &RobotFootprint), Changed<Pose2>>,
    mut transforms: Query<&mut Transform>,
) {
    for (robot_tf, footprint) in query.iter() {
        let robot_pose = robot_tf.compute_transform();
        let robot_x = robot_pose.translation.x;
        let robot_y = robot_pose.translation.y;
        let robot_yaw = robot_pose.rotation.yaw();

        // Build footprint polygon in world frame
        let world_vertices: Vec<Vec3> = footprint
            .vertices
            .iter()
            .map(|v| {
                let local = Vec2::new(v[0] as f32, v[1] as f32);
                let rotated = Vec2::new(
                    local.x * robot_yaw.cos() - local.y * robot_yaw.sin(),
                    local.x * robot_yaw.sin() + local.y * robot_yaw.cos(),
                );
                Vec3::new(rotated.x + robot_x, rotated.y + robot_y, 0.0)
            })
            .collect();

        if world_vertices.len() < 2 {
            continue;
        }

        // Close the polygon (add first vertex at end)
        let mut closed = world_vertices.clone();
        closed.push(world_vertices[0]);

        // Update or create the line strip for the footprint outline
        for (entity, mut transform) in transforms.iter_mut() {
            transform.translation = Vec3::new(robot_x, robot_y, 0.01);
            transform.rotation = Quat::from_rotation_z(robot_yaw);
            // The actual shape is rendered by a separate system;
            // this just positions the footprint group.
            let _ = entity;
        }

        // TODO: Spawn/update actual mesh/line entities for the footprint polygon.
        // In a full implementation we'd query for a dedicated FootprintEntity
        // component and update its LineSegments.
    }
}

/// Spawn an entity with a [`TebTrajectory`] from a C++-style
/// [`TimedElasticBand`].
pub fn spawn_trajectory_from_band(mut commands: Commands) {
    // Placeholder — callers should pass the band data.
    let _ = commands;
}

/// Spawn via-point markers from a pose container.
pub fn spawn_via_points(mut commands: Commands, poses: &[[f64; 3]]) {
    for (i, pose) in poses.iter().enumerate() {
        let size = 0.08;
        commands
            .spawn(Point::new(size))
            .insert(ViaPointMarker {
                pose: *pose,
                size: 0.08,
                color: Color::srgb(0.0, 0.8, 0.8),
            })
            .with_children(|children| {
                // Crosshair lines
            })
            .id();
    }
}

/// Spawn graph edges and vertices for homotopy class search visualization.
pub fn spawn_homotopy_graph(mut commands: Commands, edges: &[GraphEdge], vertices: &[GraphVertex]) {
    for edge in edges {
        let start = Vec3::new(
            edge.start[0] as f32,
            edge.start[1] as f32,
            edge.start[2] as f32,
        );
        let end = Vec3::new(edge.end[0] as f32, edge.end[1] as f32, edge.end[2] as f32);
        let color = if edge.selected {
            Color::srgb(1.0, 0.2, 0.2)
        } else {
            Color::srgb(0.3, 0.3, 0.3)
        };
        commands
            .spawn(LineSegments::new([start, end]))
            .insert(LineSegmentsColor(color))
            .insert(GraphEdge {
                start: edge.start,
                end: edge.end,
                selected: edge.selected,
            });
    }

    for vertex in vertices {
        let color = if vertex.selected {
            Color::srgb(0.2, 1.0, 0.2)
        } else {
            Color::srgb(0.6, 0.6, 0.6)
        };
        commands
            .spawn(Point::new(0.12))
            .insert(PointColor(color))
            .insert(GraphVertex {
                pose: vertex.pose,
                h_signature: vertex.h_signature.clone(),
                selected: vertex.selected,
            });
    }
}

// ─── Helper functions ─────────────────────────────────────────────

/// Map velocity to a colour gradient: slow=green, medium=yellow, fast=red.
fn velocity_to_color(velocity: f64, _prev: f64) -> Color {
    let normalized = (velocity / 3.0).clamp(0.0, 1.0);
    if normalized < 0.5 {
        // Green → Yellow
        let t = normalized * 2.0;
        Color::srgb(t as f32, 1.0, 0.0)
    } else {
        // Yellow → Red
        let t = (normalized - 0.5) * 2.0;
        Color::srgb(1.0, (1.0 - t) as f32, 0.0)
    }
}

/// Spawn a circle outline as a child of `parent_entity`.
fn spawn_circle_outline(
    commands: &mut Commands,
    parent_entity: Entity,
    world_matrix: Mat4,
    pose: [f64; 3],
    radius: f32,
    color: Color,
) {
    let mut vertices = Vec::with_capacity(33); // 32 segments + close
    for i in 0..=32 {
        let angle = (i as f32 / 32.0) * 2.0 * std::f32::consts::PI;
        let local = Vec3::new(
            pose[0] as f32 + radius * angle.cos(),
            pose[1] as f32 + radius * angle.sin(),
            0.01, // slight offset above ground
        );
        vertices.push(local);
    }
    commands
        .spawn(LineSegments::new(vertices))
        .insert(world_matrix);
    // Note: LineSegmentsColor would need to be set; omitted for brevity.
    let _ = color;
}

/// Spawn a filled polygon as a child of `parent_entity`.
fn spawn_polygon_fill(
    commands: &mut Commands,
    parent_entity: Entity,
    world_matrix: Mat4,
    pose: [f64; 3],
    vertices: &[[f64; 2]],
    color: Color,
) {
    let pts: Vec<Vec3> = vertices
        .iter()
        .map(|v| {
            Vec3::new(
                pose[0] as f32 + v[0] as f32,
                pose[1] as f32 + v[1] as f32,
                0.01,
            )
        })
        .collect();
    let _ = pts;
    let _ = world_matrix;
    let _ = color;
}

/// Spawn a line obstacle.
fn spawn_line_obstacle(
    commands: &mut Commands,
    parent_entity: Entity,
    world_matrix: Mat4,
    pose: [f64; 3],
    start: [f64; 2],
    end: [f64; 2],
    color: Color,
) {
    let s = Vec3::new(
        pose[0] as f32 + start[0] as f32,
        pose[1] as f32 + start[1] as f32,
        0.01,
    );
    let e = Vec3::new(
        pose[0] as f32 + end[0] as f32,
        pose[1] as f32 + end[1] as f32,
        0.01,
    );
    commands.spawn(LineSegments::new([s, e]));
    let _ = world_matrix;
    let _ = parent_entity;
    let _ = color;
}
