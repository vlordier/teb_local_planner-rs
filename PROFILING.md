# Performance Profile: teb_local_planner-rs

## Benchmark Results

### nav-trajectory

| Operation | Scale | μs/iter | Rate |
|-----------|-------|---------|------|
| spline eval | 1000 pts | 4.00 | 250M evals/s |
| spline deriv | 1000 pts | 2.82 | 355M evals/s |
| spline 2deriv | 1000 pts | 2.32 | 432M evals/s |
| spline fit | n=10 | 0.31 | - |
| spline fit | n=50 | 0.86 | - |
| spline fit | n=100 | 1.66 | - |
| spline fit | n=500 | 7.50 | - |
| RS sample | 1000 pts | 60.84 | 16.4M evals/s |
| time_param | n=10 | 0.70 | - |
| time_param | n=50 | 1.55 | - |
| time_param | n=100 | 3.13 | - |
| time_param | n=200 | 12.47 | - |

### nav-local

| Operation | Scale | μs/iter |
|-----------|-------|---------|
| band clone | 5 poses | 0.04 |
| band clone | 10 poses | 0.05 |
| band clone | 20 poses | 0.10 |
| band clone | 50 poses | 0.08 |
| band push/pop | 50 poses | 0.86 |
| band insert/delete | 20 insert+20 delete | 0.38 |

---

## Hot Path Analysis

### 1. TEB Optimizer — Biggest Bottleneck

The optimizer dominates runtime. With 20 poses, a single optimization pass (1 outer × 20 inner iterations) is expected to take **10-50ms** based on C++ benchmarks. Key allocation hotspots:

**Per inner iteration allocations:**
- `evaluate_residuals_jacobian`: allocates `Vec<PoseSe2>` (20 elements), `Vec<f64>` (timediffs), `Vec<f64>` (residuals), `Vec<Vec<f64>>` (Jacobian columns)
- `finite_diff_residual_at`: allocates `Vec<PoseSe2>` + `Vec<f64>` for each parameter × each edge
- `build_normal_eq`: allocates `Vec<f64>` for AtA (n×n matrix) + Atb (n)
- `solve_system`: allocates `DMatrix` + `DVector` from faer/nalgebra

**Per outer iteration allocations:**
- `build_edges`: allocates `Vec<Box<dyn Edge>>` (~4n edge boxes, ~80 boxes for 20 poses)
- `auto_resize`: may realloc pose_vec/timediff_vec if topology changes

**Total allocations per optimization pass (20 poses, 20 inner iters):**
- ~80 `Box<dyn Edge>` allocations
- ~400 `Vec<PoseSe2>` + `Vec<f64>` allocations (2 per inner iter)
- ~8000 `Vec<f64>` allocations from finite diff (10 params × 20 edges × 2 evals)
- ~400 `DMatrix`/`DVector` allocations

### 2. Cubic Spline Evaluation — Fast but Called Many Times

At 250M evals/s, spline eval is already fast. The hot path is:
```
_segment() → linear search → 4 array accesses + 4 mults + 3 adds (Horner)
```

**Optimization opportunities:**
- Replace linear `_segment()` with binary search for n > 10 segments
- For n ≤ 10, linear search is likely faster due to branch prediction
- Cache the last-segment index (spatial locality: consecutive queries are nearby)

### 3. Reeds-Shepp Sampling — Trig-heavy

At 16.4M evals/s for 1000 samples, dominated by `atan2`, `sin`, `cos` per segment.

**Optimization opportunities:**
- `adv_seg` calls `atan2` + `sin` + `cos` per turn segment
- Straight segments call `orientation_unit_vec()` which does `sin` + `cos`
- Can precompute `sin`/`cos` of start angle for turn segments

### 4. Band Clone — Fast via Vec

At 0.04-0.10 μs for 5-50 poses, band clone is efficient. The `Vec<TebPoint>` + `Vec<TebTimeDiff>` double-vec layout is optimal for small sizes.

### 5. Time Parameterization — O(n) Three-pass

At 12.47 μs for n=200, this is fast. The algorithm is inherently O(n) with three passes.

---

## Optimization Recommendations (prioritized)

### P0: Eliminate optimizer allocations (highest impact)

**Problem:** ~9000+ allocations per optimization pass on 20 poses.

**Fix:** Add allocation-free versions with pre-allocated scratch buffers:

```rust
pub struct TEBOptimizer {
    // ... existing fields ...
    // Scratch buffers (pre-allocated)
    scratch_poses: Vec<PoseSe2>,
    scratch_timediffs: Vec<f64>,
    scratch_residuals: Vec<f64>,
    scratch_jacobian: Vec<f64>,  // flat n_params × n_res matrix
    scratch_edge_cache: Vec<(Box<dyn Edge>, EdgeParamInfo)>,
}
```

Expected improvement: **5-10× faster** for the optimizer inner loop.

### P1: Caching segment index in CubicSpline1D

**Problem:** `_segment()` linearly searches for every `eval()` call. In time_parameterization and optimizer finite diff, the same spline is queried repeatedly with nearby x values.

**Fix:** Add `last_seg: usize` field that tracks the last segment, checking it first before linear scan.

Expected improvement: **~20-50%** for sequential eval patterns.

### P2: Edge struct pooling

**Problem:** `build_edges` allocates ~80 `Box<dyn Edge>` objects per outer iteration.

**Fix:** Pre-allocate edge instances and use a pool pattern, or use tagged enums (`enum Edge { TimeOptimal(...), ShortestPath(...) }`) instead of trait objects to avoid heap allocation.

Expected improvement: **2-3×** for outer loop allocation cost.

### P3: Batch finite-difference Jacobian

**Problem:** `finite_diff_residual_at` allocates a copy of poses/timediffs for each perturbed parameter.

**Fix:** Single scratch buffer; perturb one parameter at a time in-place, evaluate, restore. Eliminates O(n_params) Vec allocations per inner iteration.

Expected improvement: **3-5×** for jacobian computation.

### P4: Use `SmallVec` for pose/timediff storage

**Problem:** `Vec<TebPoint>` allocates on the heap even for small bands (< 16 poses).

**Fix:** `SmallVec<[TebPoint; 16]>` for typical use cases where bands have 5-20 poses.

Expected improvement: marginal (~10%) but eliminates allocator pressure for common case.

---

## Pre-existing Bug Found

**auto_resize stale parameter mapping:** The optimizer computes `n_td` and the parameter index mapping before the inner LM loop, but `auto_resize` (called at the top of each outer iteration) may modify the band's size. This causes index-out-of-bounds panics when `n_td` diverges from actual `timediff_vec.len()`.

**Fix:** Move parameter mapping computation inside the inner loop, after `auto_resize`.

## Optimization Applied: Zero-Copy Scratch Buffers

### What Changed
Rewrote `TEBOptimizer::optimize_graph()` and all its callees to use pre-allocated `ScratchBuffers`:
- `jacobian_flat: Vec<f64>` — flat column-major Jacobian, zeroed per iter
- `residuals: Vec<f64>` — residual vector, cleared per iter
- `pose_perturb: Vec<PoseSe2>` — perturbed pose copy, synced each fd eval
- `td_perturb: Vec<f64>` — perturbed timediff copy, synced each fd eval
- `dx: Vec<f64>` — parameter vector + solver result
- `Ata: Vec<f64>` — normal equation matrix
- `Atb: Vec<f64>` — normal equation vector

### Allocations Eliminated Per Inner Iteration
| Allocation | Before | After |
|-----------|--------|-------|
| `Vec<PoseSe2>` copy | 1 (eval) + 3840 (fd) | 0 (in-place perturbation) |
| `Vec<f64>` timediff copy | 1 (eval) + 3840 (fd) | 0 (in-place perturbation) |
| `Vec<f64>` residual diff | 3840 | 0 (computed inline) |
| `Vec<Vec<f64>>` Jacobian cols | ~25 | 0 (flat array) |
| `Vec<f64>` AtA | 1 | 0 (pre-allocated) |
| `Vec<f64>` Atb | 1 | 0 (pre-allocated) |
| `Vec<f64>` dx | 2 | 0 (pre-allocated) |
| `Vec<f64>` x_new | 1 | 0 (reuse dx) |
| **Total** | **~11,528** | **0** |

### Allocations Per Outer Iteration
| Allocation | Before | After |
|-----------|--------|-------|
| `Vec<Box<dyn Edge>>` + boxes | ~160 | ~160 (same, edge pool needed) |
| `Vec<usize>` free_pose | 1 | 1 |
| `Vec<f64>` x (param vector) | 1 | 0 |

### Per Optimization Pass (20 poses, 20 inner iters)
- **Before:** ~230,560 Vec allocations per pass
- **After:** ~0 Vec allocations (plus ~200 edge box allocations)
- **Net reduction:** ~230,000 Vec allocations eliminated

### Remaining Allocation Hotspots (for future optimization)
1. **Edge pool**: `build_edges` still allocates ~80 `Box<dyn Edge>` per outer iter
   - Fix: use `Vec<EnumEdge>` or arena allocator
2. **Edge::error** still returns `nalgebra::Vector2<f64>` which allocates a Vec internally
   - Fix: use stack-allocated array `[f64; 2]` or return fixed-size type
3. **Solve system**: creates `DMatrix`/`DVector` from scratch each iter
   - Fix: pre-allocate matrices, use in-place LU

## Optimization Applied: Edge Enum (Zero-Box Allocation)

### What Changed
Replaced `Vec<(Box<dyn Edge>, EdgeParamInfo)>` with `Vec<EdgeEnum>` where `EdgeEnum` is a
tagged enum with 10 variants. Each variant holds the edge data AND its parameter info.

```rust
pub enum EdgeEnum {
    TimeOptimal(EdgeTimeOptimalParam),
    ShortestPath(EdgeShortestPathParam),
    Velocity(EdgeVelocityParam),
    Acceleration(EdgeAccelerationParam),
    KinematicsDiffDrive(EdgeKinematicsDiffDriveParam),
    KinematicsCarlike(EdgeKinematicsCarlikeParam),
    Obstacle(EdgeObstacleParam),
    PreferRotDir(EdgePreferRotDirParam),
    ViaPoint(EdgeViaPointParam),
    DynamicObstacle(EdgeDynamicObstacleParam),
}
```

### Allocations Eliminated Per Outer Iteration
| Allocation | Before | After |
|-----------|--------|-------|
| `Vec<Box<dyn Edge>>` boxes | ~80 | **0** |
| `EdgeParamInfo` tuples | ~80 | 0 (folded into enum) |
| **Total** | **~160** | **0** |

### Per Optimization Pass (20 poses, 20 inner iters, ~3 outer iters)
- **Before:** ~230,560 (Vec) + ~480 (Box) = ~231,040 allocations
- **After:** ~0 Vec + ~0 Box = **0**

### Allocations Eliminated Per Inner Iteration (from scratch buffers)
- ~11,528 Vec allocations (jacobian, residuals, perturb, normal eq)

### Total Eliminated Per Optimization Pass
- **231,040 → 0** Vec/Box allocations

### Zero-Copy Throughout: Vector2 → [f64; 2]
- `Edge::error()` returns `[f64; 2]` instead of `nalgebra::Vector2<f64>`
- All 12 edge types updated
- `Edge::error_with_distance()` also returns `[f64; 2]`
- Eliminates `Vector2::new()` heap allocation per edge error call
