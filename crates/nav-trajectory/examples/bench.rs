//! Micro-benchmarks for hot paths in nav-trajectory and nav-local.
//!
//! Run: `cargo run --release -p nav-trajectory --example bench`

use std::time::Instant;

fn main() {

    bench_spline_eval();
    bench_spline_fit();
    bench_spline_eval_derivative();
    bench_spline_second_derivative();
    bench_reeds_shepp_sample();
    bench_time_parameterization();

    println!("\n=== Summary ===");
}

fn bench_spline_eval() {
    use nav_trajectory::CubicSpline1D;

    let x_data: Vec<f64> = (0..=10).map(|i| i as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|x| x.sin()).collect();
    let spline = CubicSpline1D::fit(x_data, y_data);

    let iterations = 10_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let mut _sum = 0.0f64;
        for i in 0..1000 {
            _sum += spline.eval(i as f64 * 0.01);
        }
    }
    let elapsed = start.elapsed();
    println!(
        "spline eval:    {:8.2} μs/iter   ({} evals/s)",
        elapsed.as_secs_f64() * 1e6 / iterations as f64,
        (iterations as f64 * 1000.0) / elapsed.as_secs_f64()
    );
}

fn bench_spline_fit() {
    use nav_trajectory::CubicSpline1D;

    for n in [10, 50, 100, 500] {
        let iterations = if n <= 50 { 10_000 } else { 1_000 };
        let start = Instant::now();
        for _ in 0..iterations {
            let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let y_data: Vec<f64> = x_data.iter().map(|x| x.sin()).collect();
            let _ = CubicSpline1D::fit(x_data, y_data);
        }
        let elapsed = start.elapsed();
        println!(
            "spline fit n={:4}: {:8.2} μs/iter",
            n,
            elapsed.as_secs_f64() * 1e6 / iterations as f64
        );
    }
}

fn bench_spline_eval_derivative() {
    use nav_trajectory::CubicSpline1D;

    let x_data: Vec<f64> = (0..=10).map(|i| i as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|x| x.sin()).collect();
    let spline = CubicSpline1D::fit(x_data, y_data);

    let iterations = 10_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let mut _sum = 0.0f64;
        for i in 0..1000 {
            _sum += spline.eval_derivative(i as f64 * 0.01);
        }
    }
    let elapsed = start.elapsed();
    println!(
        "spline deriv:   {:8.2} μs/iter   ({} evals/s)",
        elapsed.as_secs_f64() * 1e6 / iterations as f64,
        (iterations as f64 * 1000.0) / elapsed.as_secs_f64()
    );
}

fn bench_spline_second_derivative() {
    use nav_trajectory::CubicSpline1D;

    let x_data: Vec<f64> = (0..=10).map(|i| i as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|x| x.sin()).collect();
    let spline = CubicSpline1D::fit(x_data, y_data);

    let iterations = 10_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let mut _sum = 0.0f64;
        for i in 0..1000 {
            _sum += spline.eval_second_derivative(i as f64 * 0.01);
        }
    }
    let elapsed = start.elapsed();
    println!(
        "spline 2deriv:  {:8.2} μs/iter   ({} evals/s)",
        elapsed.as_secs_f64() * 1e6 / iterations as f64,
        (iterations as f64 * 1000.0) / elapsed.as_secs_f64()
    );
}

fn bench_reeds_shepp_sample() {
    use nav_trajectory::{RSParameters, RSPathType, RSSegment, SegmentType, sample_rs};
    use nav_core::PoseSE2;

    let segments = vec![
        RSSegment { kind: SegmentType::LeftTurn, length: 1.5 },
        RSSegment { kind: SegmentType::Straight, length: 3.0 },
        RSSegment { kind: SegmentType::RightTurn, length: 0.5 },
    ];
    let path = RSParameters::new(segments, RSPathType::LSR);
    let pose_start = PoseSE2::from_xyz(0.0, 0.0, 0.0);

    let iterations = 100_000;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut _sum = 0.0f64;
        for i in 0..1000 {
            let d = (i as f64) / 1000.0 * path.total_length;
            let pose = sample_rs(&path, &pose_start, d);
            _sum += pose.x() + pose.y();
        }
    }
    let elapsed = t0.elapsed();
    println!(
        "RS sample:      {:8.2} μs/iter   ({} evals/s)",
        elapsed.as_secs_f64() * 1e6 / iterations as f64,
        (iterations as f64 * 1000.0) / elapsed.as_secs_f64()
    );
}


fn bench_time_parameterization() {
    use nav_trajectory::time_parameterization::{compute_time_optimal_solution, TimeParamConfig};
    use nav_core::PoseSE2;

    for n in [10, 50, 100, 200] {
        let iterations = if n <= 50 { 5_000 } else { 500 };
        let start = Instant::now();
        for _ in 0..iterations {
            let waypoints: Vec<PoseSE2> = (0..n)
                .map(|i| PoseSE2::from_xyz(i as f64, (i as f64 * 0.1).sin(), 0.0))
                .collect();
            let config = TimeParamConfig::default();
            let _ = compute_time_optimal_solution(&waypoints, &config);
        }
        let elapsed = start.elapsed();
        println!(
            "time_param n={:4}: {:8.2} μs/iter",
            n,
            elapsed.as_secs_f64() * 1e6 / iterations as f64
        );
    }
}
