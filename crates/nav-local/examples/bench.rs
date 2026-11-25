//! Nav-local hot path benchmarks: band ops and optimizer.
//!
//! Run: `cargo run --release -p nav-local --example bench`

use std::time::Instant;

fn main() {
    println!("=== nav-local Hot Path Benchmarks ===\n");

    bench_band_clone();
    bench_band_push_pop();
    bench_band_insert_remove();

    println!("\n=== Summary ===");
}

fn bench_band_clone() {
    use nav_local::teb::band::TimedElasticBand;
    use nav_local::teb::pose_se2::PoseSe2;
    use nalgebra::Vector2;

    let mut band = TimedElasticBand::new();
    for n in [5, 10, 20, 50] {
        let b = &mut band;
        b.clear();
        for i in 0..n {
            let pose = PoseSe2::new(Vector2::new(i as f64, 0.0), 0.0);
            b.add_pose_and_timediff(pose, 0.1);
        }

        let iterations = if n <= 20 { 100_000 } else { 10_000 };
        let t0 = Instant::now();
        for _ in 0..iterations {
            let _ = band.clone();
        }
        let elapsed = t0.elapsed();
        println!(
            "band clone({:3}): {:8.2} μs/iter",
            n,
            elapsed.as_secs_f64() * 1e6 / iterations as f64
        );
    }
}

fn bench_band_push_pop() {
    use nav_local::teb::band::TimedElasticBand;
    use nav_local::teb::pose_se2::PoseSe2;
    use nalgebra::Vector2;

    let iterations = 10_000;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut band = TimedElasticBand::new();
        band.add_pose_and_timediff(PoseSe2::new(Vector2::new(0.0, 0.0), 0.0), 0.1);
        for i in 1..=50 {
            band.add_pose_and_timediff(PoseSe2::new(Vector2::new(i as f64, 0.0), 0.0), 0.1);
        }
        for _ in 0..49 {
            band.delete_pose(1);
        }
    }
    let elapsed = t0.elapsed();
    println!(
        "band push/pop(50): {:8.2} μs/iter",
        elapsed.as_secs_f64() * 1e6 / iterations as f64
    );
}

fn bench_band_insert_remove() {
    use nav_local::teb::band::TimedElasticBand;
    use nav_local::teb::pose_se2::PoseSe2;
    use nalgebra::Vector2;

    let iterations = 10_000;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut band = TimedElasticBand::new();
        band.add_pose_and_timediff(PoseSe2::new(Vector2::new(0.0, 0.0), 0.0), 0.1);
        band.add_pose_and_timediff(PoseSe2::new(Vector2::new(10.0, 0.0), 0.0), 0.1);
        // Insert 20 intermediate poses
        for i in 1..=20 {
            let pose = PoseSe2::new(Vector2::new(i as f64 * 0.5, 0.0), 0.0);
            band.insert_pose(1, pose);
            band.set_pose_fixed(1, false);
        }

        // Remove all intermediate
        for _ in 0..20 {
            band.delete_pose(1);
        }
    }
    let elapsed = t0.elapsed();
    println!(
        "band insert/delete(20): {:8.2} μs/iter",
        elapsed.as_secs_f64() * 1e6 / iterations as f64
    );
}

