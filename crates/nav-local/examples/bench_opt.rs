use std::time::Instant;

fn main() {
    use nav_local::teb::band::TimedElasticBand;
    use nav_local::teb::optimizer::TEBOptimizer;
    use nav_local::teb::pose_se2::PoseSe2;
    use nav_core::constraints::TebConfig;
    use nalgebra::Vector2;

    // Build band with 20 poses
    let mut band = TimedElasticBand::new();
    for i in 0..20 {
        let pose = PoseSe2::new(Vector2::new(i as f64, (i as f64 * 0.1).sin()), 0.0);
        band.add_pose_and_timediff(pose, 0.1);
    }

    let config = TebConfig::default();

    // Just time one full optimization pass
    let iterations = 5;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut opt = TEBOptimizer::new(band.clone(), config.clone());
        opt.set_max_outer_iterations(1);
        opt.set_max_inner_iterations(20);
        let _ = opt.optimize();
    }
    let elapsed = t0.elapsed();
    println!(
        "optimizer(20 poses, 1 outer x 20 inner): {:8.2} ms/iter",
        elapsed.as_secs_f64() * 1e3 / iterations as f64
    );

    // Now with 5 inner iterations
    let iterations = 10;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut opt = TEBOptimizer::new(band.clone(), config.clone());
        opt.set_max_outer_iterations(1);
        opt.set_max_inner_iterations(5);
        let _ = opt.optimize();
    }
    let elapsed = t0.elapsed();
    println!(
        "optimizer(20 poses, 1 outer x 5 inner):  {:8.2} ms/iter",
        elapsed.as_secs_f64() * 1e3 / iterations as f64
    );
}
