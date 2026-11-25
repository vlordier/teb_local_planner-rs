
//! Kani verification properties for tridiagonal solver used in cubic splines.
//!
//! The tridiagonal solver (`_solve_tridiagonal`) is private, so we verify its
//! correctness through public eval APIs: the solver must produce finite
//! coefficients that yield finite results for well-conditioned inputs.

#![no_main]

extern crate nav_core;
extern crate nav_trajectory;

mod tridiagonal_kani {
    use nav_trajectory::CubicSpline1D;

    /// For a well-conditioned tridiagonal system, the solver produces finite results.
    /// We test with concrete inputs to avoid CBMC floatbv limitations.
    #[kani::proof]
    #[kani::unwind(6)]
    fn tridiagonal_solver_produces_finite_results() {
        let x_data: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
        let y_data: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        // All coefficients should be finite — verify via eval.
        for i in 0..4 {
            let val = spline.eval(x_data[i]);
            assert!(
                val.is_finite(),
                "spline eval at {} should be finite, got {}",
                x_data[i],
                val
            );
        }
    }

    /// Spline derivative and second derivative are finite at all data points.
    #[kani::proof]
    #[kani::unwind(6)]
    fn tridiagonal_solver_derivatives_finite() {
        let x_data: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data: [f64; 5] = [1.0, 2.5, 0.5, 3.0, 1.5];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        for i in 0..5 {
            let x = x_data[i];
            let val = spline.eval(x);
            let deriv = spline.eval_derivative(x);
            let second_deriv = spline.eval_second_derivative(x);

            assert!(val.is_finite(), "eval at {} is not finite: {}", x, val);
            assert!(
                deriv.is_finite(),
                "derivative at {} is not finite: {}",
                x,
                deriv
            );
            assert!(
                second_deriv.is_finite(),
                "second derivative at {} is not finite: {}",
                x,
                second_deriv
            );
        }
    }

    /// Interpolation property: spline evaluates to exact y-value at each data point.
    #[kani::proof]
    #[kani::unwind(6)]
    fn tridiagonal_solver_interpolation_exact() {
        let x_data: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
        let y_data: [f64; 4] = [10.0, -5.0, 3.0, 7.5];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        for i in 0..4 {
            let y_at_x = spline.eval(x_data[i]);
            let expected = y_data[i];
            let diff = (y_at_x - expected).abs();
            assert!(
                diff < 1e-8,
                "interpolation at x={} expected {} got {} (diff {})",
                x_data[i],
                expected,
                y_at_x,
                diff
            );
        }
    }

    /// Natural spline: second derivative is zero (or near-zero) at endpoints.
    #[kani::proof]
    #[kani::unwind(6)]
    fn tridiagonal_solver_natural_boundary() {
        let x_data: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
        let y_data: [f64; 4] = [0.0, 1.0, 0.5, 2.0];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        let second_start = spline.eval_second_derivative(x_data[0]);
        let second_end = spline.eval_second_derivative(x_data[3]);

        assert!(
            second_start.abs() < 1e-8,
            "start second derivative should be ~0, got {}",
            second_start
        );
        assert!(
            second_end.abs() < 1e-8,
            "end second derivative should be ~0, got {}",
            second_end
        );
    }

    /// Larger system (n=8) with varying y-values stays finite.
    #[kani::proof]
    #[kani::unwind(6)]
    fn tridiagonal_solver_larger_system_finite() {
        let x_data: [f64; 8] =
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let y_data: [f64; 8] = [1.0, 0.5, -0.3, 0.8, -1.2, 0.4, 1.8, 0.9];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        // Eval at every data point and midpoints.
        for i in 0..8 {
            let val = spline.eval(x_data[i]);
            assert!(val.is_finite(), "eval at x={} is not finite: {}", x_data[i], val);
        }
        // Midpoints
        for i in 0..7 {
            let mid = (x_data[i] + x_data[i + 1]) / 2.0;
            let val = spline.eval(mid);
            assert!(val.is_finite(), "eval at midpoint {} is not finite: {}", mid, val);
        }
    }
}
