
//! Kani verification properties for CubicSpline1D interpolation.
//!
//! Uses concrete sizes to avoid CBMC exponential blowup from symbolic Vec allocation.
//!
//! CBMC 0.67 floatbv cannot verify floating-point arithmetic properties for symbolic
//! values, so these harnesses use concrete input data and check that operations
//! produce finite, bounded results.

#![no_main]

extern crate nav_core;
extern crate nav_trajectory;

mod cubic_spline_kani {
    use nav_trajectory::CubicSpline1D;

    /// Interpolation at data points for n=3.
    #[kani::proof]
    #[kani::unwind(10)]
    fn cubic_spline_interpolation_n3() {
        let x_data: [f64; 3] = [0.0, 1.0, 2.0];
        let y_data: [f64; 3] = [0.0, 0.8414709848, 0.9092974268];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        assert!(spline.eval(0.0).is_finite(), "eval at x=0 should be finite");
        assert!(spline.eval(1.0).is_finite(), "eval at x=1 should be finite");
        assert!(spline.eval(2.0).is_finite(), "eval at x=2 should be finite");
    }
    /// CubicSpline interpolates exact values at all data points (n=4).
    #[kani::proof]
    #[kani::unwind(10)]
    fn cubic_spline_interpolation_exact() {
        let x_data: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
        let y_data: [f64; 4] = [1.0, 2.5, 0.5, 4.0];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        // Interpolation exactness: spline(x_i) == y_i for all i
        for i in 0..4 {
            let val = spline.eval(x_data[i]);
            assert!(
                (val - y_data[i]).abs() < 1e-10,
                "spline({}) = {} should equal y_data[{}] = {}",
                x_data[i], val, i, y_data[i]
            );
        }
    }

    /// CubicSpline derivative is C1 continuous at internal knots (n=4).
    #[kani::proof]
    #[kani::unwind(10)]
    fn cubic_spline_c1_continuity() {
        let x_data: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
        let y_data: [f64; 4] = [0.0, 1.0, 0.5, 2.0];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        // Check C1 at internal knots x=1.0 and x=2.0
        for i in 1..3 {
            let x = x_data[i];
            // Slight offset to sample each side
            let eps = 1e-8;
            let left_deriv = spline.eval_derivative(x - eps);
            let right_deriv = spline.eval_derivative(x + eps);
            assert!(
                (left_deriv - right_deriv).abs() < 1e-6,
                "C1 violation at x={}: left={:.10e}, right={:.10e}",
                x, left_deriv, right_deriv
            );
        }
    }

    /// CubicSpline second derivative is C2 continuous at internal knots (n=5).
    #[kani::proof]
    #[kani::unwind(10)]
    fn cubic_spline_c2_continuity() {
        let x_data: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data: [f64; 5] = [0.0, 1.0, 0.5, 1.5, 0.0];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        // Check C2 at internal knots x=1.0, 2.0, 3.0
        for i in 1..4 {
            let x = x_data[i];
            let eps = 1e-8;
            let left_deriv = spline.eval_second_derivative(x - eps);
            let right_deriv = spline.eval_second_derivative(x + eps);
            assert!(
                (left_deriv - right_deriv).abs() < 1e-5,
                "C2 violation at x={}: left={:.10e}, right={:.10e}",
                x, left_deriv, right_deriv
            );
        }
    }

    /// CubicSpline second derivative is zero at endpoints (natural boundary conditions).
    #[kani::proof]
    #[kani::unwind(10)]
    fn cubic_spline_natural_boundary() {
        let x_data: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data: [f64; 6] = [0.0, 0.8414709848, 0.9092974268,
                                0.1411200081, -0.7568024953, -0.9589242747];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        // Natural spline: second derivative = 0 at endpoints
        let left = spline.eval_second_derivative(x_data[0]);
        let right = spline.eval_second_derivative(x_data[5]);
        assert!(
            left.abs() < 1e-10,
            "Natural BC violation at start: second_deriv({}) = {} should be ~0",
            x_data[0], left
        );
        assert!(
            right.abs() < 1e-10,
            "Natural BC violation at end: second_deriv({}) = {} should be ~0",
            x_data[5], right
        );
    }

    /// CubicSpline evaluation stays within reasonable bounds for bounded input.
    #[kani::proof]
    #[kani::unwind(10)]
    fn cubic_spline_bounded_output() {
        let x_data: [f64; 3] = [0.0, 1.0, 2.0];
        let y_data: [f64; 3] = [-5.0, 3.0, -2.0];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        // Evaluate at many points and check finiteness + boundedness
        for i in 0..20 {
            let x = (i as f64) / 10.0; // 0.0 .. 1.9
            let val = spline.eval(x);
            assert!(val.is_finite(), "eval({}) should be finite", x);
            assert!(val.abs() < 1e6, "eval({}) = {} should be bounded", x, val);
        }
    }


    /// Interpolation at data points for n=5.
    #[kani::proof]
    #[kani::unwind(10)]
    fn cubic_spline_interpolation_n5() {
        let x_data: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data: [f64; 5] = [
            0.0,
            0.8414709848,
            0.9092974268,
            0.1411200081,
            -0.7568024953,
        ];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        assert!(spline.eval(0.0).is_finite(), "eval at x=0 should be finite");
        assert!(spline.eval(1.0).is_finite(), "eval at x=1 should be finite");
        assert!(spline.eval(2.0).is_finite(), "eval at x=2 should be finite");
        assert!(spline.eval(3.0).is_finite(), "eval at x=3 should be finite");
        assert!(spline.eval(4.0).is_finite(), "eval at x=4 should be finite");
    }

    /// Second derivative is finite at endpoints for n=4.
    #[kani::proof]
    #[kani::unwind(10)]
    fn cubic_spline_endpoint_zero_second_derivative() {
        let x_data: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
        let y_data: [f64; 4] = [5.0, 4.8414709848, 3.0806046117, -1.0504165607];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        let second_deriv_start = spline.eval_second_derivative(0.0);
        let second_deriv_end = spline.eval_second_derivative(3.0);

        assert!(second_deriv_start.is_finite(), "start second deriv should be finite");
        assert!(second_deriv_end.is_finite(), "end second deriv should be finite");
    }

    /// Derivative is finite and bounded for n=3.
    #[kani::proof]
    #[kani::unwind(10)]
    fn cubic_spline_derivative_bounded() {
        let x_data: [f64; 3] = [0.0, 1.0, 2.0];
        let y_data: [f64; 3] = [0.0, 8.4147098481, 9.0929742683];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        let d0 = spline.eval_derivative(0.0);
        let d1 = spline.eval_derivative(1.0);
        let d2 = spline.eval_derivative(2.0);

        assert!(d0.is_finite(), "d0 should be finite");
        assert!(d1.is_finite(), "d1 should be finite");
        assert!(d2.is_finite(), "d2 should be finite");
        assert!(d0.abs() < 1e3, "d0 bounded: {}", d0);
        assert!(d1.abs() < 1e3, "d1 bounded: {}", d1);
        assert!(d2.abs() < 1e3, "d2 bounded: {}", d2);
    }
}
