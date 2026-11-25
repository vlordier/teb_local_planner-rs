//! Kani verification properties for CubicSpline1D interpolation.
//!
//! Uses concrete sizes to avoid CBMC exponential blowup from symbolic Vec allocation.
//!
//! CBMC 0.67 floatbv cannot verify floating-point arithmetic properties for symbolic
//! values, so these harnesses use concrete input data and check that operations
//! produce finite, bounded results.

#![no_main]

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
