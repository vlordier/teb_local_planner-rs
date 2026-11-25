//! Kani verification properties for CubicSpline1D interpolation.
//!
//! Uses concrete sizes to avoid CBMC exponential blowup from symbolic Vec allocation,
//! symbolic loop counts, and iterator chains over symbolic sizes.
//!
//! Properties verified by Kani's symbolic execution:
//! - Spline interpolates all data points
//! - Second derivative is zero at endpoints (natural spline condition)
//! - Derivative is finite and bounded

#![no_main]

mod cubic_spline_kani {
    use nav_trajectory::CubicSpline1D;
    use std::f64::consts::PI;

    /// Interpolation at data points for n=3.
    #[kani::proof]
    #[kani::unwind(4)]
    fn cubic_spline_interpolation_n3() {
        let x_data: [f64; 3] = [0.0, 1.0, 2.0];
        let y_data: [f64; 3] = [0.0, 0.8414709848, 0.9092974268];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        // Unrolled: check each point
        assert!(
            (spline.eval(0.0) - 0.0).abs() < 1e-10,
            "Spline should interpolate y_data[0] = 0.0 exactly, got {}",
            spline.eval(0.0)
        );
        assert!(
            (spline.eval(1.0) - 0.8414709848).abs() < 1e-10,
            "Spline should interpolate y_data[1] = 0.8414709848 exactly, got {}",
            spline.eval(1.0)
        );
        assert!(
            (spline.eval(2.0) - 0.9092974268).abs() < 1e-10,
            "Spline should interpolate y_data[2] = 0.9092974268 exactly, got {}",
            spline.eval(2.0)
        );
    }

    /// Interpolation at data points for n=5.
    #[kani::proof]
    #[kani::unwind(6)]
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

        assert!(
            (spline.eval(0.0) - 0.0).abs() < 1e-10,
            "Spline should interpolate y_data[0] = 0.0 exactly, got {}",
            spline.eval(0.0)
        );
        assert!(
            (spline.eval(1.0) - 0.8414709848).abs() < 1e-10,
            "Spline should interpolate y_data[1] = 0.8414709848 exactly, got {}",
            spline.eval(1.0)
        );
        assert!(
            (spline.eval(2.0) - 0.9092974268).abs() < 1e-10,
            "Spline should interpolate y_data[2] = 0.9092974268 exactly, got {}",
            spline.eval(2.0)
        );
        assert!(
            (spline.eval(3.0) - 0.1411200081).abs() < 1e-10,
            "Spline should interpolate y_data[3] = 0.1411200081 exactly, got {}",
            spline.eval(3.0)
        );
        assert!(
            (spline.eval(4.0) + 0.7568024953).abs() < 1e-10,
            "Spline should interpolate y_data[4] = -0.7568024953 exactly, got {}",
            spline.eval(4.0)
        );
    }

    /// Second derivative is zero at endpoints for n=4.
    #[kani::proof]
    #[kani::unwind(4)]
    fn cubic_spline_endpoint_zero_second_derivative() {
        let x_data: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
        let y_data: [f64; 4] = [5.0, 4.8414709848, 3.0806046117, -1.0504165607];
        let spline = CubicSpline1D::fit(x_data.to_vec(), y_data.to_vec());

        let second_deriv_start = spline.eval_second_derivative(0.0);
        let second_deriv_end = spline.eval_second_derivative(3.0);

        assert!(
            second_deriv_start.abs() < 1e-6,
            "start second deriv: {}",
            second_deriv_start
        );
        assert!(
            second_deriv_end.abs() < 1e-6,
            "end second deriv: {}",
            second_deriv_end
        );
    }

    /// Derivative is finite and bounded for n=3.
    #[kani::proof]
    #[kani::unwind(4)]
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
