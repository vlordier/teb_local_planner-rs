//! Kani verification properties for CubicSpline1D interpolation.
//!
//! Properties verified by Kani's symbolic execution:
//! - Spline interpolates all data points
//! - Second derivative is zero at endpoints (natural spline condition)

#![no_main]

mod cubic_spline_kani {
    /// CubicSpline interpolates all data points exactly.
    #[kani::proof]
    #[kani::unwind(4)]
    fn cubic_spline_interpolation() {
        use nav_trajectory::CubicSpline1D;

        // Construct a spline with a known number of points
        let n: u8 = kani::any();
        let n = ((n % 5) as usize) + 2; // 2..=6 points
        let scale: f64 = kani::any();
        let scale = scale.abs().max(0.1);

        let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y_data: Vec<f64> = (0..n)
            .map(|i| (i as f64).sin() * scale)
            .collect();

        let spline = CubicSpline1D::fit(x_data.clone(), y_data.clone());

        // Check that the spline interpolates every data point
        for i in 0..n {
            let x = x_data[i];
            let y = y_data[i];
            let interp = spline.eval(x);
            assert!(
                (interp - y).abs() < 1e-10,
                "Spline should interpolate y_data[{}] = {} exactly, got {}",
                i,
                y,
                interp
            );
        }
    }

    /// Natural cubic spline: second derivative is zero at endpoints.
    #[kani::proof]
    #[kani::unwind(4)]
    fn cubic_spline_endpoint_zero_second_derivative() {
        use nav_trajectory::CubicSpline1D;

        let n: u8 = kani::any();
        let n = ((n % 5) as usize) + 2;

        let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y_data: Vec<f64> = (0..n)
            .map(|i| (i as f64).cos() * 5.0)
            .collect();

        let spline = CubicSpline1D::fit(x_data, y_data);

        // Natural spline: second derivative = 0 at both endpoints
        let second_deriv_start = spline.eval_second_derivative(0.0);
        let second_deriv_end = spline.eval_second_derivative(n as f64 - 1.0);

        assert!(
            second_deriv_start.abs() < 1e-10,
            "Natural spline second derivative at start should be ~0, got {}",
            second_deriv_start
        );
        assert!(
            second_deriv_end.abs() < 1e-10,
            "Natural spline second derivative at end should be ~0, got {}",
            second_deriv_end
        );
    }

    /// CubicSpline derivative is bounded.
    #[kani::proof]
    #[kani::unwind(4)]
    fn cubic_spline_derivative_bounded() {
        use nav_trajectory::CubicSpline1D;

        let n: u8 = kani::any();
        let n = ((n % 5) as usize) + 2;

        let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y_data: Vec<f64> = (0..n)
            .map(|i| (i as f64).sin() * 10.0)
            .collect();

        let spline = CubicSpline1D::fit(x_data, y_data);

        // Check derivative at all data points is finite
        for &x in &spline.x_data {
            let deriv = spline.eval_derivative(x);
            assert!(
                deriv.is_finite(),
                "Spline derivative at x={} should be finite, got {}",
                x,
                deriv
            );
            assert!(
                deriv.abs() < 1e6,
                "Spline derivative at x={} should be bounded, got {}",
                x,
                deriv
            );
        }
    }
}
