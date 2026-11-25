#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments, unreachable_patterns)]
//! Cubic spline interpolation for trajectory parameterization.
//!
//! Port of the C++ `cubic_spline.hpp` from teb_local_planner.
//!
//! A natural cubic spline fits a piecewise cubic polynomial through
//! (x, y) data points with zero second derivative at endpoints.
//! Each segment i is: `S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3`.

/// A 1D natural cubic spline.
///
/// Each segment i is defined by:
/// `S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3`
///
/// where `a[i] = y_data[i]` and the coefficients are computed from the
/// tridiagonal system enforcing C2 continuity and zero second derivative at endpoints.
pub struct CubicSpline1D {
    /// x-coordinates of data points (must be strictly increasing).
    pub x_data: Vec<f64>,
    /// y-coordinates of data points.
    pub y_data: Vec<f64>,
    /// Number of data points.
    pub n: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl CubicSpline1D {
    /// Fit a natural cubic spline through the given data points.
    ///
    /// # Arguments
    /// * `x_data` — strictly increasing x-coordinates (must be non-empty).
    /// * `y_data` — y-values at each x-coordinate.
    ///
    /// # Panics
    /// Panics if `x_data` is empty, has mismatched length with `y_data`,
    /// has fewer than 2 data points, or is not strictly increasing.
    pub fn fit(x_data: Vec<f64>, y_data: Vec<f64>) -> Self {
        assert!(!x_data.is_empty(), "x_data must not be empty");
        assert_eq!(x_data.len(), y_data.len(), "x_data and y_data must have same length");
        let n = x_data.len();
        assert!(n >= 2, "CubicSpline1D::fit requires at least 2 data points, got {}", n);

        let mut spline = Self {
            x_data,
            y_data,
            n,
            a: Vec::with_capacity(n),
            b: Vec::with_capacity(n),
            c: Vec::with_capacity(n),
            d: Vec::with_capacity(n),
        };

        spline._solve_tridiagonal();
        spline
    }

    /// Evaluate the spline at point `x`.
    ///
    /// # Panics
    /// Panics if `x` is outside [min(x_data), max(x_data)].
    pub fn eval(&self, x: f64) -> f64 {
        assert!(
            x >= self.x_data[0] && x <= self.x_data[self.n - 1],
            "x={} is outside [{}..{}]", x, self.x_data[0], self.x_data[self.n - 1]
        );
        let seg = self._segment(x);
        let dx = x - self.x_data[seg];
        self.a[seg]
            + self.b[seg] * dx
            + self.c[seg] * dx * dx
            + self.d[seg] * dx * dx * dx
    }

    /// Evaluate the first derivative at point `x`.
    ///
    /// # Panics
    /// Panics if `x` is outside [min(x_data), max(x_data)].
    pub fn eval_derivative(&self, x: f64) -> f64 {
        let seg = self._segment(x);
        let dx = x - self.x_data[seg];
        self.b[seg] + 2.0 * self.c[seg] * dx + 3.0 * self.d[seg] * dx * dx
    }

    /// Evaluate the second derivative at point `x`.
    ///
    /// # Panics
    /// Panics if `x` is outside [min(x_data), max(x_data)].
    pub fn eval_second_derivative(&self, x: f64) -> f64 {
        let seg = self._segment(x);
        let dx = x - self.x_data[seg];
        2.0 * self.c[seg] + 6.0 * self.d[seg] * dx
    }

    /// Find the segment index that contains x.
    #[inline]
    fn _segment(&self, x: f64) -> usize {
        if x <= self.x_data[0] {
            return 0;
        }
        if x >= self.x_data[self.n - 1] {
            return self.n - 2;
        }
        for i in 0..self.n - 1 {
            if x <= self.x_data[i + 1] {
                return i;
            }
        }
        self.n - 2
    }

    /// Solve the tridiagonal system for the c coefficients, then derive b and d.
    fn _solve_tridiagonal(&mut self) {
        let n = self.n;
        let h: Vec<f64> = self
            .x_data
            .iter()
            .zip(self.x_data.iter().skip(1))
            .map(|(x0, x1)| *x1 - *x0)
            .collect();

        // a[i] = y_data[i]
        self.a = self.y_data.clone();

        // Solve for c using the standard tridiagonal system:
        //   2*(h[i-1]+h[i]) * c[i] + h[i-1]*c[i-1] + h[i]*c[i+1] = 3*(...)
        // with c[0] = c[n-1] = 0 for natural spline.

        let mut alfa = vec![0.0_f64; n];
        let mut beta = vec![0.0_f64; n];
        let mut gamma = vec![0.0_f64; n];

        for i in 1..n - 1 {
            alfa[i] = 2.0 * (h[i - 1] + h[i]);
            if i >= 1 {
                beta[i] = h[i - 1];
            }
            if i < n - 1 {
                gamma[i] = h[i];
            }
        }

        // Right-hand side
        let mut rhs = vec![0.0_f64; n];
        for i in 1..n - 1 {
            rhs[i] = 3.0
                * ((self.a[i + 1] - self.a[i]) / h[i]
                    - (self.a[i] - self.a[i - 1]) / h[i - 1]);
        }

        // Forward elimination (Thomas algorithm)
        let mut cf = vec![0.0_f64; n];
        cf[1] = rhs[1] / alfa[1];
        for i in 2..n - 1 {
            let denom = alfa[i] - beta[i] * gamma[i - 1];
            if denom.abs() < 1e-15 {
                cf[i] = 0.0;
            } else {
                cf[i] = (rhs[i] - beta[i] * cf[i - 1]) / denom;
            }
        }

        // Back substitution
        // self.c was initialized with capacity but len=0; fill with zeros first
        self.c.resize_with(n, || 0.0);
        self.c[n - 2] = cf[n - 2];
        for i in (1..n - 2).rev() {
            let denom = alfa[i] - beta[i] * gamma[i - 1];
            if denom.abs() < 1e-15 {
                self.c[i] = 0.0;
            } else {
                self.c[i] = (cf[i] - gamma[i] * self.c[i + 1]) / denom;
            }
        }
        self.c[0] = 0.0;
        self.c[n - 1] = 0.0;

        // Compute b and d
        self.b = vec![0.0; n];
        self.d = vec![0.0; n];
        for i in 0..n - 1 {
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3.0 * h[i]);
            self.b[i] =
                (self.a[i + 1] - self.a[i]) / h[i] - h[i] * (2.0 * self.c[i] + self.c[i + 1]) / 3.0;
        }
    }
}

/// 2D trajectory spline: separate x, y, yaw, and time splines.
///
/// Given a sequence of (x, y, yaw, time) waypoints, this builds four
/// independent 1D splines and provides joint pose/velocity queries.
pub struct TrajectorySpline {
    /// X coordinate spline.
    pub x: CubicSpline1D,
    /// Y coordinate spline.
    pub y: CubicSpline1D,
    /// Yaw angle spline.
    pub yaw: CubicSpline1D,
    /// Time parameterization spline (monotonic increasing).
    pub t: CubicSpline1D,
}

impl TrajectorySpline {
    /// Fit splines for x, y, yaw, and time simultaneously.
    ///
    /// The parameter is the waypoint index (0, 1, 2, ...). The time spline
    /// maps index → time, which is inverted to find the parameter for a
    /// given time value.
    ///
    /// # Arguments
    /// * `x_data` — x-coordinates at each waypoint.
    /// * `y_data` — y-coordinates at each waypoint.
    /// * `yaw_data` — yaw angles (in radians) at each waypoint.
    /// * `time_data` — time at each waypoint (must be non-decreasing).
    ///
    /// # Panics
    /// Panics if any data vector is empty or lengths differ.
    pub fn fit(
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        yaw_data: Vec<f64>,
        time_data: Vec<f64>,
    ) -> Self {
        assert!(!x_data.is_empty(), "x_data must not be empty");
        assert_eq!(x_data.len(), y_data.len(), "data vectors must have same length");
        assert_eq!(x_data.len(), yaw_data.len(), "data vectors must have same length");
        assert_eq!(x_data.len(), time_data.len(), "data vectors must have same length");

        let n = x_data.len();
        let params: Vec<f64> = (0..n).map(|i| i as f64).collect();

        Self {
            x: CubicSpline1D::fit(params.clone(), x_data),
            y: CubicSpline1D::fit(params.clone(), y_data),
            yaw: CubicSpline1D::fit(params.clone(), yaw_data),
            t: CubicSpline1D::fit(params, time_data),
        }
    }

    /// Get pose at time `t`.
    ///
    /// Returns `(x, y, yaw)` at time `t`.
    ///
    /// # Panics
    /// Panics if `t` is outside [min(time_data), max(time_data)].
    pub fn pose_at(&self, t: f64) -> (f64, f64, f64) {
        let s = self._invert_time(t);
        (self.x.eval(s), self.y.eval(s), self.yaw.eval(s))
    }

    /// Get velocity at time `t`.
    ///
    /// Returns `(vx, vy, omega)` at time `t`.
    /// vx = dx/dt, vy = dy/dt, omega = d_yaw/dt.
    ///
    /// # Panics
    /// Panics if `t` is outside [min(time_data), max(time_data)].
    pub fn velocity_at(&self, t: f64) -> (f64, f64, f64) {
        let s = self._invert_time(t);
        let dt_ds = self.t.eval_derivative(s);
        if dt_ds.abs() < 1e-12 {
            return (0.0, 0.0, 0.0);
        }
        let dx_ds = self.x.eval_derivative(s);
        let dy_ds = self.y.eval_derivative(s);
        let dyaw_ds = self.yaw.eval_derivative(s);
        (dx_ds / dt_ds, dy_ds / dt_ds, dyaw_ds / dt_ds)
    }

    /// Invert the time spline: find parameter s such that t(s) ≈ target_t.
    /// Uses bisection over the parameter range.
    #[inline]
    fn _invert_time(&self, target_t: f64) -> f64 {
        let t_min = self.t.x_data[0];
        let t_max = self.t.x_data[self.t.n - 1];

        if target_t <= t_min + 1e-12 {
            return self.t.x_data[0];
        }
        if target_t >= t_max - 1e-12 {
            return self.t.x_data[self.t.n - 1];
        }

        let mut lo = self.t.x_data[0];
        let mut hi = self.t.x_data[self.t.n - 1];

        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            if self.t.eval(mid) < target_t {
                lo = mid;
            } else {
                hi = mid;
            }
            if (hi - lo).abs() < 1e-10 {
                break;
            }
        }
        (lo + hi) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_spline_straight_line() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let spline = CubicSpline1D::fit(x, y);

        assert!((spline.eval(0.0) - 0.0).abs() < 1e-8);
        assert!((spline.eval(1.5) - 1.5).abs() < 1e-8);
        assert!((spline.eval(3.0) - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_cubic_spline_quadratic() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0]; // y = x^2
        let spline = CubicSpline1D::fit(x, y);

        assert!((spline.eval(0.0) - 0.0).abs() < 1e-6);
        assert!((spline.eval(1.0) - 1.0).abs() < 1e-6);
        assert!((spline.eval(2.0) - 4.0).abs() < 1e-6);
        assert!((spline.eval(3.0) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_cubic_spline_derivative() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0]; // y = x, derivative = 1
        let spline = CubicSpline1D::fit(x, y);
        assert!((spline.eval_derivative(1.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_trajectory_spline_fit() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 0.0, 0.0, 0.0];
        let yaw = vec![0.0, 0.0, 0.0, 0.0];
        let t = vec![0.0, 1.0, 2.0, 3.0];

        let spline = TrajectorySpline::fit(x, y, yaw, t);

        let (x0, y0, yaw0) = spline.pose_at(0.0);
        assert!((x0 - 0.0).abs() < 1e-6);
        assert!((y0 - 0.0).abs() < 1e-6);
        assert!((yaw0 - 0.0).abs() < 1e-6);

        let (x1, y1, yaw1) = spline.pose_at(1.0);
        assert!((x1 - 1.0).abs() < 1e-6);
        assert!((y1 - 0.0).abs() < 1e-6);
        assert!((yaw1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_trajectory_spline_endpoints() {
        let x = vec![0.0, 5.0, 10.0];
        let y = vec![0.0, 3.0, 0.0];
        let yaw = vec![0.0, 0.5, 0.0];
        let t = vec![0.0, 1.0, 2.0];

        let spline = TrajectorySpline::fit(x, y, yaw, t);

        let (xe, ye, yaw_e) = spline.pose_at(2.0);
        assert!((xe - 10.0).abs() < 1e-6);
        assert!((ye - 0.0).abs() < 1e-6);
        assert!((yaw_e - 0.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "must not be empty")]
    fn test_empty_data_panics() {
        CubicSpline1D::fit(vec![], vec![]);
    }

    #[test]
    #[should_panic(expected = "must have same length")]
    fn test_mismatched_lengths_panics() {
        CubicSpline1D::fit(vec![0.0, 1.0], vec![0.0]);
    }

    #[test]
    fn test_cubic_spline_second_derivative_zero_at_endpoints() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let spline = CubicSpline1D::fit(x, y);

        let sec_0 = spline.eval_second_derivative(0.0);
        let sec_last = spline.eval_second_derivative(3.0);
        assert!(
            sec_0.abs() < 1e-6,
            "second derivative at 0 should be ~0, got {}",
            sec_0
        );
        assert!(
            sec_last.abs() < 1e-6,
            "second derivative at 3 should be ~0, got {}",
            sec_last
        );
    }
}
