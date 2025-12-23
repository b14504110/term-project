import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

# -----------------------------
# Constants & presets
# -----------------------------
G_DEFAULT = 9.81

SHAPES = {
    "Sphere": {"Cd": 0.47, "d": 0.10},
    "Cylinder": {"Cd": 0.82, "d": 0.10},
    "Bullet (Ogive)": {"Cd": 0.15, "d": 0.10},
    "Custom": {"Cd": 0.47, "d": 0.10},
}


def area_from_diameter(d: float) -> float:
    d = max(float(d), 1e-6)
    return np.pi * d * d / 4.0


def analytic_no_drag(v0: float, theta_deg: float, y0: float, g: float, t: np.ndarray) -> np.ndarray:
    th = np.deg2rad(theta_deg)
    vx0 = v0 * np.cos(th)
    vy0 = v0 * np.sin(th)
    x = vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2
    vx = np.full_like(t, vx0)
    vy = vy0 - g * t
    return np.column_stack([x, y, vx, vy])


def deriv(state: np.ndarray, params: dict) -> np.ndarray:
    """
    d/dt [x, y, vx, vy] with gravity + drag + wind (relative velocity).
    """
    x, y, vx, vy = state
    m = params["m"]
    g = params["g"]
    model = params["drag_model"]  # "None" | "Linear" | "Quadratic"
    k_lin = params["k_lin"]
    rho = params["rho"]
    Cd = params["Cd"]
    A = params["A"]
    wx = params["wx"]
    wy = params["wy"]

    dxdt = vx
    dydt = vy

    ax = 0.0
    ay = -g

    if model != "None":
        v = np.array([vx, vy], dtype=float)
        w = np.array([wx, wy], dtype=float)
        v_rel = v - w
        speed = np.linalg.norm(v_rel)

        if model == "Linear":
            a_drag = -(k_lin / m) * v_rel
            ax += a_drag[0]
            ay += a_drag[1]

        elif model == "Quadratic":
            if speed > 1e-12:
                a_drag = -(0.5 * rho * Cd * A / m) * speed * v_rel
                ax += a_drag[0]
                ay += a_drag[1]

    return np.array([dxdt, dydt, ax, ay], dtype=float)


def step_euler(state: np.ndarray, dt: float, params: dict) -> np.ndarray:
    return state + dt * deriv(state, params)


def step_rk4(state: np.ndarray, dt: float, params: dict) -> np.ndarray:
    k1 = deriv(state, params)
    k2 = deriv(state + 0.5 * dt * k1, params)
    k3 = deriv(state + 0.5 * dt * k2, params)
    k4 = deriv(state + dt * k3, params)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_numeric(v0: float, theta_deg: float, y0: float, params: dict):
    """
    Simulate until y hits ground or tmax.
    Returns t (N,), states (N,4) [x,y,vx,vy]
    """
    dt = params["dt"]
    tmax = params["tmax"]
    solver = params["solver"]  # "Euler" | "RK4"

    th = np.deg2rad(theta_deg)
    state = np.array([0.0, y0, v0 * np.cos(th), v0 * np.sin(th)], dtype=float)
    stepper = step_rk4 if solver == "RK4" else step_euler

    times = [0.0]
    states = [state.copy()]

    t = 0.0
    for _ in range(int(np.ceil(tmax / dt))):
        t += dt
        state = stepper(state, dt, params)
        times.append(t)
        states.append(state.copy())
        if state[1] <= 0.0 and t > 0.0:
            break

    times = np.array(times, dtype=float)
    states = np.vstack(states)

    # Interpolate last point to y=0
    if states.shape[0] >= 2 and states[-1, 1] < 0.0:
        y1, y2 = states[-2, 1], states[-1, 1]
        if (y1 - y2) != 0:
            alpha = y1 / (y1 - y2)
            states[-1] = states[-2] + alpha * (states[-1] - states[-2])
            times[-1] = times[-2] + alpha * (times[-1] - times[-2])
            states[-1, 1] = 0.0

    return times, states


def summary_metrics(t: np.ndarray, S: np.ndarray) -> dict:
    x, y, vx, vy = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    speed = np.sqrt(vx * vx + vy * vy)
    imax = int(np.argmax(y))
    return {
        "range": float(x[-1]),
        "flight_time": float(t[-1]),
        "max_height": float(y[imax]),
        "t_at_max": float(t[imax]),
        "v_impact": float(speed[-1]),
    }


# -----------------------------
# App
# -----------------------------
class ProjectileApp:
    def __init__(self):
        # state
        self.is_animating = False
        self.anim_i = 0

        # parameters
        self.v0 = 50.0
        self.theta = 45.0
        self.y0 = 0.0

        self.m = 1.0
        self.g = G_DEFAULT

        self.drag_model = "Quadratic"
        self.solver = "RK4"

        self.dt = 0.01
        self.tmax = 30.0

        self.rho = 1.225
        self.k_lin = 0.15

        self.shape_name = "Bullet (Ogive)"
        self.Cd = SHAPES[self.shape_name]["Cd"]
        self.d = SHAPES[self.shape_name]["d"]

        self.wx = 0.0
        self.wy = 0.0

        self.compare = True

        # Figure: reserve bottom strip for buttons
        self.fig = plt.figure(figsize=(12.8, 6.8))
        self.fig.canvas.manager.set_window_title("Projectile Motion Simulator (NumPy + Matplotlib)")

        # Main plot (leave room at bottom for button row)
        self.ax = self.fig.add_axes([0.06, 0.18, 0.58, 0.76])
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.grid(True, alpha=0.3)

        # Info panel
        self.ax_info = self.fig.add_axes([0.68, 0.86, 0.30, 0.10])
        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(0.0, 0.55, "", va="center", fontsize=10)

        # Right panel (sliders + radios only)
        self._build_right_panel()

        # Bottom buttons under x-axis (ALL buttons here)
        self._build_bottom_buttons()

        # Lines
        (self.line_drag,) = self.ax.plot([], [], lw=2, label="With Drag (numeric)")
        (self.line_nodrag,) = self.ax.plot([], [], lw=2, linestyle="--", label="No Drag (analytic)")
        (self.point,) = self.ax.plot([], [], marker="o", markersize=6, linestyle="None")
        self.ax.legend(loc="upper right")

        # Compute initial
        self._recompute_all()
        self._draw_static()

        # timer
        self.timer = self.fig.canvas.new_timer(interval=20)
        self.timer.add_callback(self._on_timer)

    def _build_right_panel(self):
        x0, w = 0.68, 0.30

        # --- sliders layout ---
        top = 0.82
        row_h = 0.035
        gap = 0.012
        n_sliders = 10

        def row(i, h=row_h):
            y = top - i * (h + gap)
            return [x0, y, w, h]

    # sliders
        self.sl_v0  = Slider(self.fig.add_axes(row(0)), "v0 (m/s)", 1, 200, valinit=self.v0, valstep=1)
        self.sl_th  = Slider(self.fig.add_axes(row(1)), "theta (deg)", 0, 89, valinit=self.theta, valstep=1)
        self.sl_y0  = Slider(self.fig.add_axes(row(2)), "y0 (m)", 0, 50, valinit=self.y0, valstep=0.5)
        self.sl_wx  = Slider(self.fig.add_axes(row(3)), "wind wx", -30, 30, valinit=self.wx, valstep=0.5)
        self.sl_wy  = Slider(self.fig.add_axes(row(4)), "wind wy", -10, 10, valinit=self.wy, valstep=0.5)
        self.sl_cd  = Slider(self.fig.add_axes(row(5)), "Cd", 0.05, 1.2, valinit=self.Cd, valstep=0.01)
        self.sl_d   = Slider(self.fig.add_axes(row(6)), "diameter d (m)", 0.01, 0.5, valinit=self.d, valstep=0.005)
        self.sl_rho = Slider(self.fig.add_axes(row(7)), "rho (kg/m^3)", 0.5, 1.5, valinit=self.rho, valstep=0.005)
        self.sl_k   = Slider(self.fig.add_axes(row(8)), "k_lin", 0.0, 1.5, valinit=self.k_lin, valstep=0.01)
        self.sl_dt  = Slider(self.fig.add_axes(row(9)), "dt (s)", 0.001, 0.05, valinit=self.dt, valstep=0.001)

        for sl in [self.sl_v0, self.sl_th, self.sl_y0, self.sl_wx, self.sl_wy,
               self.sl_cd, self.sl_d, self.sl_rho, self.sl_k, self.sl_dt]:
            sl.on_changed(self._on_params_changed)

    # --- compute slider bottom, then place radios BELOW it ---
        sliders_bottom = top - (n_sliders - 1) * (row_h + gap)  # y of last slider
        sliders_bottom = sliders_bottom - 0.02                  # extra breathing room under sliders

        radio_h1 = 0.17
        radio_gap = 0.03
        left_w = (w - 0.02) / 2

        radio_y0 = sliders_bottom - radio_h1  # start right below sliders

        ax_drag = self.fig.add_axes([x0, radio_y0, left_w, radio_h1])
        self.rb_drag = RadioButtons(
        ax_drag, ("None", "Linear", "Quadratic"),
        active=("None", "Linear", "Quadratic").index(self.drag_model)
    )
        ax_drag.set_title("Drag Model", fontsize=10)

        ax_solver = self.fig.add_axes([x0 + left_w + 0.02, radio_y0, left_w, radio_h1])
        self.rb_solver = RadioButtons(
        ax_solver, ("Euler", "RK4"),
        active=("Euler", "RK4").index(self.solver)
    )
        ax_solver.set_title("Solver", fontsize=10)

    # shape block below drag/solver
        shape_h = 0.16
        shape_y = radio_y0 - radio_gap - shape_h

        ax_shape = self.fig.add_axes([x0, shape_y, w, shape_h])
        self.rb_shape = RadioButtons(
        ax_shape, ("Sphere", "Cylinder", "Bullet (Ogive)", "Custom"),
        active=("Sphere", "Cylinder", "Bullet (Ogive)", "Custom").index(self.shape_name)
    )
        ax_shape.set_title("Shape Preset", fontsize=10)

        self.rb_drag.on_clicked(self._on_radio_changed)
        self.rb_solver.on_clicked(self._on_radio_changed)
        self.rb_shape.on_clicked(self._on_shape_changed)


    def _build_bottom_buttons(self):
        # place all controls under x-axis (below main plot)
        # main plot is [0.06, 0.18, 0.58, 0.76]
        # bottom strip between y=0.06..0.15 is safe
        y = 0.07
        h = 0.08
        x_left = 0.06
        total_w = 0.58

        # 4 blocks: Reset, Play, Update, Compare(checkbox)
        bw = 0.12
        gap = 0.02

        ax_reset = self.fig.add_axes([x_left, y, bw, h])
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_reset.on_clicked(self._on_reset)

        ax_play = self.fig.add_axes([x_left + (bw + gap), y, bw, h])
        self.btn_play = Button(ax_play, "Play")
        self.btn_play.on_clicked(self._on_play_pause)

        ax_update = self.fig.add_axes([x_left + 2 * (bw + gap), y, bw, h])
        self.btn_update = Button(ax_update, "Update")
        self.btn_update.on_clicked(self._on_force_update)

        # Compare checkbox (wider)
        ax_cmp = self.fig.add_axes([x_left + 3 * (bw + gap), y, 0.18, h])
        self.cb = CheckButtons(ax_cmp, ["Compare"], [self.compare])
        self.cb.on_clicked(self._on_check_changed)

    def _collect_params(self) -> dict:
        return {
            "m": self.m,
            "g": self.g,
            "drag_model": self.drag_model,
            "k_lin": self.k_lin,
            "rho": self.rho,
            "Cd": self.Cd,
            "A": area_from_diameter(self.d),
            "wx": self.wx,
            "wy": self.wy,
            "dt": self.dt,
            "tmax": self.tmax,
            "solver": self.solver,
        }

    def _recompute_all(self):
        params = self._collect_params()
        self.t_num, self.S_num = simulate_numeric(self.v0, self.theta, self.y0, params)
        self.metrics_num = summary_metrics(self.t_num, self.S_num)

        t_end = self.t_num[-1] if self.t_num.size > 1 else 0.0
        t_ref = np.linspace(0.0, max(t_end, 1e-6), 500)
        S_ana = analytic_no_drag(self.v0, self.theta, self.y0, self.g, t_ref)

        # clip analytic to ground
        y = S_ana[:, 1]
        hit = np.where(y < 0.0)[0]
        if hit.size > 0:
            i = int(hit[0])
            if i > 0:
                y1, y2 = y[i - 1], y[i]
                if (y1 - y2) != 0:
                    alpha = y1 / (y1 - y2)
                    S_ana[i] = S_ana[i - 1] + alpha * (S_ana[i] - S_ana[i - 1])
                    S_ana[i, 1] = 0.0
                S_ana = S_ana[: i + 1]
            else:
                S_ana = S_ana[:1]
        self.S_ana = S_ana

        self.anim_i = 0
        self._update_info_text()

    def _update_info_text(self):
        A = area_from_diameter(self.d)
        txt = (
            f"Model: {self.drag_model} | Solver: {self.solver}\n"
            f"Shape: {self.shape_name} (Cd={self.Cd:.2f}, d={self.d:.3f} m, A={A:.4f} m²)\n"
            f"Wind: wx={self.wx:.1f}, wy={self.wy:.1f} | dt={self.dt:.3f}\n"
            f"Range={self.metrics_num['range']:.2f} m,  T={self.metrics_num['flight_time']:.2f} s,  "
            f"Hmax={self.metrics_num['max_height']:.2f} m"
        )
        self.info_text.set_text(txt)

    def _autoscale_axes(self):
        xmax = float(np.max(self.S_num[:, 0])) if self.S_num.size else 1.0
        ymax = float(np.max(self.S_num[:, 1])) if self.S_num.size else 1.0
        if self.compare and self.S_ana is not None and self.S_ana.size:
            xmax = max(xmax, float(np.max(self.S_ana[:, 0])))
            ymax = max(ymax, float(np.max(self.S_ana[:, 1])))
        self.ax.set_xlim(0, max(1.0, xmax * 1.05))
        self.ax.set_ylim(0, max(1.0, ymax * 1.10))

    def _draw_static(self):
        self._autoscale_axes()
        self.line_drag.set_data(self.S_num[:, 0], self.S_num[:, 1])
        if self.compare:
            self.line_nodrag.set_visible(True)
            self.line_nodrag.set_data(self.S_ana[:, 0], self.S_ana[:, 1])
        else:
            self.line_nodrag.set_visible(False)
        self.point.set_data([self.S_num[0, 0]], [self.S_num[0, 1]])
        self.fig.canvas.draw_idle()

    def _draw_anim_frame(self):
        i = max(1, min(self.anim_i, self.S_num.shape[0]))
        self.line_drag.set_data(self.S_num[:i, 0], self.S_num[:i, 1])
        self.point.set_data([self.S_num[i - 1, 0]], [self.S_num[i - 1, 1]])
        if self.compare:
            self.line_nodrag.set_visible(True)
            self.line_nodrag.set_data(self.S_ana[:, 0], self.S_ana[:, 1])
        else:
            self.line_nodrag.set_visible(False)
        self.fig.canvas.draw_idle()

    # --------- callbacks ----------
    def _on_params_changed(self, _):
        self.v0 = float(self.sl_v0.val)
        self.theta = float(self.sl_th.val)
        self.y0 = float(self.sl_y0.val)
        self.wx = float(self.sl_wx.val)
        self.wy = float(self.sl_wy.val)
        self.Cd = float(self.sl_cd.val)
        self.d = float(self.sl_d.val)
        self.rho = float(self.sl_rho.val)
        self.k_lin = float(self.sl_k.val)
        self.dt = float(self.sl_dt.val)

        if not self.is_animating:
            self._recompute_all()
            self._draw_static()

    def _on_radio_changed(self, _):
        self.drag_model = self.rb_drag.value_selected
        self.solver = self.rb_solver.value_selected
        if not self.is_animating:
            self._recompute_all()
            self._draw_static()

    def _on_shape_changed(self, label):
        self.shape_name = label
        if label != "Custom":
            self.sl_cd.set_val(SHAPES[label]["Cd"])
            self.sl_d.set_val(SHAPES[label]["d"])
        if not self.is_animating:
            self._recompute_all()
            self._draw_static()

    def _on_check_changed(self, _):
        self.compare = bool(self.cb.get_status()[0])
        if not self.is_animating:
            self._draw_static()

    def _on_force_update(self, _):
        if not self.is_animating:
            self._recompute_all()
            self._draw_static()

    def _on_reset(self, _):
        self.is_animating = False
        self.btn_play.label.set_text("Play")

        # reset values
        self.v0 = 50.0
        self.theta = 45.0
        self.y0 = 0.0
        self.wx = 0.0
        self.wy = 0.0
        self.rho = 1.225
        self.k_lin = 0.15
        self.dt = 0.01
        self.drag_model = "Quadratic"
        self.solver = "RK4"
        self.compare = True
        self.shape_name = "Bullet (Ogive)"
        self.Cd = SHAPES[self.shape_name]["Cd"]
        self.d = SHAPES[self.shape_name]["d"]

        # push to widgets
        self.sl_v0.set_val(self.v0)
        self.sl_th.set_val(self.theta)
        self.sl_y0.set_val(self.y0)
        self.sl_wx.set_val(self.wx)
        self.sl_wy.set_val(self.wy)
        self.sl_rho.set_val(self.rho)
        self.sl_k.set_val(self.k_lin)
        self.sl_dt.set_val(self.dt)
        self.sl_cd.set_val(self.Cd)
        self.sl_d.set_val(self.d)

        self.rb_drag.set_active(("None", "Linear", "Quadratic").index(self.drag_model))
        self.rb_solver.set_active(("Euler", "RK4").index(self.solver))
        self.rb_shape.set_active(("Sphere", "Cylinder", "Bullet (Ogive)", "Custom").index(self.shape_name))

        # compare checkbox
        if self.cb.get_status()[0] != self.compare:
            self.cb.set_active(0)

        self._recompute_all()
        self._draw_static()

    def _on_play_pause(self, _):
        self.is_animating = not self.is_animating
        self.btn_play.label.set_text("Pause" if self.is_animating else "Play")

        if self.is_animating:
            self._recompute_all()
            self._autoscale_axes()
            self.anim_i = 0
            self.timer.start()
        else:
            self.timer.stop()
            self._draw_static()

    def _on_timer(self):
        if not self.is_animating:
            return
        step = max(1, int(0.02 / max(self.dt, 1e-6)))
        self.anim_i += step
        if self.anim_i >= self.S_num.shape[0]:
            self.anim_i = self.S_num.shape[0]
            self.is_animating = False
            self.btn_play.label.set_text("Play")
            self.timer.stop()
            self._draw_static()
            return
        self._draw_anim_frame()


def main():
    ProjectileApp()
    plt.show()


if __name__ == "__main__":
    main()

