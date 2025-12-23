"""
Projectile Motion Simulator (NumPy + Matplotlib) — Full Script (Re-layout fixed)

✅ Features (as requested):
- 2D projectile motion under gravity
- Air resistance (drag): None / Linear / Quadratic
- Wind field (constant wind vector wx, wy) via relative velocity v_rel = v - w
- Projectile "shape" presets (Sphere/Cylinder/Bullet/Custom) controlling Cd and diameter -> area A
- Good control panel (sliders + radio buttons + checkbox + buttons), with stable layout (no overlap)
- Compare mode: show No-Drag analytic vs Drag numeric
- Play/Pause animation

Dependencies:
    numpy, matplotlib

Run:
    python projectile_simulator.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons


# -----------------------------
# Physics / Models (NumPy core)
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
    """Return [x,y,vx,vy] over time for no drag (analytic)."""
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
    d/dt [x, y, vx, vy] with gravity + drag + wind.
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
            # Fd = -k * v_rel
            a_drag = -(k_lin / m) * v_rel
            ax += a_drag[0]
            ay += a_drag[1]

        elif model == "Quadratic":
            # Fd = -0.5*rho*Cd*A*|v_rel|*v_rel
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
    Simulate until y hits ground (y<=0) or tmax. Returns (t, states).
    states columns: [x,y,vx,vy]
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
    steps = int(np.ceil(tmax / dt))
    for _ in range(steps):
        t += dt
        state = stepper(state, dt, params)
        times.append(t)
        states.append(state.copy())

        if state[1] <= 0.0 and t > 0.0:
            break

    times = np.array(times, dtype=float)
    states = np.vstack(states)

    # Interpolate last point to y=0 for cleaner "landing"
    if states.shape[0] >= 2 and states[-1, 1] < 0.0:
        y1 = states[-2, 1]
        y2 = states[-1, 1]
        if (y1 - y2) != 0:
            alpha = y1 / (y1 - y2)
            states[-1] = states[-2] + alpha * (states[-1] - states[-2])
            times[-1] = times[-2] + alpha * (times[-1] - times[-2])
            states[-1, 1] = 0.0

    return times, states


def summary_metrics(t: np.ndarray, states: np.ndarray) -> dict:
    x = states[:, 0]
    y = states[:, 1]
    vx = states[:, 2]
    vy = states[:, 3]
    speed = np.sqrt(vx * vx + vy * vy)
    idx_max = int(np.argmax(y))
    return {
        "range": float(x[-1]),
        "flight_time": float(t[-1]),
        "max_height": float(y[idx_max]),
        "t_at_max": float(t[idx_max]),
        "v_impact": float(speed[-1]),
    }


# -----------------------------
# App UI + Visualization
# -----------------------------
class ProjectileApp:
    def __init__(self):
        # ----- state -----
        self.is_animating = False
        self.anim_i = 0

        # initial params
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

        # ----- figure layout (fixed panel, no overlap) -----
        self.fig = plt.figure(figsize=(12.8, 6.2))
        self.fig.canvas.manager.set_window_title("Projectile Motion Simulator (NumPy + Matplotlib)")

        # main plot
        self.ax = self.fig.add_axes([0.06, 0.12, 0.58, 0.82])
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.grid(True, alpha=0.3)

        # info panel (top-right)
        self.ax_info = self.fig.add_axes([0.68, 0.86, 0.30, 0.10])
        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(0.0, 0.55, "", va="center", fontsize=10)

        # build widgets (re-layout)
        self._build_widgets()

        # lines
        (self.line_drag,) = self.ax.plot([], [], lw=2, label="With Drag (numeric)")
        (self.line_nodrag,) = self.ax.plot([], [], lw=2, linestyle="--", label="No Drag (analytic)")
        (self.point,) = self.ax.plot([], [], marker="o", markersize=6, linestyle="None")

        self.ax.legend(loc="upper right")

        # compute & draw
        self._recompute_all()
        self._draw_static()

        # timer for animation
        self.timer = self.fig.canvas.new_timer(interval=20)
        self.timer.add_callback(self._on_timer)

    # ---------- layout-fixed widgets ----------
    def _build_widgets(self):
        x0 = 0.68
        w = 0.30
        top = 0.82
        row_h = 0.035
        gap = 0.012

        def row(i, h=row_h):
            y = top - i * (h + gap)
            return [x0, y, w, h]

        # Sliders
        ax_v0 = self.fig.add_axes(row(0))
        self.sl_v0 = Slider(ax_v0, "v0 (m/s)", 1.0, 200.0, valinit=self.v0, valstep=1.0)

        ax_th = self.fig.add_axes(row(1))
        self.sl_th = Slider(ax_th, "theta (deg)", 0.0, 89.0, valinit=self.theta, valstep=1.0)

        ax_y0 = self.fig.add_axes(row(2))
        self.sl_y0 = Slider(ax_y0, "y0 (m)", 0.0, 50.0, valinit=self.y0, valstep=0.5)

        ax_wx = self.fig.add_axes(row(3))
        self.sl_wx = Slider(ax_wx, "wind wx", -30.0, 30.0, valinit=self.wx, valstep=0.5)

        ax_wy = self.fig.add_axes(row(4))
        self.sl_wy = Slider(ax_wy, "wind wy", -10.0, 10.0, valinit=self.wy, valstep=0.5)

        ax_cd = self.fig.add_axes(row(5))
        self.sl_cd = Slider(ax_cd, "Cd", 0.05, 1.20, valinit=self.Cd, valstep=0.01)

        ax_d = self.fig.add_axes(row(6))
        self.sl_d = Slider(ax_d, "diameter d (m)", 0.01, 0.50, valinit=self.d, valstep=0.005)

        ax_rho = self.fig.add_axes(row(7))
        self.sl_rho = Slider(ax_rho, "rho (kg/m^3)", 0.5, 1.5, valinit=self.rho, valstep=0.005)

        ax_k = self.fig.add_axes(row(8))
        self.sl_k = Slider(ax_k, "k_lin", 0.0, 1.5, valinit=self.k_lin, valstep=0.01)

        ax_dt = self.fig.add_axes(row(9))
        self.sl_dt = Slider(ax_dt, "dt (s)", 0.001, 0.05, valinit=self.dt, valstep=0.001)

        # Buttons + compare checkbox (one row)
        btn_y = top - 10 * (row_h + gap) - 0.015
        btn_h = 0.05
        chk_w = 0.05
        btn_w = (w - 0.06) / 3.0

        ax_chk = self.fig.add_axes([x0, btn_y, chk_w, btn_h])
        self.cb = CheckButtons(ax_chk, ["cmp"], [self.compare])

        ax_btn_reset = self.fig.add_axes([x0 + chk_w + 0.01, btn_y, btn_w, btn_h])
        self.btn_reset = Button(ax_btn_reset, "Reset")

        ax_btn_play = self.fig.add_axes([x0 + chk_w + 0.01 + btn_w + 0.01, btn_y, btn_w, btn_h])
        self.btn_play = Button(ax_btn_play, "Play")

        ax_btn_update = self.fig.add_axes([x0 + chk_w + 0.01 + 2 * (btn_w + 0.01), btn_y, btn_w, btn_h])
        self.btn_update = Button(ax_btn_update, "Update")

        # Radio groups (bottom)
        radio_y0 = 0.30
        radio_h1 = 0.18
        radio_h2 = 0.16
        left_w = (w - 0.02) / 2.0

        ax_drag = self.fig.add_axes([x0, radio_y0, left_w, radio_h1])
        self.rb_drag = RadioButtons(
            ax_drag,
            ("None", "Linear", "Quadratic"),
            active=("None", "Linear", "Quadratic").index(self.drag_model),
        )
        ax_drag.set_title("Drag Model", fontsize=10)

        ax_solver = self.fig.add_axes([x0 + left_w + 0.02, radio_y0, left_w, radio_h1])
        self.rb_solver = RadioButtons(
            ax_solver,
            ("Euler", "RK4"),
            active=("Euler", "RK4").index(self.solver),
        )
        ax_solver.set_title("Solver", fontsize=10)

        ax_shape = self.fig.add_axes([x0, 0.08, w, radio_h2])
        self.rb_shape = RadioButtons(
            ax_shape,
            ("Sphere", "Cylinder", "Bullet (Ogive)", "Custom"),
            active=("Sphere", "Cylinder", "Bullet (Ogive)", "Custom").index(self.shape_name),
        )
        ax_shape.set_title("Shape Preset", fontsize=10)

        # Hook events
        for sl in [
            self.sl_v0, self.sl_th, self.sl_y0, self.sl_wx, self.sl_wy,
            self.sl_cd, self.sl_d, self.sl_rho, self.sl_k, self.sl_dt
        ]:
            sl.on_changed(self._on_params_changed)

        self.rb_drag.on_clicked(self._on_radio_changed)
        self.rb_solver.on_clicked(self._on_radio_changed)
        self.rb_shape.on_clicked(self._on_shape_changed)
        self.cb.on_clicked(self._on_check_changed)

        self.btn_reset.on_clicked(self._on_reset)
        self.btn_play.on_clicked(self._on_play_pause)
        self.btn_update.on_clicked(self._on_force_update)

    # ---------- helpers ----------
    def _collect_params(self) -> dict:
        A = area_from_diameter(self.d)
        return {
            "m": self.m,
            "g": self.g,
            "drag_model": self.drag_model,
            "k_lin": self.k_lin,
            "rho": self.rho,
            "Cd": self.Cd,
            "A": A,
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

        # analytic no-drag for compare
        t_end = self.t_num[-1] if self.t_num.size > 1 else 0.0
        t_ref = np.linspace(0.0, max(t_end, 1e-6), 500)
        S_ana = analytic_no_drag(self.v0, self.theta, self.y0, self.g, t_ref)

        # clip to ground
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
        x_num = self.S_num[:, 0]
        y_num = self.S_num[:, 1]
        xmax = float(np.max(x_num)) if x_num.size else 1.0
        ymax = float(np.max(y_num)) if y_num.size else 1.0

        if self.compare and self.S_ana is not None and self.S_ana.size:
            xmax = max(xmax, float(np.max(self.S_ana[:, 0])))
            ymax = max(ymax, float(np.max(self.S_ana[:, 1])))

        xmax = max(xmax, 1.0)
        ymax = max(ymax, 1.0)
        self.ax.set_xlim(0.0, xmax * 1.05)
        self.ax.set_ylim(0.0, ymax * 1.10)

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

    # ---------- event handlers ----------
    def _on_params_changed(self, _val):
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

    def _on_radio_changed(self, _label):
        self.drag_model = self.rb_drag.value_selected
        self.solver = self.rb_solver.value_selected
        if not self.is_animating:
            self._recompute_all()
            self._draw_static()

    def _on_shape_changed(self, label):
        self.shape_name = label
        preset = SHAPES[label]
        if label != "Custom":
            # update sliders (these calls trigger callbacks; fine when not animating)
            self.sl_cd.set_val(preset["Cd"])
            self.sl_d.set_val(preset["d"])
        if not self.is_animating:
            self._recompute_all()
            self._draw_static()

    def _on_check_changed(self, _label):
        self.compare = bool(self.cb.get_status()[0])
        if not self.is_animating:
            self._draw_static()

    def _on_force_update(self, _event):
        if not self.is_animating:
            self._recompute_all()
            self._draw_static()

    def _on_reset(self, _event):
        self.is_animating = False
        self.btn_play.label.set_text("Play")

        # Reset params
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

        # Push to widgets
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

        # ensure compare checked
        if self.cb.get_status()[0] != self.compare:
            self.cb.set_active(0)

        self._recompute_all()
        self._draw_static()

    def _on_play_pause(self, _event):
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

        # advance frame
        # (roughly scale speed vs dt so it doesn't crawl when dt small)
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
