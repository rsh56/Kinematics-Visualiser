"""
3D Kinematics Visualiser
========================
A professional engineering tool demonstrating four clean design layers:

  1. KinematicsModel    - Physics engine (fully vectorised NumPy, zero for-loops)
  2. TrajectoryAnalyser - Analytical metrics and drag-free validation reference
  3. Visualiser3D       - 3-D matplotlib canvas with quivers and plane projections
  4. GUIController      - Tkinter GUI: text entry boxes, live stats, NaN-separator capping

Run:
    python kinematics_visualiser.py
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional


# =============================================================================
# 1.  PHYSICS ENGINE
# =============================================================================

class KinematicsModel:
    """
    Vectorised 3-D projectile under gravity with optional linear drag.

    The full trajectory is resolved in one NumPy broadcast - no Python
    for-loops appear anywhere in the computation.  Velocity and acceleration
    are derived automatically from position data via np.gradient().

    Parameters
    ----------
    t    : 1-D time array (s)
    r0   : initial position  [x0, y0, z0]  (m)
    v0   : initial velocity  [vx0, vy0, vz0]  (m/s)
    g    : gravitational acceleration  (m/s^2)
    drag : linear drag coefficient  k  (kg/s)  - 0 = drag-free
    mass : particle mass  (kg)
    """

    def __init__(
        self,
        t:    np.ndarray,
        r0:   np.ndarray,
        v0:   np.ndarray,
        g:    float = 9.81,
        drag: float = 0.0,
        mass: float = 1.0,
    ) -> None:
        self.t    = np.asarray(t,  dtype=float)
        self.r0   = np.asarray(r0, dtype=float)
        self.v0   = np.asarray(v0, dtype=float)
        self.g    = float(g)
        self.drag = float(drag)
        self.mass = float(mass)

        self._pos: Optional[np.ndarray] = None

    def compute(self) -> None:
        """Populate the position cache with a fully vectorised calculation."""
        t               = self.t
        x0, y0, z0      = self.r0
        vx0, vy0, vz0   = self.v0

        if self.drag == 0.0:
            x = x0 + vx0 * t
            y = y0 + vy0 * t
            z = z0 + vz0 * t - 0.5 * self.g * t ** 2
        else:
            km  = self.drag / self.mass
            e   = np.exp(-km * t)
            inv = (1.0 - e) / km

            x = x0 + vx0 * inv
            y = y0 + vy0 * inv

            v_term = -self.mass * self.g / self.drag
            z = z0 + v_term * t + (vz0 - v_term) * inv

        self._pos = np.vstack((x, y, z))

    @property
    def position(self) -> np.ndarray:
        """(3, N) position array [x; y; z]."""
        if self._pos is None:
            self.compute()
        return self._pos

    @property
    def velocity(self) -> np.ndarray:
        """(3, N) velocity array derived from position via np.gradient."""
        p = self.position
        return np.vstack([np.gradient(p[i], self.t) for i in range(3)])

    @property
    def acceleration(self) -> np.ndarray:
        """(3, N) acceleration array derived from velocity via np.gradient."""
        v = self.velocity
        return np.vstack([np.gradient(v[i], self.t) for i in range(3)])

    def invalidate(self) -> None:
        """Clear the cached result so the next property access recomputes."""
        self._pos = None


# =============================================================================
# 2.  ANALYTICAL LAYER
# =============================================================================

class TrajectoryAnalyser:
    """
    Transforms raw KinematicsModel arrays into engineering metrics.

    Responsibilities
    ----------------
    - arc length and instantaneous speed
    - ground-hit time detection (z = 0 crossing)
    - axis-aligned bounding-box containment mask
    - drag-free analytical reference values for validation
    """

    def __init__(self, model: KinematicsModel) -> None:
        self.model = model

    def arc_length(self) -> float:
        """Total path length along the 3-D trajectory (m)."""
        diff = np.diff(self.model.position, axis=1)
        return float(np.sum(np.linalg.norm(diff, axis=0)))

    def speed(self) -> np.ndarray:
        """Instantaneous scalar speed at every time step (m/s), shape (N,)."""
        return np.linalg.norm(self.model.velocity, axis=0)

    def ground_hit_time(self) -> Optional[float]:
        """
        Linearly-interpolated time at which z first crosses from >= 0 to < 0.
        Returns None if the particle does not reach z = 0 within the window.
        """
        z = self.model.position[2]
        t = self.model.t
        crossings = np.where((z[:-1] >= 0.0) & (z[1:] < 0.0))[0]
        if len(crossings) == 0:
            return None
        i    = crossings[0]
        frac = z[i] / (z[i] - z[i + 1])
        return float(t[i] + frac * (t[i + 1] - t[i]))

    def in_bounds(
        self,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
        z_lim: tuple[float, float],
    ) -> np.ndarray:
        """Boolean mask of shape (N,) - True where the particle is inside the AABB."""
        x, y, z = self.model.position
        return (
            (x >= x_lim[0]) & (x <= x_lim[1]) &
            (y >= y_lim[0]) & (y <= y_lim[1]) &
            (z >= z_lim[0]) & (z <= z_lim[1])
        )

    def validation_data(self) -> dict:
        """
        Analytical reference for a drag-free parabola with the same initial
        conditions. Used to cross-check the numerical trajectory.

        Returns a dict with keys:
            max_height_analytical   (m)
            flight_time_analytical  (s)
            range_analytical        (m)
            arc_length_numerical    (m)
        """
        m               = self.model
        vx0, vy0, vz0   = m.v0
        _,   _,   z0    = m.r0
        g               = m.g

        v_horiz  = np.hypot(vx0, vy0)
        t_peak   = vz0 / g if g > 0 else np.inf
        z_max    = z0 + vz0 * t_peak - 0.5 * g * t_peak ** 2

        disc = vz0 ** 2 + 2.0 * g * z0
        if disc < 0.0 or g == 0.0:
            t_flight = np.nan
            h_range  = np.nan
        else:
            t_flight = (vz0 + np.sqrt(disc)) / g
            h_range  = v_horiz * t_flight

        return {
            "max_height_analytical":  z_max,
            "flight_time_analytical": t_flight,
            "range_analytical":       h_range,
            "arc_length_numerical":   self.arc_length(),
        }


# =============================================================================
# 3.  VISUALISATION LAYER
# =============================================================================

class Visualiser3D:
    """
    Renders the 3-D trajectory onto a matplotlib Figure using mplot3d.

    Features
    --------
    - Main 3-D line with start / end markers
    - Velocity quivers (normalised arrows) sampled at regular stride intervals
    - Shadow projections onto the XY (floor), XZ (back-Y), and YZ (back-X) planes
    - NaN-separator trick: a NaN column is inserted at the first z < 0 crossing
      so matplotlib draws a clean above-ground line even at high point counts
    - Hard cap on rendered points to keep the GUI responsive
    """

    _SHADOW_ALPHA = 0.20

    def __init__(
        self,
        model: KinematicsModel,
        fig:   Optional[plt.Figure] = None,
    ) -> None:
        self.model = model
        self.fig   = fig if fig is not None else plt.figure(figsize=(10, 7))
        self.ax:   Optional[Axes3D] = None

    def draw(
        self,
        show_vectors:     bool = True,
        vector_stride:    int  = 10,
        show_projections: bool = True,
        cap_points:       int  = 2000,
    ) -> None:
        """Clear the figure and render the complete visualisation."""
        self.fig.clear()
        ax: Axes3D = self.fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3-D Particle Trajectory", pad=12)
        self.ax = ax

        pos_raw  = self.model.position
        pos_plot = self._prepare_plot_data(pos_raw, cap=cap_points)
        x, y, z  = pos_plot

        ax.plot(x, y, z, color="royalblue", linewidth=1.8, label="Trajectory")

        valid = ~np.isnan(x)
        if valid.any():
            i0 = int(np.argmax(valid))
            i1 = int(len(x) - 1 - np.argmax(valid[::-1]))
            ax.scatter(x[i0], y[i0], z[i0],
                       color="limegreen", s=70, depthshade=False, label="Start")
            ax.scatter(x[i1], y[i1], z[i1],
                       color="crimson",   s=70, depthshade=False, label="End")

        if show_projections:
            self._draw_projections(ax, pos_plot)

        if show_vectors:
            self._draw_velocity_quivers(ax, pos_raw, cap_points, vector_stride)

        ax.legend(fontsize=8, loc="upper right")
        self.fig.tight_layout()

    @staticmethod
    def _prepare_plot_data(pos: np.ndarray, cap: int = 2000) -> np.ndarray:
        """
        Downsample to at most cap points, then insert a NaN column after the
        first z < 0 crossing so matplotlib stops drawing underground.
        """
        N = pos.shape[1]
        if N > cap:
            idx = np.round(np.linspace(0, N - 1, cap)).astype(int)
            pos = pos[:, idx]

        z     = pos[2]
        below = np.where(z < 0.0)[0]
        if len(below) == 0:
            return pos

        cut     = below[0]
        nan_col = np.full((3, 1), np.nan)
        return np.concatenate([pos[:, :cut], nan_col], axis=1)

    def _draw_projections(self, ax: Axes3D, pos: np.ndarray) -> None:
        """Draw dashed shadow lines on the three coordinate planes."""
        x, y, z = pos
        mask     = ~np.isnan(x)
        xv, yv, zv = x[mask], y[mask], z[mask]
        if xv.size == 0:
            return

        z_floor = min(float(zv.min()), 0.0) - 1.0
        x_back  = float(xv.max()) + 1.0
        y_back  = float(yv.max()) + 1.0

        kw = dict(color="slategray", alpha=self._SHADOW_ALPHA,
                  linewidth=1.0, linestyle="--")

        ax.plot(xv, yv, z_floor * np.ones_like(xv), **kw)
        ax.plot(xv, y_back * np.ones_like(xv), zv,  **kw)
        ax.plot(x_back * np.ones_like(yv), yv, zv,  **kw)

    def _draw_velocity_quivers(
        self,
        ax:       Axes3D,
        pos_raw:  np.ndarray,
        cap:      int,
        stride:   int,
    ) -> None:
        """Plot normalised velocity arrows at every stride-th sampled point."""
        N     = pos_raw.shape[1]
        n_pts = min(N, cap)
        idx   = np.round(np.linspace(0, N - 1, n_pts)).astype(int)

        above  = pos_raw[2, idx] >= 0.0
        sample = idx[above][::stride]
        if len(sample) == 0:
            return

        vel         = self.model.velocity
        px, py, pz  = pos_raw[:, sample]
        vx, vy, vz  = vel[:, sample]

        mag   = np.linalg.norm(np.vstack([vx, vy, vz]), axis=0)
        mag   = np.where(mag == 0.0, 1.0, mag)

        span  = float(np.ptp(pos_raw, axis=1).max())
        scale = 0.05 * span if span > 0 else 1.0

        ax.quiver(
            px, py, pz,
            (vx / mag) * scale, (vy / mag) * scale, (vz / mag) * scale,
            color="darkorange", arrow_length_ratio=0.35,
            linewidth=0.9, label="Velocity (direction)",
        )


# =============================================================================
# 4.  INTERACTIVE CONTROLLER
# =============================================================================

class GUIController:
    """
    Tkinter front-end for the 3-D Kinematics Visualiser.

    Layout
    ------
    Left panel  : labelled text-entry boxes for every initial-condition parameter
                  (type any value and press Enter or click Run Simulation).
    Right panel : embedded matplotlib canvas showing the live 3-D plot.
    Bottom left : numerical statistics readout (arc length, range, etc.).

    The "Run Simulation" button (and pressing Enter in any field) rebuilds the
    KinematicsModel, re-runs TrajectoryAnalyser, redraws through Visualiser3D,
    and refreshes the canvas - all within a single event-handler call.
    """

    # (display label, dict key, default value)
    _PARAMS: list[tuple] = [
        ("x0  (m)",      "x0",    0.0  ),
        ("y0  (m)",      "y0",    0.0  ),
        ("z0  (m)",      "z0",    0.0  ),
        ("vx0 (m/s)",    "vx0",  20.0  ),
        ("vy0 (m/s)",    "vy0",  10.0  ),
        ("vz0 (m/s)",    "vz0",  30.0  ),
        ("g   (m/s^2)",  "g",     9.81 ),
        ("drag (kg/s)",  "drag",  0.0  ),
        ("mass (kg)",    "mass",  1.0  ),
        ("t_max (s)",    "t_max", 7.0  ),
        ("N points",     "n_pts", 500.0),
    ]

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("3D Kinematics Visualiser")
        self.root.minsize(1100, 600)

        self._vars:    dict[str, tk.StringVar]      = {}
        self._fig:     Optional[plt.Figure]         = None
        self._canvas:  Optional[FigureCanvasTkAgg]  = None
        self._show_vectors = tk.BooleanVar(value=True)
        self._show_proj    = tk.BooleanVar(value=True)

        self._build_layout()
        self._run()

    def _build_layout(self) -> None:
        ctrl = ttk.LabelFrame(self.root, text="Simulation Parameters", padding=10)
        ctrl.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        for row_i, (label, key, default) in enumerate(self._PARAMS):
            var = tk.StringVar(value=str(default))
            self._vars[key] = var

            ttk.Label(ctrl, text=label, width=14, anchor="w").grid(
                row=row_i, column=0, sticky="w", pady=4)

            entry = ttk.Entry(ctrl, textvariable=var, width=14, justify="right")
            entry.grid(row=row_i, column=1, padx=6, sticky="ew")
            entry.bind("<Return>", lambda _e: self._run())

        n = len(self._PARAMS)
        tog = ttk.Frame(ctrl)
        tog.grid(row=n, column=0, columnspan=3, pady=(10, 2), sticky="w")
        ttk.Checkbutton(tog, text="Velocity vectors",
                        variable=self._show_vectors).pack(side="left", padx=4)
        ttk.Checkbutton(tog, text="Plane projections",
                        variable=self._show_proj).pack(side="left", padx=4)

        ttk.Button(
            ctrl, text="Run Simulation", command=self._run,
        ).grid(row=n + 1, column=0, columnspan=3, pady=(8, 4), sticky="ew")

        self._stats_var = tk.StringVar(value="")
        ttk.Label(
            ctrl, textvariable=self._stats_var,
            justify="left", foreground="#1a4480", wraplength=280,
            font=("Courier", 9),
        ).grid(row=n + 2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        plot_frame = ttk.LabelFrame(self.root, text="3-D Plot", padding=4)
        plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self._fig    = plt.Figure(figsize=(9, 6.5))
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _get(self, key: str) -> float:
        """Parse a StringVar entry to float, raising ValueError on bad input."""
        return float(self._vars[key].get().strip())

    def _run(self) -> None:
        try:
            r0    = np.array([self._get("x0"),  self._get("y0"),  self._get("z0")])
            v0    = np.array([self._get("vx0"), self._get("vy0"), self._get("vz0")])
            g     = max(self._get("g"),     1e-9)
            drag  = max(self._get("drag"),  0.0)
            mass  = max(self._get("mass"),  1e-3)
            t_max = max(self._get("t_max"), 0.1)
            n_pts = int(max(self._get("n_pts"), 10))
        except ValueError as exc:
            messagebox.showerror("Invalid Input", f"Please enter a valid number.\n\n{exc}")
            return

        try:
            t        = np.linspace(0.0, t_max, n_pts)
            model    = KinematicsModel(t, r0, v0, g=g, drag=drag, mass=mass)
            analyser = TrajectoryAnalyser(model)
            vis      = Visualiser3D(model, fig=self._fig)

            vis.draw(
                show_vectors     = self._show_vectors.get(),
                vector_stride    = max(n_pts // 50, 1),
                show_projections = self._show_proj.get(),
                cap_points       = 2000,
            )
            self._canvas.draw()

            data  = analyser.validation_data()
            t_hit = analyser.ground_hit_time()
            sp    = analyser.speed()

            t_hit_str = f"{t_hit:.2f} s" if t_hit is not None else "not in window"

            lines = [
                f"Arc length  : {data['arc_length_numerical']:.2f} m",
                f"Max height  : {data['max_height_analytical']:.2f} m  (analytical)",
                f"Flight time : {data['flight_time_analytical']:.2f} s  (analytical)",
                f"Range       : {data['range_analytical']:.2f} m  (analytical)",
                f"Ground hit  : {t_hit_str}",
                f"Max speed   : {sp.max():.2f} m/s",
                f"Min speed   : {sp.min():.2f} m/s",
            ]
            self._stats_var.set("\n".join(lines))

        except Exception as exc:
            messagebox.showerror("Simulation Error", str(exc))

    def run(self) -> None:
        """Enter the Tkinter main-loop (blocking)."""
        self.root.mainloop()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    GUIController().run()
