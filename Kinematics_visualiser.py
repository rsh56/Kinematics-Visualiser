"""
#rsh56
3D Kinematics Visualiser
========================
A professional engineering tool demonstrating four clean design layers:

  1. KinematicsModel    – Physics engine (fully vectorised NumPy, zero for-loops)
  2. TrajectoryAnalyser – Analytical metrics and drag-free validation reference
  3. Visualiser3D       – 3-D matplotlib canvas with quivers and plane projections
  4. GUIController      – Tkinter GUI: sliders, live stats, NaN-separator capping

Run:
    python kinematics_visualiser.py
"""
from __future__ import annotations          # PEP 563 – lazy annotation evaluation

import numpy as np

import matplotlib
matplotlib.use("TkAgg")                     # must be set before pyplot is imported
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401 — registers "3d" projection

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from typing import Optional


# =============================================================================
# 1.  PHYSICS ENGINE
# =============================================================================

class KinematicsModel:
    """
    Vectorised 3-D projectile under gravity with optional linear drag.

    The full trajectory is resolved in one NumPy broadcast – no Python
    for-loops appear anywhere in the computation.  Velocity and acceleration
    are derived automatically from position data via np.gradient().

    Parameters
    ----------
    t    : 1-D time array (s)
    r0   : initial position  [x0, y0, z0]  (m)
    v0   : initial velocity  [vx0, vy0, vz0]  (m/s)
    g    : gravitational acceleration  (m s⁻²)
    drag : linear drag coefficient  k  (kg s⁻¹)  – 0 = drag-free
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

        self._pos: Optional[np.ndarray] = None   # (3, N) — lazily computed

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self) -> None:
        """Populate the position cache with a fully vectorised calculation."""
        t               = self.t
        x0, y0, z0      = self.r0
        vx0, vy0, vz0   = self.v0

        if self.drag == 0.0:
            # ── Drag-free analytical solution ──────────────────────────
            x = x0 + vx0 * t
            y = y0 + vy0 * t
            z = z0 + vz0 * t - 0.5 * self.g * t ** 2

        else:
            # ── Linear drag  F_drag = −k v  (analytical) ────────────
            #   x(t) = x0 + vx0 (m/k)(1 − e^{−kt/m})   (same for y)
            #   z(t) = z0 + v_t t + (vz0 − v_t)(m/k)(1 − e^{−kt/m})
            #   where v_t = −mg/k  (terminal velocity in z)
            km  = self.drag / self.mass            # k/m  (s⁻¹)
            e   = np.exp(-km * t)                  # decay envelope  – shape (N,)
            inv = (1.0 - e) / km                   # ∫₀ᵗ e^{−kms} ds

            x = x0 + vx0 * inv
            y = y0 + vy0 * inv

            v_term = -self.mass * self.g / self.drag   # terminal z-velocity
            z = z0 + v_term * t + (vz0 - v_term) * inv

        self._pos = np.vstack((x, y, z))           # (3, N)

    # ------------------------------------------------------------------
    # Properties derived via np.gradient
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """(3, N) position array  [x; y; z]."""
        if self._pos is None:
            self.compute()
        return self._pos                            # type: ignore[return-value]

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

    # ------------------------------------------------------------------
    # Distance & speed
    # ------------------------------------------------------------------

    def arc_length(self) -> float:
        """Total path length along the 3-D trajectory (m)."""
        diff = np.diff(self.model.position, axis=1)    # (3, N-1)
        return float(np.sum(np.linalg.norm(diff, axis=0)))

    def speed(self) -> np.ndarray:
        """Instantaneous scalar speed at every time step (m s⁻¹), shape (N,)."""
        return np.linalg.norm(self.model.velocity, axis=0)

    # ------------------------------------------------------------------
    # Boundary / ground detection
    # ------------------------------------------------------------------

    def ground_hit_time(self) -> Optional[float]:
        """
        Linearly-interpolated time at which z first crosses from ≥ 0 to < 0.

        Returns None if the particle does not reach z = 0 within the window.
        """
        z = self.model.position[2]
        t = self.model.t
        crossings = np.where((z[:-1] >= 0.0) & (z[1:] < 0.0))[0]
        if len(crossings) == 0:
            return None
        i    = crossings[0]
        frac = z[i] / (z[i] - z[i + 1])               # linear interpolation weight
        return float(t[i] + frac * (t[i + 1] - t[i]))

    def in_bounds(
        self,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
        z_lim: tuple[float, float],
    ) -> np.ndarray:
        """
        Boolean mask of shape (N,) — True where the particle is inside the AABB.
        """
        x, y, z = self.model.position
        return (
            (x >= x_lim[0]) & (x <= x_lim[1]) &
            (y >= y_lim[0]) & (y <= y_lim[1]) &
            (z >= z_lim[0]) & (z <= z_lim[1])
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_data(self) -> dict:
        """
        Analytical reference for a drag-free parabola with the same initial
        conditions.  Used to cross-check the numerical trajectory.

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

    # ------------------------------------------------------------------
    # Public draw entry-point
    # ------------------------------------------------------------------

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

        # Prepare plot-safe data: downsampled + NaN-clipped at z = 0
        pos_raw  = self.model.position                  # (3, N) – full resolution
        pos_plot = self._prepare_plot_data(pos_raw, cap=cap_points)
        x, y, z  = pos_plot

        ax.plot(x, y, z, color="royalblue", linewidth=1.8, label="Trajectory")

        # Start / end markers (skip any leading/trailing NaNs)
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

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_plot_data(pos: np.ndarray, cap: int = 2000) -> np.ndarray:
        """
        1. Downsample to at most `cap` evenly-spaced points.
        2. Apply the NaN-separator trick: insert a NaN column immediately after
           the first index where z < 0 so matplotlib stops drawing underground.

        Returns a (3, M) array where M ≤ cap + 1.
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

        z_floor = min(float(zv.min()), 0.0) - 1.0   # slightly below lowest point
        x_back  = float(xv.max()) + 1.0
        y_back  = float(yv.max()) + 1.0

        kw = dict(color="slategray", alpha=self._SHADOW_ALPHA,
                  linewidth=1.0, linestyle="--")

        ax.plot(xv, yv, z_floor * np.ones_like(xv), **kw)          # XY floor
        ax.plot(xv, y_back * np.ones_like(xv), zv,  **kw)          # XZ back-Y wall
        ax.plot(x_back * np.ones_like(yv), yv, zv,  **kw)          # YZ back-X wall

    def _draw_velocity_quivers(
        self,
        ax:           Axes3D,
        pos_raw:      np.ndarray,
        cap:          int,
        stride:       int,
    ) -> None:
        """
        Plot normalised velocity arrows at every `stride`-th sampled point
        (restricted to above-ground positions).
        """
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
        mag   = np.where(mag == 0.0, 1.0, mag)                     # guard ÷0

        # Scale arrow length to ~5 % of the trajectory's bounding-box span
        span  = float(np.ptp(pos_raw, axis=1).max())
        scale = 0.05 * span if span > 0 else 1.0

        ax.quiver(
            px, py, pz,
            (vx / mag) * scale, (vy / mag) * scale, (vz / mag) * scale,
            color="darkorange", arrow_length_ratio=0.35,
            linewidth=0.9, label="Velocity (direction)",
        )

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def animate(
        self,
        show_projections: bool = True,
        n_frames: int = 250,
        interval: int = 30,
        repeat: bool = True,
    ) -> FuncAnimation:
        """
        Build and return a FuncAnimation that plays the trajectory back in time.

        Each frame advances a growing trail line and moves a dot along the path.
        A single velocity arrow is redrawn at the current position every frame.
        Axis limits are fixed before the first frame so the view never jumps.
        """
        self.fig.clear()
        ax: Axes3D = self.fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3-D Particle Trajectory  (Animated)", pad=12)
        self.ax = ax

        pos_raw = self.model.position       # (3, N_full)
        vel_raw = self.model.velocity       # (3, N_full)
        N_full  = pos_raw.shape[1]

        # Keep only above-ground frames, then subsample to n_frames
        above     = np.where(pos_raw[2] >= 0.0)[0]
        if len(above) == 0:
            above = np.arange(N_full)
        frame_idx = above[
            np.round(np.linspace(0, len(above) - 1,
                                 min(n_frames, len(above)))).astype(int)
        ]

        xf  = pos_raw[0, frame_idx]
        yf  = pos_raw[1, frame_idx]
        zf  = pos_raw[2, frame_idx]
        vxf = vel_raw[0, frame_idx]
        vyf = vel_raw[1, frame_idx]
        vzf = vel_raw[2, frame_idx]
        N_f = len(frame_idx)

        # Fix axis limits before animation starts so the view stays still
        x_pad = max((xf.max() - xf.min()) * 0.05, 1.0)
        y_pad = max((yf.max() - yf.min()) * 0.05, 1.0)
        z_pad = max((zf.max() - zf.min()) * 0.05, 1.0)
        ax.set_xlim(xf.min() - x_pad, xf.max() + x_pad)
        ax.set_ylim(yf.min() - y_pad, yf.max() + y_pad)
        ax.set_zlim(min(zf.min(), 0.0) - z_pad, zf.max() + z_pad)

        # Static elements drawn once
        if show_projections:
            self._draw_projections(ax, self._prepare_plot_data(pos_raw, cap=500))
        ax.scatter(xf[0],  yf[0],  zf[0],  color="limegreen", s=70,
                   depthshade=False, label="Start", zorder=5)
        ax.scatter(xf[-1], yf[-1], zf[-1], color="crimson",   s=40,
                   depthshade=False, alpha=0.35, label="End", zorder=5)

        # Mutable: growing trail + moving dot
        trail, = ax.plot([], [], [], color="royalblue", linewidth=1.8,
                         label="Trajectory")
        dot,   = ax.plot([], [], [], "o", color="white", markersize=9,
                         markeredgecolor="royalblue", markeredgewidth=2.5,
                         zorder=6)

        # Arrow scale fixed to trajectory extent
        span    = float(np.ptp(pos_raw, axis=1).max())
        v_scale = 0.06 * span if span > 0 else 1.0
        arrow_holder: list = []     # holds the current Quiver3D so we can remove it

        ax.legend(fontsize=8, loc="upper right")

        def _init():
            trail.set_data([], [])
            trail.set_3d_properties([])
            dot.set_data([], [])
            dot.set_3d_properties([])
            return trail, dot

        def _update(frame: int):
            i = frame + 1
            trail.set_data(xf[:i], yf[:i])
            trail.set_3d_properties(zf[:i])
            dot.set_data([xf[frame]], [yf[frame]])
            dot.set_3d_properties([zf[frame]])

            # Remove previous velocity arrow and draw a fresh one
            for a in arrow_holder:
                try:
                    a.remove()
                except Exception:
                    pass
            arrow_holder.clear()

            mag = float(np.linalg.norm([vxf[frame], vyf[frame], vzf[frame]]))
            if mag > 0:
                arrow_holder.append(ax.quiver(
                    xf[frame], yf[frame], zf[frame],
                    (vxf[frame] / mag) * v_scale,
                    (vyf[frame] / mag) * v_scale,
                    (vzf[frame] / mag) * v_scale,
                    color="darkorange", arrow_length_ratio=0.4, linewidth=1.5,
                ))
            return trail, dot

        return FuncAnimation(
            self.fig, _update, frames=N_f,
            init_func=_init, interval=interval,
            blit=False, repeat=repeat,
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
    and refreshes the canvas – all within a single event-handler call.
    """

    # (display label, dict key, default value)
    _PARAMS: list[tuple] = [
        ("x₀  (m)",      "x0",    0.0  ),
        ("y₀  (m)",      "y0",    0.0  ),
        ("z₀  (m)",      "z0",    0.0  ),
        ("vx₀ (m/s)",    "vx0",  20.0  ),
        ("vy₀ (m/s)",    "vy0",  10.0  ),
        ("vz₀ (m/s)",    "vz0",  30.0  ),
        ("g   (m/s²)",   "g",     9.81 ),
        ("drag (kg/s)",  "drag",  0.0  ),
        ("mass (kg)",    "mass",  1.0  ),
        ("t_max (s)",    "t_max", 7.0  ),
        ("N points",     "n_pts", 500.0),
    ]

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("3D Kinematics Visualiser")
        self.root.minsize(1100, 600)

        self._vars:    dict[str, tk.StringVar]        = {}
        self._fig:     Optional[plt.Figure]          = None
        self._canvas:  Optional[FigureCanvasTkAgg]   = None
        self._anim:    Optional[FuncAnimation]        = None
        self._show_vectors = tk.BooleanVar(value=True)
        self._show_proj    = tk.BooleanVar(value=True)

        self._build_layout()
        self._run()         # draw the default scenario on start-up

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        # ── Left: control panel ────────────────────────────────────────

        ctrl = ttk.LabelFrame(self.root, text="Simulation Parameters", padding=10)
        ctrl.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        for row_i, (label, key, default) in enumerate(self._PARAMS):
            var = tk.StringVar(value=str(default))
            self._vars[key] = var                               # type: ignore[assignment]

            ttk.Label(ctrl, text=label, width=14, anchor="w").grid(
                row=row_i, column=0, sticky="w", pady=4)

            entry = ttk.Entry(ctrl, textvariable=var, width=14, justify="right")
            entry.grid(row=row_i, column=1, padx=6, sticky="ew")
            entry.bind("<Return>", lambda _e: self._run())      # Enter key runs sim

        # ── Toggle checkboxes ──────────────────────────────────────────
        n = len(self._PARAMS)
        tog = ttk.Frame(ctrl)
        tog.grid(row=n, column=0, columnspan=3, pady=(10, 2), sticky="w")
        ttk.Checkbutton(tog, text="Velocity vectors",
                        variable=self._show_vectors).pack(side="left", padx=4)
        ttk.Checkbutton(tog, text="Plane projections",
                        variable=self._show_proj).pack(side="left", padx=4)

        # ── Buttons ────────────────────────────────────────────────────
        ttk.Button(
            ctrl, text="▶  Run Simulation", command=self._run,
        ).grid(row=n + 1, column=0, columnspan=3, pady=(8, 2), sticky="ew")

        btn_row = ttk.Frame(ctrl)
        btn_row.grid(row=n + 2, column=0, columnspan=3, pady=(2, 4), sticky="ew")
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)
        ttk.Button(btn_row, text="▶  Animate",
                   command=self._animate).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(btn_row, text="■  Stop",
                   command=self._stop_anim).grid(row=0, column=1, sticky="ew", padx=(2, 0))

        # ── Stats readout ──────────────────────────────────────────────
        self._stats_var = tk.StringVar(value="")
        ttk.Label(
            ctrl, textvariable=self._stats_var,
            justify="left", foreground="#1a4480", wraplength=280,
            font=("Courier", 9),
        ).grid(row=n + 3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # ── Right: plot canvas ─────────────────────────────────────────
        plot_frame = ttk.LabelFrame(self.root, text="3-D Plot", padding=4)
        plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self._fig    = plt.Figure(figsize=(9, 6.5))
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _get(self, key: str) -> float:
        """Parse a StringVar entry to float, raising ValueError on bad input."""
        raw = self._vars[key].get().strip()                     # type: ignore[union-attr]
        return float(raw)

    def _parse_inputs(self) -> tuple:
        """Read all entry boxes and return validated simulation parameters."""
        r0    = np.array([self._get("x0"),  self._get("y0"),  self._get("z0")])
        v0    = np.array([self._get("vx0"), self._get("vy0"), self._get("vz0")])
        g     = max(self._get("g"),     1e-9)
        drag  = max(self._get("drag"),  0.0)
        mass  = max(self._get("mass"),  1e-3)
        t_max = max(self._get("t_max"), 0.1)
        n_pts = int(max(self._get("n_pts"), 10))
        return r0, v0, g, drag, mass, t_max, n_pts

    def _stop_anim(self) -> None:
        """Stop any running FuncAnimation."""
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None

    # ------------------------------------------------------------------
    # Simulation trigger
    # ------------------------------------------------------------------

    def _run(self) -> None:
        self._stop_anim()                   # cancel animation before static redraw

        try:
            r0, v0, g, drag, mass, t_max, n_pts = self._parse_inputs()
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
            self._canvas.draw()             # type: ignore[union-attr]

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

    # ------------------------------------------------------------------
    # Animation trigger
    # ------------------------------------------------------------------

    def _animate(self) -> None:
        self._stop_anim()                   # stop any existing animation first

        try:
            r0, v0, g, drag, mass, t_max, n_pts = self._parse_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", f"Please enter a valid number.\n\n{exc}")
            return

        try:
            t     = np.linspace(0.0, t_max, n_pts)
            model = KinematicsModel(t, r0, v0, g=g, drag=drag, mass=mass)
            vis   = Visualiser3D(model, fig=self._fig)

            self._anim = vis.animate(
                show_projections = self._show_proj.get(),
                n_frames         = min(n_pts, 300),
                interval         = 30,
                repeat           = True,
            )
            self._canvas.draw()             # type: ignore[union-attr]
            self._stats_var.set("Animation running — press  ■ Stop  to freeze.")

        except Exception as exc:
            messagebox.showerror("Animation Error", str(exc))

    # ------------------------------------------------------------------

    def run(self) -> None:
        """Enter the Tkinter main-loop (blocking)."""
        self.root.mainloop()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    GUIController().run()
#runwith C:\Users\ruper\AppData\Local\Python\bin\python.exe Kinematics_visualiser.py


