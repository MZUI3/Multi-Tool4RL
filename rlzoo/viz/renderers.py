"""Frame sources for the visualiser.

`FrameSource` first tries the environment's own `render()` (gymnasium `rgb_array`; needs pygame for the
classic-control envs, Box2D / MuJoCo for theirs). When that is unavailable it falls back to a built-in
matplotlib drawing of the environment state, which covers the classic-control tasks and every env that ships
with rlzoo (BitFlip, PointReach, PixelGrid, Spread, Predator-Prey, Matrix game).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np


class MplCanvas:
    """Off-screen matplotlib figure rendered to an RGB array."""

    def __init__(self, width: float = 4.0, height: float = 4.0, dpi: int = 100):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_axis_off()

    def begin(self, xlim, ylim, aspect="equal"):
        ax = self.ax
        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if aspect:
            ax.set_aspect(aspect)
        return ax

    def array(self) -> np.ndarray:
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()


# ------------------------------------------------------------------------------ classic control


def _cartpole(env, cv: MplCanvas):
    x, _, theta, _ = env.unwrapped.state
    ax = cv.begin((-2.6, 2.6), (-0.6, 1.6))
    ax.axhline(0, color="#999", lw=1)
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((x - 0.25, -0.15), 0.5, 0.3, color="#333"))
    length = 1.0
    ax.plot([x, x + length * math.sin(theta)], [0, length * math.cos(theta)], color="#d9822b", lw=6, solid_capstyle="round")
    ax.plot([x], [0], "o", color="#4a90e2", ms=8)
    return cv.array()


def _mountaincar(env, cv: MplCanvas):
    pos, vel = env.unwrapped.state[:2]
    ax = cv.begin((-1.3, 0.7), (-1.3, 1.3), aspect=None)
    xs = np.linspace(-1.2, 0.6, 200)
    ax.plot(xs, np.sin(3 * xs), color="#999", lw=2)
    ax.plot([0.5], [np.sin(3 * 0.5) + 0.08], marker="v", color="#4caf50", ms=12)
    ax.plot([pos], [np.sin(3 * pos) + 0.06], "o", color="#d9822b", ms=14)
    ax.text(-1.25, 1.1, f"vel {vel:+.3f}", fontsize=10)
    return cv.array()


def _pendulum(env, cv: MplCanvas):
    theta, thetadot = env.unwrapped.state
    ax = cv.begin((-1.3, 1.3), (-1.3, 1.3))
    ax.plot([0, math.sin(theta)], [0, math.cos(theta)], color="#d9822b", lw=8, solid_capstyle="round")
    ax.plot([0], [0], "o", color="#333", ms=10)
    ax.text(-1.2, 1.1, f"omega {thetadot:+.2f}", fontsize=10)
    return cv.array()


def _acrobot(env, cv: MplCanvas):
    s = env.unwrapped.state
    t1, t2 = s[0], s[1]
    ax = cv.begin((-2.3, 2.3), (-2.3, 2.3))
    p1 = (math.sin(t1), -math.cos(t1))
    p2 = (p1[0] + math.sin(t1 + t2), p1[1] - math.cos(t1 + t2))
    ax.axhline(1.0, color="#4caf50", lw=1, ls="--")
    ax.plot([0, p1[0]], [0, p1[1]], color="#4a90e2", lw=6, solid_capstyle="round")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#d9822b", lw=6, solid_capstyle="round")
    ax.plot([0, p1[0]], [0, p1[1]], "o", color="#333", ms=7)
    return cv.array()


# ------------------------------------------------------------------------------ built-in MARL envs


def _spread(env, cv: MplCanvas):
    ax = cv.begin((-1.6, 1.6), (-1.6, 1.6))
    for lm in env.landmarks:
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"]).Circle(lm, env.agent_size, color="#4caf50", alpha=0.6))
    colors = ["#4a90e2", "#d9822b", "#9b59b6", "#e74c3c", "#1abc9c", "#f1c40f"]
    for i, p in enumerate(env.pos):
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"]).Circle(p, env.agent_size, color=colors[i % len(colors)]))
        ax.text(p[0], p[1], str(i), ha="center", va="center", color="white", fontsize=8)
    ax.text(-1.55, 1.45, f"t={env.t}", fontsize=10)
    return cv.array()


def _predator_prey(env, cv: MplCanvas):
    G = env.G
    ax = cv.begin((-0.5, G - 0.5), (-0.5, G - 0.5))
    ax.invert_yaxis()
    for i in range(G + 1):
        ax.axhline(i - 0.5, color="#ddd", lw=0.5)
        ax.axvline(i - 0.5, color="#ddd", lw=0.5)
    for a in env.agents:
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((a[1] - 0.4, a[0] - 0.4), 0.8, 0.8, color="#4a90e2"))
    for p, alive in zip(env.prey, env.alive):
        if alive:
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"]).Circle((p[1], p[0]), 0.35, color="#e74c3c"))
    return cv.array()


def _matrix_game(env, cv: MplCanvas):
    pay = env.payoff
    ax = cv.begin((-0.5, pay.shape[1] - 0.5), (-0.5, pay.shape[0] - 0.5), aspect=None)
    ax.invert_yaxis()
    ax.imshow(pay, cmap="RdYlGn")
    for i in range(pay.shape[0]):
        for j in range(pay.shape[1]):
            ax.text(j, i, f"{pay[i, j]:.0f}", ha="center", va="center", fontsize=12)
    last = getattr(env, "_last_actions", None)
    if last is not None:
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((last[1] - 0.5, last[0] - 0.5), 1, 1, fill=False, lw=3, color="black"))
    return cv.array()


_BY_CLASS = {
    "CartPoleEnv": _cartpole,
    "MountainCarEnv": _mountaincar,
    "Continuous_MountainCarEnv": _mountaincar,
    "PendulumEnv": _pendulum,
    "AcrobotEnv": _acrobot,
    "SpreadEnv": _spread,
    "PredatorPreyEnv": _predator_prey,
    "MatrixGameEnv": _matrix_game,
}


def fallback_renderer(env) -> Optional[Callable[[], np.ndarray]]:
    base = getattr(env, "unwrapped", env)
    fn = _BY_CLASS.get(type(base).__name__)
    if fn is None:
        return None
    cv = MplCanvas()
    return lambda: fn(env, cv)


class FrameSource:
    """Callable returning the current RGB frame of `env`, using native rendering when it works."""

    def __init__(self, env, prefer_native: bool = True):
        self.env = env
        self.native = None
        self.reason = ""
        if prefer_native and hasattr(env, "render"):
            try:
                if hasattr(env, "reset"):
                    env.reset(seed=0)  # gymnasium refuses render() before the first reset
                frame = env.render()
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    self.native = env.render
                else:
                    self.reason = "env.render() returned no rgb_array"
            except Exception as exc:  # noqa: BLE001 - e.g. gymnasium DependencyNotInstalled (pygame)
                self.reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
        self.fallback = fallback_renderer(env) if self.native is None else None
        if self.native is None and self.fallback is None:
            raise RuntimeError(
                f"No renderer for {type(getattr(env, 'unwrapped', env)).__name__}: {self.reason or 'render unsupported'}. "
                "Install the env's rendering dependency (e.g. `pip install pygame` for classic control) or add a "
                "fallback drawer in rlzoo/viz/renderers.py")

    @property
    def mode(self) -> str:
        return "native" if self.native is not None else "fallback"

    def __call__(self) -> np.ndarray:
        frame = self.native() if self.native is not None else self.fallback()
        return np.ascontiguousarray(frame[..., :3])
