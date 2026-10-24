"""Small built-in Gymnasium environments (registered on import; no extra dependencies).

BitFlip-v0    goal-conditioned bit flipping (Andrychowicz et al., 2017) - the classic HER test, discrete.
PointReach-v0 goal-conditioned 2-D point mass with continuous force actions - HER with DDPG/TD3/SAC.
PixelGrid-v0  grid-world navigation with an RGB image observation - tests CNN encoders / DRQN.

All three implement `render()` (rgb_array) so the visualiser can show them.
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None


if gym is not None:

    class BitFlipEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

        def __init__(self, n_bits: int = 8, render_mode=None):
            self.n = int(n_bits)
            self.render_mode = render_mode
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(0, 1, (self.n,), np.float32),
                "achieved_goal": spaces.Box(0, 1, (self.n,), np.float32),
                "desired_goal": spaces.Box(0, 1, (self.n,), np.float32),
            })
            self.action_space = spaces.Discrete(self.n)
            self.state = np.zeros(self.n, np.float32)
            self.goal = np.zeros(self.n, np.float32)
            self.t = 0

        def _obs(self):
            return {"observation": self.state.copy(), "achieved_goal": self.state.copy(), "desired_goal": self.goal.copy()}

        def compute_reward(self, achieved, desired, info=None):
            achieved, desired = np.asarray(achieved), np.asarray(desired)
            return -(np.abs(achieved - desired).sum(-1) > 0).astype(np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.state = self.np_random.integers(0, 2, self.n).astype(np.float32)
            self.goal = self.np_random.integers(0, 2, self.n).astype(np.float32)
            while np.array_equal(self.state, self.goal):
                self.goal = self.np_random.integers(0, 2, self.n).astype(np.float32)
            self.t = 0
            return self._obs(), {}

        def step(self, action):
            self.state[int(action)] = 1.0 - self.state[int(action)]
            self.t += 1
            reward = float(self.compute_reward(self.state, self.goal))
            success = reward == 0.0
            truncated = self.t >= self.n and not success
            return self._obs(), reward, bool(success), bool(truncated), {"is_success": float(success)}

        def render(self):
            cell = 24
            img = np.full((2 * cell + 8, self.n * cell, 3), 255, np.uint8)
            for row, bits in enumerate((self.state, self.goal)):
                y0 = row * (cell + 8)
                for i, b in enumerate(bits):
                    color = (40, 120, 220) if b > 0.5 else (230, 230, 230)
                    img[y0 + 2:y0 + cell - 2, i * cell + 2:(i + 1) * cell - 2] = color
            return img

    class PointReachEnv(gym.Env):
        """Point mass in [-1, 1]^2 pushed by a 2-D force; reward -1 until within `threshold` of the goal."""

        metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

        def __init__(self, threshold: float = 0.1, max_steps: int = 50, render_mode=None):
            self.threshold, self.max_steps = threshold, max_steps
            self.render_mode = render_mode
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(-2, 2, (4,), np.float32),
                "achieved_goal": spaces.Box(-1, 1, (2,), np.float32),
                "desired_goal": spaces.Box(-1, 1, (2,), np.float32),
            })
            self.action_space = spaces.Box(-1, 1, (2,), np.float32)
            self.pos = np.zeros(2, np.float32)
            self.vel = np.zeros(2, np.float32)
            self.goal = np.zeros(2, np.float32)
            self.t = 0

        def _obs(self):
            return {"observation": np.concatenate([self.pos, self.vel]).astype(np.float32),
                    "achieved_goal": self.pos.copy(), "desired_goal": self.goal.copy()}

        def compute_reward(self, achieved, desired, info=None):
            d = np.linalg.norm(np.asarray(achieved) - np.asarray(desired), axis=-1)
            return -(d > self.threshold).astype(np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.pos = self.np_random.uniform(-1, 1, 2).astype(np.float32)
            self.vel = np.zeros(2, np.float32)
            self.goal = self.np_random.uniform(-1, 1, 2).astype(np.float32)
            self.t = 0
            return self._obs(), {}

        def step(self, action):
            a = np.clip(np.asarray(action, np.float32), -1, 1)
            self.vel = (self.vel * 0.8 + a * 0.1).astype(np.float32)
            self.pos = np.clip(self.pos + self.vel, -1, 1).astype(np.float32)
            self.t += 1
            reward = float(self.compute_reward(self.pos, self.goal))
            success = reward == 0.0
            truncated = self.t >= self.max_steps and not success
            return self._obs(), reward, bool(success), bool(truncated), {"is_success": float(success)}

        def render(self):
            size = 200
            img = np.full((size, size, 3), 250, np.uint8)

            def px(p):
                return int((p[0] + 1) / 2 * (size - 1)), int((1 - (p[1] + 1) / 2) * (size - 1))

            for p, color, r in ((self.goal, (60, 180, 75), int(self.threshold * size / 2)), (self.pos, (220, 60, 50), 5)):
                cx, cy = px(p)
                ys, xs = np.ogrid[:size, :size]
                img[(xs - cx) ** 2 + (ys - cy) ** 2 <= r * r] = color
            return img

    class PixelGridEnv(gym.Env):
        """Agent (red) must reach the goal (green) on a grid while avoiding walls; observation is an RGB image."""

        metadata = {"render_modes": ["rgb_array"], "render_fps": 8}
        _MOVES = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])  # up, down, left, right (x, y)

        def __init__(self, size: int = 8, cell: int = 4, n_walls: int = 6, max_steps: int = 50, render_mode=None):
            self.size, self.cell, self.n_walls, self.max_steps = int(size), int(cell), int(n_walls), int(max_steps)
            self.render_mode = render_mode
            px = self.size * self.cell
            self.observation_space = spaces.Box(0, 255, (px, px, 3), np.uint8)
            self.action_space = spaces.Discrete(4)
            self.walls = np.zeros((self.size, self.size), bool)
            self.pos = np.zeros(2, int)
            self.goal = np.zeros(2, int)
            self.t = 0

        def _image(self, scale=1):
            c = self.cell * scale
            img = np.full((self.size * c, self.size * c, 3), 255, np.uint8)
            for y in range(self.size):
                for x in range(self.size):
                    if self.walls[y, x]:
                        img[y * c:(y + 1) * c, x * c:(x + 1) * c] = (60, 60, 60)
            gx, gy = self.goal
            img[gy * c:(gy + 1) * c, gx * c:(gx + 1) * c] = (60, 180, 75)
            ax, ay = self.pos
            img[ay * c:(ay + 1) * c, ax * c:(ax + 1) * c] = (220, 60, 50)
            return img

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.walls[:] = False
            cells = self.np_random.permutation(self.size * self.size)
            self.pos = np.array([cells[0] % self.size, cells[0] // self.size])
            self.goal = np.array([cells[1] % self.size, cells[1] // self.size])
            for idx in cells[2:2 + self.n_walls]:
                self.walls[idx // self.size, idx % self.size] = True
            self.t = 0
            return self._image(), {}

        def step(self, action):
            nxt = self.pos + self._MOVES[int(action)]
            if 0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size and not self.walls[nxt[1], nxt[0]]:
                self.pos = nxt
            self.t += 1
            done = bool(np.array_equal(self.pos, self.goal))
            reward = 1.0 if done else -0.01
            truncated = self.t >= self.max_steps and not done
            return self._image(), reward, done, truncated, {"is_success": float(done)}

        def render(self):
            return self._image(scale=max(1, 160 // (self.size * self.cell)))

    for _id, _cls in (("BitFlip-v0", BitFlipEnv), ("PointReach-v0", PointReachEnv), ("PixelGrid-v0", PixelGridEnv)):
        if _id not in gym.registry:
            gym.register(id=_id, entry_point=f"{__name__}:{_cls.__name__}")
