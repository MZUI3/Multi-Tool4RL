"""Watch a trained agent in its environment.

* `Simulation` steps env + policy and returns frames plus diagnostics (action, reward, Q-values / action
  probabilities / value from `Algorithm.explain`).
* `Player` is an interactive matplotlib window: play / pause / single-step / reset / speed / deterministic toggle.
* `save_episode` exports recorded steps to .gif (PIL), .mp4 (ffmpeg via matplotlib) or a self-contained .html
  player with a scrubber and the same diagnostics.
"""
from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rlzoo.envs.gym_env import flatten_obs, make_gym_env, spec_from_env, to_env_action
from rlzoo.envs.marl import make_marl_env
from rlzoo.registry import get_info
from rlzoo.viz.renderers import FrameSource


@dataclass
class StepRecord:
    episode: int
    step: int
    frame: np.ndarray
    action: Any
    reward: float
    ret: float
    done: bool
    explain: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)


def _to_list(value):
    if isinstance(value, np.ndarray):
        return [round(float(v), 4) for v in value.ravel()] if value.ndim == 1 else [[round(float(v), 4) for v in row] for row in value]
    if isinstance(value, (np.floating, np.integer)):
        return round(float(value), 4)
    return value


class Simulation:
    """Owns one environment and the policy; `step()` advances one environment step."""

    def __init__(self, algo, cfg: Dict[str, Any], seed: int = 0, deterministic: bool = True, max_steps: int = 10_000):
        self.algo, self.cfg, self.seed = algo, cfg, int(seed)
        self.kind = get_info(cfg["algo"]).kind
        self.deterministic = deterministic
        self.max_steps = int(max_steps)
        env_cfg = cfg["env"]
        if self.kind == "rl":
            try:
                self.env = make_gym_env(env_cfg, render_mode="rgb_array")
            except Exception:  # noqa: BLE001 - env refuses the render mode: use state drawing instead
                self.env = make_gym_env(env_cfg)
            self.spec = spec_from_env(self.env)
        else:
            self.env = make_marl_env(env_cfg)
            self.spec = None
        self.frames = FrameSource(self.env)
        self.episode = -1
        self.step_idx = 0
        self.ret = 0.0
        self.returns: List[float] = []
        self.done = True
        self._obs = None
        self.last: Optional[StepRecord] = None

    # ---------------------------------------------------------------- env wrappers
    def reset(self) -> StepRecord:
        self.episode += 1
        self.step_idx, self.ret, self.done = 0, 0.0, False
        seed = self.seed + self.episode
        if self.kind == "rl":
            obs, _ = self.env.reset(seed=seed)
            self._obs = flatten_obs(obs, self.spec)
        else:
            self._obs = self.env.reset(seed=seed)  # (obs, state, avail)
        if hasattr(self.algo, "reset_hidden"):
            self.algo.reset_hidden(1)
        self.last = StepRecord(self.episode, 0, self.frames(), None, 0.0, 0.0, False, self._explain(), {})
        return self.last

    def _explain(self) -> Dict[str, Any]:
        try:
            if self.kind == "rl":
                out = self.algo.explain(self._obs)
            else:
                o, s, av = self._obs
                out = self.algo.explain(o, s, av)
        except Exception:  # noqa: BLE001 - diagnostics are optional
            out = {}
        return {k: _to_list(v) for k, v in (out or {}).items()}

    def step(self) -> StepRecord:
        if self.done:
            return self.reset()
        if self.kind == "rl":
            a = self.algo.act(self._obs[None], deterministic=self.deterministic)[0]
            obs, r, term, trunc, info = self.env.step(to_env_action(a, self.spec))
            if hasattr(self.algo, "observe"):
                self.algo.observe(float(r))
            self._obs = flatten_obs(obs, self.spec)
            action = int(a) if self.spec.discrete else [round(float(x), 3) for x in np.asarray(a).ravel()]
        else:
            o, s, av = self._obs
            a = self.algo.act(o[None], s[None], av[None], deterministic=self.deterministic)[0]
            if hasattr(self.env, "payoff"):
                self.env._last_actions = (int(a[0]), int(a[1]))
            o, s, av, r, term, trunc, info = self.env.step(a)
            self._obs = (o, s, av)
            action = [int(x) for x in a]
        self.step_idx += 1
        self.ret += float(r)
        self.done = bool(term or trunc) or self.step_idx >= self.max_steps
        if self.done:
            self.returns.append(self.ret)
        info = {k: _to_list(v) for k, v in (info or {}).items() if np.isscalar(v)}
        self.last = StepRecord(self.episode, self.step_idx, self.frames(), action, float(r), self.ret, self.done,
                               self._explain(), info)
        return self.last

    def close(self):
        self.env.close()


def record_episodes(sim: Simulation, episodes: int = 1, max_steps: Optional[int] = None) -> List[StepRecord]:
    records = []
    for _ in range(int(episodes)):
        records.append(sim.reset())
        while not sim.done:
            records.append(sim.step())
            if max_steps and sim.step_idx >= max_steps:
                break
    return records


# ---------------------------------------------------------------------------------------- export


def _resize(frame: np.ndarray, max_width: int):
    from PIL import Image

    img = Image.fromarray(frame)
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    elif img.width < max_width // 2:  # tiny native frames (grid worlds, bit strings): upscale without blur
        k = max(1, max_width // img.width)
        img = img.resize((img.width * k, img.height * k), Image.NEAREST)
    return img


def save_gif(records: List[StepRecord], path, fps: int = 30, max_width: int = 480) -> Path:
    frames = [_resize(r.frame, max_width) for r in records]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return path


def save_mp4(records: List[StepRecord], path, fps: int = 30) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    if not animation.FFMpegWriter.isAvailable():
        raise RuntimeError("ffmpeg not found; install it (or export .gif / .html instead)")
    h, w = records[0].frame.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(records[0].frame)
    writer = animation.FFMpegWriter(fps=fps)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(path), dpi=100):
        for r in records:
            im.set_data(r.frame)
            writer.grab_frame()
    plt.close(fig)
    return path


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
 .row{display:flex;gap:16px;flex-wrap:wrap} img{background:#000;max-width:100%;border:1px solid #333}
 .panel{min-width:260px;flex:1} table{border-collapse:collapse;font-size:14px} td{padding:2px 8px}
 .bar{height:14px;background:#4a90e2;margin:2px 0} .barlabel{font-size:11px;color:#aaa}
 button{margin:2px;padding:4px 10px} input[type=range]{width:100%}
 .grid{display:grid;gap:2px} .cell{height:16px;font-size:10px;text-align:center;color:#fff}
</style></head><body>
<h3>__TITLE__</h3>
<div class="row">
 <div><img id="frame" width="__W__"><br>
  <input type="range" id="slider" min="0" max="0" value="0"><br>
  <button id="play">Pause</button><button id="prev">&lt;</button><button id="next">&gt;</button>
  <button id="slower">slower</button><button id="faster">faster</button> <span id="speed"></span></div>
 <div class="panel"><table id="hud"></table><div id="explain"></div></div>
</div>
<script>
const steps=__STEPS__; const frames=__FRAMES__; let i=0, playing=true, fps=__FPS__;
const img=document.getElementById('frame'), slider=document.getElementById('slider');
slider.max=steps.length-1;
function fmt(v){return (typeof v==='number')?v.toFixed(3):JSON.stringify(v);}
function show(k){ i=(k+steps.length)%steps.length; const s=steps[i]; img.src='data:image/png;base64,'+frames[i]; slider.value=i;
 let rows=[['episode',s.episode],['step',s.step],['action',JSON.stringify(s.action)],['reward',s.reward.toFixed(3)],['return',s.ret.toFixed(3)],['done',s.done]];
 for(const k in s.info) rows.push([k,fmt(s.info[k])]);
 document.getElementById('hud').innerHTML=rows.map(r=>`<tr><td>${r[0]}</td><td><b>${r[1]}</b></td></tr>`).join('');
 let ex=''; for(const k in s.explain){ const v=s.explain[k];
  if(Array.isArray(v)&&Array.isArray(v[0])){ ex+=`<div class="barlabel">${k}</div><div class="grid" style="grid-template-columns:repeat(${v[0].length},1fr)">`;
    const flat=v.flat(); const lo=Math.min(...flat), hi=Math.max(...flat);
    for(const row of v) for(const x of row){ const t=(hi>lo)?(x-lo)/(hi-lo):0.5; ex+=`<div class="cell" style="background:rgba(74,144,226,${0.15+0.85*t})">${x.toFixed(2)}</div>`;} ex+='</div>';}
  else if(Array.isArray(v)){ const lo=Math.min(0,...v), hi=Math.max(0,...v); ex+=`<div class="barlabel">${k}</div>`;
    v.forEach((x,j)=>{const w=(hi>lo)?100*(x-lo)/(hi-lo):50; ex+=`<div class="barlabel">[${j}] ${x.toFixed(3)}</div><div class="bar" style="width:${Math.max(1,w)}%"></div>`;});}
  else ex+=`<div class="barlabel">${k}: ${fmt(v)}</div>`; }
 document.getElementById('explain').innerHTML=ex; document.getElementById('speed').textContent=fps.toFixed(0)+' fps'; }
let timer=null; function loop(){ if(playing) show(i+1); timer=setTimeout(loop,1000/fps);} loop(); show(0);
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play';};
document.getElementById('prev').onclick=()=>{playing=false;show(i-1);}; document.getElementById('next').onclick=()=>{playing=false;show(i+1);};
document.getElementById('slower').onclick=()=>{fps=Math.max(1,fps/1.5);}; document.getElementById('faster').onclick=()=>{fps=Math.min(120,fps*1.5);};
slider.oninput=()=>{playing=false;show(parseInt(slider.value));};
document.addEventListener('keydown',e=>{ if(e.key===' '){document.getElementById('play').click();e.preventDefault();} if(e.key==='ArrowRight')document.getElementById('next').click(); if(e.key==='ArrowLeft')document.getElementById('prev').click();});
</script></body></html>
"""


def save_html(records: List[StepRecord], path, fps: int = 30, title: str = "rlzoo episode", max_width: int = 480) -> Path:
    frames = []
    for r in records:
        buf = io.BytesIO()
        _resize(r.frame, max_width).save(buf, format="PNG", optimize=True)
        frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    steps = [dict(episode=r.episode, step=r.step, action=_to_list(r.action), reward=r.reward, ret=r.ret, done=r.done,
                  explain=r.explain, info=r.info) for r in records]
    w = min(max_width, records[0].frame.shape[1])
    html = (_HTML.replace("__TITLE__", title).replace("__W__", str(w)).replace("__FPS__", str(int(fps)))
            .replace("__STEPS__", json.dumps(steps)).replace("__FRAMES__", json.dumps(frames)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def save_episode(records: List[StepRecord], path, fps: int = 30, title: str = "rlzoo episode") -> Path:
    suffix = Path(path).suffix.lower()
    if suffix == ".gif":
        return save_gif(records, path, fps)
    if suffix == ".mp4":
        return save_mp4(records, path, fps)
    if suffix in (".html", ".htm"):
        return save_html(records, path, fps, title)
    raise ValueError(f"unsupported export format '{suffix}' (use .gif, .mp4 or .html)")


# ---------------------------------------------------------------------------------------- player


class Player:
    """Interactive matplotlib viewer.

    keys: space play/pause | n or -> step | r reset | d deterministic/stochastic | + / - speed | s screenshot | q quit
    """

    def __init__(self, sim: Simulation, fps: int = 30, title: str = ""):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        self.sim, self.fps, self.paused = sim, float(fps), False
        self.title = title
        self.fig = plt.figure(figsize=(11, 6))
        self.fig.canvas.manager.set_window_title(title or "rlzoo viewer") if getattr(self.fig.canvas, "manager", None) else None
        gs = GridSpec(3, 2, width_ratios=[3, 1.6], height_ratios=[1.2, 1, 1], figure=self.fig)
        self.ax_img = self.fig.add_subplot(gs[:, 0])
        self.ax_img.set_axis_off()
        self.ax_hud = self.fig.add_subplot(gs[0, 1])
        self.ax_hud.set_axis_off()
        self.ax_explain = self.fig.add_subplot(gs[1, 1])
        self.ax_returns = self.fig.add_subplot(gs[2, 1])
        rec = sim.reset()
        self.im = self.ax_img.imshow(rec.frame)
        self.hud_text = self.ax_hud.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=10)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._draw(rec)

    def on_key(self, event):
        import matplotlib.pyplot as plt

        k = event.key
        if k == " ":
            self.paused = not self.paused
        elif k in ("n", "right"):
            self.paused = True
            self._draw(self.sim.step())
        elif k == "r":
            self._draw(self.sim.reset())
        elif k == "d":
            self.sim.deterministic = not self.sim.deterministic
        elif k in ("+", "="):
            self.fps = min(240.0, self.fps * 1.5)
            self._set_interval()
        elif k == "-":
            self.fps = max(1.0, self.fps / 1.5)
            self._set_interval()
        elif k == "s":
            out = Path(f"screenshot_ep{self.sim.episode}_step{self.sim.step_idx}.png")
            self.fig.savefig(out, dpi=120)
            print(f"[viewer] saved {out}")
        elif k == "q":
            plt.close(self.fig)

    def _set_interval(self):
        if getattr(self, "anim", None) is not None and getattr(self.anim, "event_source", None) is not None:
            self.anim.event_source.interval = 1000.0 / self.fps

    def _draw(self, rec: StepRecord):
        sim = self.sim
        self.im.set_data(rec.frame)
        self.im.set_extent((0, rec.frame.shape[1], rec.frame.shape[0], 0))
        lines = [self.title, f"episode {rec.episode}   step {rec.step}", f"action  {rec.action}",
                 f"reward  {rec.reward:+.3f}", f"return  {rec.ret:+.3f}",
                 f"{'deterministic' if sim.deterministic else 'stochastic'}   {self.fps:.0f} fps   {'PAUSED' if self.paused else ''}"]
        lines += [f"{k} {v}" for k, v in rec.info.items()]
        lines.append("space pause | n step | r reset | d det/stoch | +/- speed | s shot | q quit")
        self.hud_text.set_text("\n".join(lines))
        ax = self.ax_explain
        ax.clear()
        for k, v in rec.explain.items():
            if isinstance(v, list) and v and isinstance(v[0], list):
                ax.imshow(np.asarray(v, dtype=float), aspect="auto", cmap="viridis")
                ax.set_title(k, fontsize=9)
                ax.set_xlabel("action")
                ax.set_ylabel("agent")
                break
            if isinstance(v, list):
                ax.bar(range(len(v)), v, color="#4a90e2")
                ax.set_title(k + "  " + "  ".join(f"{kk}={vv:.3g}" for kk, vv in rec.explain.items() if not isinstance(vv, list)), fontsize=9)
                ax.axhline(0, color="#999", lw=0.8)
                break
        else:
            ax.set_axis_off()
            ax.text(0.0, 0.5, "  ".join(f"{k}={v:.3g}" for k, v in rec.explain.items() if not isinstance(v, list)), fontsize=9)
        axr = self.ax_returns
        axr.clear()
        if sim.returns:
            axr.plot(range(len(sim.returns)), sim.returns, "o-", color="#d9822b", ms=4)
        axr.set_title("episode returns", fontsize=9)
        axr.grid(alpha=0.3)
        self.fig.canvas.draw_idle()

    def _tick(self, _frame):
        if not self.paused:
            self._draw(self.sim.step())
        return []

    def show(self):
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        self.anim = animation.FuncAnimation(self.fig, self._tick, interval=1000.0 / self.fps, cache_frame_data=False)
        plt.show()


# ---------------------------------------------------------------------------------------- entry point


def _interactive_backend_available() -> bool:
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "agg" in backend and "tk" not in backend and "qt" not in backend and "wx" not in backend:
        try:
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False
    return True


def watch_run(run_dir, save: Optional[str] = None, window: bool = True, episodes: int = 3, fps: Optional[int] = None,
              seed: int = 0, deterministic: bool = True, max_steps: int = 2000, algo=None, cfg=None, device=None):
    """Open the live viewer for a finished run and/or export recorded episodes (.gif / .mp4 / .html)."""
    from rlzoo.runner import load_run

    if algo is None or cfg is None:
        algo, cfg = load_run(run_dir, device)
    sim = Simulation(algo, cfg, seed=seed, deterministic=deterministic, max_steps=max_steps)
    env_fps = (getattr(sim.env, "metadata", {}) or {}).get("render_fps", 30) if sim.kind == "rl" else 10
    fps = int(fps or env_fps or 30)
    title = f"{cfg['algo']} on {cfg['env']['id']}"
    print(f"[viewer] {title} | renderer: {sim.frames.mode}" + (f" ({sim.frames.reason})" if sim.frames.reason else ""))
    if save:
        records = record_episodes(sim, episodes, max_steps)
        out = save_episode(records, save, fps, title)
        rets = sim.returns
        print(f"[viewer] saved {len(records)} frames / {len(rets)} episodes (returns {[round(r, 2) for r in rets]}) -> {out}")
    if window:
        if not _interactive_backend_available():
            fallback = Path(run_dir) / "episode.html"
            print(f"[viewer] no interactive matplotlib backend; exporting {fallback} instead (open it in a browser)")
            records = record_episodes(Simulation(algo, cfg, seed=seed, deterministic=deterministic, max_steps=max_steps), episodes, max_steps)
            save_html(records, fallback, fps, title)
        else:
            Player(sim, fps=fps, title=title).show()
    sim.close()
