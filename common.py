"""
common.py —— 所有课程脚本共用的小工具。

只做四件 MuJoCo 的教学必备事情：
  1. 离屏渲染（不弹窗口，Apple Silicon 上可直接用）
  2. 把帧序列存成 GIF（方便在 Finder/浏览器里看）
  3. 弹交互式 viewer（真机窗口，需要桌面环境）
  4. 一点打印/计时的小糖

不需要装 imageio / ffmpeg，Pillow 就够了。
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import sysconfig
import time

import mujoco
import numpy as np
from PIL import Image

# 所有产物统一写到 mujoco-lab/out/
OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. 渲染
# --------------------------------------------------------------------------- #
def make_camera(azimuth: float = 90.0, elevation: float = -20.0,
                distance: float = 2.5, lookat=(0.0, 0.0, 0.4)) -> mujoco.MjvCamera:
    """MuJoCo 的相机就是三个球坐标 + 一个注视点，可以随意插值做运镜。"""
    cam = mujoco.MjvCamera()
    cam.azimuth = azimuth      # 水平角（度）
    cam.elevation = elevation  # 俯仰角（度），负数表示从上往下看
    cam.distance = distance    # 距离
    cam.lookat[:] = lookat     # 注视点
    return cam


def render_frames(model, data, *, n_frames: int = 120, steps_per_frame: int = 10,
                  width: int = 480, height: int = 360, camera=None,
                  controller=None, track=None,
                  scene_option: mujoco.MjvOption | None = None) -> list[np.ndarray]:
    """
    跑 n_frames 帧，每帧之间推进 steps_per_frame 个物理步，返回 RGB 帧列表。

    controller(model, data, t) 可选：每个物理步前调用，用来写 data.ctrl。
    track 可选：函数 (model, data) -> np.ndarray(3)，让相机跟随某个点（运镜）。
    scene_option 可选：MjvOption，用来开关接触点/接触力/关节坐标等可视化元素。
    """
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = camera if camera is not None else make_camera()
    frames = []
    dt = model.opt.timestep
    for i in range(n_frames):
        for _ in range(steps_per_frame):
            if controller is not None:
                controller(model, data, i * steps_per_frame * dt)
            mujoco.mj_step(model, data)
        if track is not None:
            cam.lookat[:] = track(model, data)
        renderer.update_scene(data, camera=cam, scene_option=scene_option)
        frames.append(renderer.render())
    renderer.close()
    return frames


def render_once(model, data, *, width: int = 640, height: int = 480,
                camera=None, scene_option: mujoco.MjvOption | None = None) -> np.ndarray:
    """渲染当前这一帧（不推进仿真），最常用的"截图"操作。"""
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=-1 if camera is None else camera,
                          scene_option=scene_option)
    frame = renderer.render()
    renderer.close()
    return frame


def vis_option(**flags) -> mujoco.MjvOption:
    """
    构造一个 MjvOption，用来开关可视化元素。

    用法：vis_option(contact_point=True, contact_force=True, joint=True)
    可用的名字就是 mjtVisFlag 去掉 mjVIS_ 前缀的小写形式，例如
      contact_point / contact_force / joint / com / inertia / camera / activation ...
    """
    opt = mujoco.MjvOption()
    for name, on in flags.items():
        # 枚举名不带下划线：contact_point -> mjVIS_CONTACTPOINT
        flag = getattr(mujoco.mjtVisFlag,
                       "mjVIS_" + name.upper().replace("_", ""))
        opt.flags[flag] = bool(on)
    return opt


def save_gif(frames: list[np.ndarray], name: str, fps: int = 30) -> pathlib.Path:
    """
    把 RGB 帧存成 GIF。体积小、无依赖，双击就能看。

    注意：PIL 保存 GIF 时会把「连续完全相同的帧」合并成一帧，但会丢掉累积时长
    （静止画面会瞬间播完）。所以这里自己做一次游程编码，把相同帧的时长相加。
    """
    path = OUT / f"{name}.gif"
    frame_ms = int(1000 / fps)
    keep: list[np.ndarray] = []
    durations: list[int] = []
    for frame in frames:
        if keep and np.array_equal(keep[-1], frame):
            durations[-1] += frame_ms
        else:
            keep.append(frame)
            durations.append(frame_ms)
    imgs = [Image.fromarray(f).convert("P", palette=Image.Palette.ADAPTIVE) for f in keep]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=durations, loop=0, optimize=True)
    total_s = sum(durations) / 1000.0
    print(f"  [gif] {path}  ({path.stat().st_size / 1e6:.2f} MB, "
          f"{len(frames)} 帧 → {len(keep)} 个关键帧, {total_s:.1f}s)")
    return path


def save_png(frame: np.ndarray, name: str) -> pathlib.Path:
    path = OUT / f"{name}.png"
    Image.fromarray(frame).save(path)
    print(f"  [png] {path}")
    return path


# --------------------------------------------------------------------------- #
# 2. 交互式 viewer（需要桌面；macOS 上还必须用 mjpython 启动）
# --------------------------------------------------------------------------- #
def is_under_mjpython() -> bool:
    """mjpython 的启动器会设置 MJPYTHON_BIN 环境变量。"""
    return bool(os.environ.get("MJPYTHON_BIN"))


def find_mjpython() -> str | None:
    """找 mjpython 可执行文件（--user 安装时它通常不在 PATH 里）。"""
    major, minor = sys.version_info[:2]
    candidates = [
        shutil.which("mjpython"),
        str(pathlib.Path(sysconfig.get_path("scripts")) / "mjpython"),
        str(pathlib.Path.home() / "Library" / "Python" / f"{major}.{minor}"
            / "bin" / "mjpython"),
        str(pathlib.Path(sys.executable).parent / "mjpython"),
    ]
    for cand in candidates:
        if cand and pathlib.Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def relaunch_under_mjpython() -> None:
    """
    macOS 上 launch_passive 要求进程由 mjpython 启动（让 Cocoa 主线程空闲）。

    这里在需要时自动用 mjpython 重新执行当前脚本；成功的话本进程会被替换，
    不会返回。非 macOS / 已经是 mjpython 时什么都不做。
    """
    if sys.platform != "darwin" or is_under_mjpython():
        return
    script = sys.argv[0] if sys.argv else ""
    if not script.endswith(".py") or not pathlib.Path(script).exists():
        return                      # 交互式环境/notebook 里不折腾
    exe = find_mjpython()
    if exe is None:
        print("[!] 找不到 mjpython：macOS 上交互窗口必须用它启动。")
        print("    试跑：python3 -m pip install --user mujoco   （mjpython 随包装）")
        return
    cmd = [exe, *sys.argv]
    print("[i] macOS 上交互窗口需要 mjpython，正在自动切换并重启脚本……")
    print("    " + " ".join(cmd))
    sys.stdout.flush()
    try:
        os.execv(exe, cmd)          # 不返回
    except OSError as exc:
        print(f"[!] 自动切换失败（{exc}），请手动执行：\n    {' '.join(cmd)}")


def mjpython_tip() -> str:
    """给 macOS 用户的手动启动提示。"""
    if sys.platform != "darwin":
        return "python3 <脚本名>"
    exe = find_mjpython() or "mjpython"
    script = sys.argv[0] if sys.argv and str(sys.argv[0]).endswith(".py") else "<脚本名>"
    return (f"{exe} {script}\n"
            f"    （或把 mjpython 所在目录加入 PATH：export PATH=\""
            f"$(dirname {exe}):$PATH\"）")



def run_viewer(model, data, *, controller=None, speed: float = 1.0,
               max_seconds: float | None = None) -> None:
    """
    打开 MuJoCo 原生交互窗口：鼠标左键旋转、右键平移、滚轮缩放、空格暂停。

    注意：必须 `import mujoco.viewer`，它不是 mujoco 包的顶层属性。
    """
    import mujoco.viewer  # noqa: F401  (延迟导入，避免无 GUI 环境下报错)

    dt = model.opt.timestep
    t = 0.0
    t_start = time.time()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if controller is not None:
                    controller(model, data, t)
                mujoco.mj_step(model, data)
                t += dt
                viewer.sync()
                if max_seconds is not None and t >= max_seconds:
                    break
                # 让仿真按 1x 实时速度走，否则会瞬间跑完
                target = t_start + t / speed
                remain = target - time.time()
                if remain > 0:
                    time.sleep(remain)
    except Exception as exc:  # pragma: no cover - 仅在没有 GUI 时触发
        print(f"\n[!] 无法打开交互窗口：{type(exc).__name__}: {exc}")
        if "mjpython" in str(exc):
            print("    macOS 上交互窗口必须用 mjpython 启动，而不是 python3：")
            print("    " + mjpython_tip())
        else:
            print("    这通常是没有图形界面（SSH/后台）导致的；"
                  "本脚本的 GIF 输出不受影响。")
            print("    想在桌面上看：请在你自己的 Terminal 里运行，"
                  "或直接双击 out/ 里的 GIF。")


# --------------------------------------------------------------------------- #
# 3. 小工具
# --------------------------------------------------------------------------- #
def enable_energy(model) -> None:
    """打开能量统计，之后 data.energy = [势能, 动能]。"""
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY


def total_energy(data) -> float:
    return float(data.energy[0] + data.energy[1])


PLOT_CJK_FONTS = ("PingFang SC", "Hiragino Sans GB", "Heiti SC", "Songti SC",
                  "STHeiti", "Arial Unicode MS", "Noto Sans CJK SC",
                  "Microsoft YaHei", "SimHei")


def setup_plot():
    """
    配置 matplotlib：Agg 后端（不弹窗）+ 找一个本机可用的中文字体。

    返回 (plt, 字体名或 None)。字体名是 None 时，调用方应该改用英文标题
    （否则中文会显示成方框）。这台 Mac 上一般会找到 Hiragino Sans GB。
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((n for n in PLOT_CJK_FONTS if n in available), None)
    if font:
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = [font] + list(rcParams["font.sans-serif"])
        rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    return plt, font


def plain_log_ticks(ax) -> None:
    """
    把对数轴的刻度标签改成纯文本（0.1、1、10…）。

    原因：matplotlib 默认用 mathtext 渲染 10^{-3} 这类标签，
    而中文字体往往缺少数学减号字形（会打印 "Font 'default' does not have
    a glyph for U+2212" 警告并显示占位符）。用 FuncFormatter 就绕开了。
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))
