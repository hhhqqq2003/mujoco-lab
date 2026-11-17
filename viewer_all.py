"""
viewer_all.py —— 一个 viewer 看遍项目里所有脚本的场景

项目里每个脚本都对应一个可看的场景，但它们的代码结构各不相同（有的用函数返回
model/data，有的在 main 里建模型）。这个文件把 11 个场景统一注册成一张表：

    python3 viewer_all.py              # 菜单选场景 → 开窗
    python3 viewer_all.py 5            # 直接开第 5 个场景
    python3 viewer_all.py --list       # 列出所有场景（不开窗）
    python3 viewer_all.py --gallery    # 生成全部场景的缩略图墙 out/gallery.png
    python3 viewer_all.py --png 3      # 只渲染第 3 个场景一张图

窗口里的快捷键：
    1-9       切到第 1~9 个场景        0        第 10 个
    n / ]     下一个场景               p / [    上一个场景
    鼠标      左键旋转 / 右键平移 / 滚轮缩放
    空格      暂停（viewer 自带）      关窗口    退出

macOS：交互窗口会自动用 mjpython 重启（launch_passive 的硬要求）。
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import time
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import common

HERE = pathlib.Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# 载入课程脚本（文件名以数字开头，不能直接 import）
# --------------------------------------------------------------------------- #
def load_lesson(filename: str):
    spec = importlib.util.spec_from_file_location(
        filename.replace(".py", ""), HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def with_floor(xml: str) -> str:
    """有些课程的 XML 没有地面（比如单摆只关心能量），出图时补一块地板更好看。"""
    floor = ('<geom name="ground" type="plane" size="4 4 0.1" '
             'rgba="0.25 0.28 0.32 1"/>')
    return xml.replace("<worldbody>", "<worldbody>" + floor, 1)


def cam(azimuth: float, elevation: float, distance: float,
        lookat=((0.0, 0.0, 0.5))) -> mujoco.MjvCamera:
    return common.make_camera(azimuth=azimuth, elevation=elevation,
                              distance=distance, lookat=lookat)


@dataclass
class Scene:
    title: str                    # 场景名
    source: str                   # 对应脚本
    desc: str                     # 一句话说明
    build: Callable               # () -> dict(model, data, controller, camera, vis)
    settle: float = 1.0           # 缩略图/单图前先仿真多少秒（让画面有动作）
    vis: object = None            # 生成时传 vis_option，默认在 build 里给


# --------------------------------------------------------------------------- #
# 每个场景的构造器
# --------------------------------------------------------------------------- #
def scene_01():
    """01_model_and_data.py：箱子自由落体 + 门被位置执行器来回驱动。"""
    model = mujoco.MjModel.from_xml_path(str(HERE / "01_model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def controller(m, d, t):
        d.ctrl[0] = 0.9 * np.sin(2 * np.pi * 0.25 * t)

    return dict(model=model, data=data, controller=controller,
                camera=cam(135, -18, 2.4, (-0.2, 0, 0.7)), vis=None)


def scene_02_ball():
    """02_physics_intuition.py：弹性地面上的弹跳球（solref 控制弹性）。"""
    L2 = load_lesson("02_physics_intuition.py")
    model = mujoco.MjModel.from_xml_string(
        L2.BALL_XML.format(dt=0.002, solref="0.02 0.1"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return dict(model=model, data=data, controller=None,
                camera=cam(120, -10, 2.4, (0, 0, 0.6)), vis=None)


def scene_02_pendulum():
    """02_physics_intuition.py：无阻尼单摆（能量守恒）。"""
    L2 = load_lesson("02_physics_intuition.py")
    model = mujoco.MjModel.from_xml_string(
        with_floor(L2.PENDULUM_XML.format(dt=0.002, integ="implicitfast")))
    data = mujoco.MjData(model)
    data.qpos[0] = 1.0
    mujoco.mj_forward(model, data)
    return dict(model=model, data=data, controller=None,
                camera=cam(135, -10, 2.0, (0, 0, 0.9)), vis=None)


def scene_ex_02a():
    """ex_02a_damping.py：带阻尼的单摆（能量单调指数衰减）。"""
    E = load_lesson("ex_02a_damping.py")
    model = mujoco.MjModel.from_xml_string(
        with_floor(E.PENDULUM_XML.format(damping=0.05, dt=0.002,
                                         integ="implicitfast")))
    data = mujoco.MjData(model)
    data.qpos[0] = 1.0
    mujoco.mj_forward(model, data)
    return dict(model=model, data=data, controller=None,
                camera=cam(135, -10, 2.0, (0, 0, 0.9)), vis=None)


def scene_03():
    """03_pendulum_pd.py：单摆 PD 跟踪阶梯参考轨迹（带重力前馈）。"""
    L3 = load_lesson("03_pendulum_pd.py")
    model = mujoco.MjModel.from_xml_string(L3.XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pd = L3.PDController(kp=40, kd=4.0, gravity_comp=True)

    def controller(m, d, t):
        d.ctrl[0] = pd(m, d, L3.reference(t))

    return dict(model=model, data=data, controller=controller,
                camera=cam(120, -12, 2.0, (0, 0, 1.05)), vis=None)


def scene_04():
    """04_cartpole_lqr.py：LQR 稳定倒立摆，每 3 秒被推一下。"""
    L4 = load_lesson("04_cartpole_lqr.py")
    model = mujoco.MjModel.from_xml_string(L4.XML)
    data = mujoco.MjData(model)
    data.qpos[1] = 0.15
    mujoco.mj_forward(model, data)
    A, B = L4.linearize(model, data)
    K = L4.lqr_gain(A, B, np.diag([5.0, 50.0, 0.5, 0.5]), np.array([[1.0]]))

    def controller(m, d, t):
        d.ctrl[0] = float(np.clip(float(-K[0] @ L4.state_of(d)),
                                  -L4.U_MAX, L4.U_MAX))
        # 周期性扰动，方便看恢复过程
        d.qfrc_applied[0] = 18.0 if (t % 3.0) < 0.08 else 0.0

    return dict(model=model, data=data, controller=controller,
                camera=cam(100, -10, 2.6, (0, 0, 0.45)), vis=None)


def scene_05():
    """05_arm_ik.py：两连杆机械臂用雅可比 IK 跟踪 mocap 目标点。"""
    L5 = load_lesson("05_arm_ik.py")
    model = mujoco.MjModel.from_xml_string(L5.XML)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q_des = data.qpos.copy()
    counter = {"n": 0}

    def controller(m, d, t):
        counter["n"] += 1
        if counter["n"] % 2 == 0:            # 500 Hz 控制
            tgt = L5.target_position(t)
            q_des[:] = L5.solve_ik(m, ik_data, tgt, q_des)
            d.mocap_pos[0] = tgt
            tau = d.qfrc_bias + L5.KP * (q_des - d.qpos) - L5.KD * d.qvel
            d.ctrl[:] = np.clip(tau, -L5.TAU_LIMIT, L5.TAU_LIMIT)

    return dict(model=model, data=data, controller=controller,
                camera=cam(150, -6, 1.5, (0.05, 0, 0.6)), vis=None)


def scene_06():
    """06_quadruped_gait.py：四足原地 trot（对角腿同相）。"""
    L6 = load_lesson("06_quadruped_gait.py")
    model, data = L6.make_data(L6.KNEE0)
    L6.settle(model, data, L6.KNEE0)

    def controller(m, d, t):
        L6.trot_control(m, d, t, L6.KNEE0)

    return dict(model=model, data=data, controller=controller,
                camera=cam(120, -16, 0.95, (0, 0, 0.22)), vis=None)


def scene_07():
    """07_cem_learning.py：用 CEM 学出来的线性策略稳定倒立摆。"""
    L7 = load_lesson("07_cem_learning.py")
    model = mujoco.MjModel.from_xml_string(L7.XML)
    data = mujoco.MjData(model)
    data.qpos[1] = 0.15
    mujoco.mj_forward(model, data)
    # 07 里一次完整 CEM 要 10 秒；viewer 直接用当时学到的增益（想重学就跑那个脚本）
    K = np.array([-2.10, -47.20, -4.80, -7.41])

    def controller(m, d, t):
        state = np.concatenate([d.qpos, d.qvel])
        d.ctrl[0] = float(np.clip(float(-K @ state), -L7.U_MAX, L7.U_MAX))

    return dict(model=model, data=data, controller=controller,
                camera=cam(100, -10, 2.6, (0, 0, 0.45)), vis=None)


def scene_09():
    """09_visualize.py：四足 + 接触点/接触力/关节/质心叠加层。"""
    L6 = load_lesson("06_quadruped_gait.py")
    model, data = L6.make_data(L6.KNEE0)
    L6.settle(model, data, L6.KNEE0)

    def controller(m, d, t):
        L6.trot_control(m, d, t, L6.KNEE0)

    vis = common.vis_option(contact_point=True, contact_force=True,
                            joint=True, com=True)
    return dict(model=model, data=data, controller=controller,
                camera=cam(125, -12, 0.8, (0.02, 0, 0.14)), vis=vis)


def scene_ex_02b():
    """ex_02b_air_drag.py：三球对比（无阻力 / 线性 / 二次空气阻力）。"""
    E = load_lesson("ex_02b_air_drag.py")
    model, data = E.build()
    mujoco.mj_forward(model, data)

    def controller(m, d, t):
        E.apply_quadratic_drag(m, d)

    return dict(model=model, data=data, controller=controller,
                camera=cam(120, -10, 4.0, (0, 0, 1.0)), vis=None)


def scene_ex_02c():
    """ex_02c_damper_actuator.py：用 <damper> 执行器做二次空气阻力的小球。"""
    E = load_lesson("ex_02c_damper_actuator.py")
    model = mujoco.MjModel.from_xml_string(E.BALL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return dict(model=model, data=data,
                controller=None,          # 阻力已经写在 XML 的 damper 执行器里
                camera=cam(120, -10, 3.2, (0, 0, 1.1)), vis=None)


SCENES: list[Scene] = [
    Scene("模型与数据（箱子 + 门）", "01_model_and_data.py",
          "自由落体的箱子 + 位置执行器驱动的门", scene_01),
    Scene("弹跳球（接触弹性）", "02_physics_intuition.py",
          "solref 阻尼比控制弹性，观察每次弹起高度", scene_02_ball, settle=0.35),
    Scene("单摆（能量守恒）", "02_physics_intuition.py",
          "无阻尼：机械能在势能/动能之间来回转换", scene_02_pendulum, settle=0.4),
    Scene("阻尼单摆（能量衰减）", "ex_02a_damping.py",
          "damping=0.05：能量单调指数衰减", scene_ex_02a, settle=0.4),
    Scene("单摆 PD 控制", "03_pendulum_pd.py",
          "PD + 重力前馈跟踪参考角（0.9 / -0.6 rad）", scene_03),
    Scene("小车倒立摆 LQR", "04_cartpole_lqr.py",
          "LQR 平衡，每 3 秒被推一下", scene_04),
    Scene("机械臂雅可比 IK", "05_arm_ik.py",
          "末端跟踪竖直平面里的红色目标点", scene_05),
    Scene("四足原地 trot", "06_quadruped_gait.py",
          "对角腿同相踏步，四只脚交替腾空", scene_06),
    Scene("CEM 学到的倒立摆策略", "07_cem_learning.py",
          "无梯度优化学出来的线性反馈 K", scene_07),
    Scene("四足 + 接触力叠加", "09_visualize.py",
          "接触点/接触力/关节轴/质心全部打开", scene_09),
    Scene("空气阻力三球对比", "ex_02b_air_drag.py",
          "红=无阻力，黄=线性阻尼，蓝=二次空气阻力", scene_ex_02b, settle=0.5),
    Scene("damper 执行器做二次阻力", "ex_02c_damper_actuator.py",
          "阻力写在 XML 的 <damper> 里，落到终端速度", scene_ex_02c, settle=0.45),
]


# --------------------------------------------------------------------------- #
# 展示
# --------------------------------------------------------------------------- #
def list_scenes() -> None:
    print(f"共 {len(SCENES)} 个场景：")
    for i, s in enumerate(SCENES, start=1):
        key = str(i) if i <= 10 else "n"
        print(f"  {key:>2}. {s.title:<26} ← {s.source}")
        print(f"      {s.desc}")


FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _cjk_font(size: int):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def brighten(model) -> None:
    """自由相机的头灯默认很暗，出缩略图时适当提亮（不改物理，只改渲染）。"""
    try:
        model.vis.headlight.ambient[:] = 0.35
        model.vis.headlight.diffuse[:] = 0.55
    except Exception:
        pass


def render_scene_png(index: int, width: int = 640, height: int = 480,
                     settle_time: float | None = None):
    """渲染单个场景（先仿真一会儿，让画面有动作）。"""
    s = SCENES[index]
    settle_time = s.settle if settle_time is None else settle_time
    cfg = s.build()
    model, data = cfg["model"], cfg["data"]
    brighten(model)
    dt = model.opt.timestep
    for i in range(int(settle_time / dt)):
        if cfg["controller"] is not None:
            cfg["controller"](model, data, i * dt)
        mujoco.mj_step(model, data)
    return common.render_once(model, data, width=width, height=height,
                              camera=cfg["camera"], scene_option=cfg["vis"])


def render_gallery(cols: int = 4, tile_w: int = 400, tile_h: int = 300,
                   settle_time: float = 1.0):
    """生成全部场景的缩略图墙（带中文标签）。"""
    rows = (len(SCENES) + cols - 1) // cols
    cap_h = 52
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + cap_h)),
                      (18, 20, 24))
    draw = ImageDraw.Draw(sheet)
    font_title = _cjk_font(17)
    font_sub = _cjk_font(13)

    for i, scene in enumerate(SCENES):
        print(f"  渲染 {i + 1}/{len(SCENES)}：{scene.title}")
        frame = render_scene_png(i, width=tile_w, height=tile_h,
                                 settle_time=settle_time or SCENES[i].settle)
        col, row = i % cols, i // cols
        x, y = col * tile_w, row * (tile_h + cap_h)
        sheet.paste(Image.fromarray(frame), (x, y))
        key = str(i + 1) if i < 10 else "-"
        draw.text((x + 8, y + tile_h + 6), f"[{key}] {scene.title}",
                  font=font_title, fill=(240, 240, 240))
        draw.text((x + 8, y + tile_h + 28), scene.source,
                  font=font_sub, fill=(140, 200, 240))

    path = common.OUT / "gallery.png"
    sheet.save(path)
    print(f"\n  [png] {path}  ({len(SCENES)} 个场景，{sheet.width}x{sheet.height})")
    return path


def print_scene_header(index: int) -> None:
    s = SCENES[index]
    print(f"\n[{index + 1}/{len(SCENES)}] {s.title}")
    print(f"  来源：{s.source}")
    print(f"  说明：{s.desc}")
    print("  切场景：1-9（0 是第 10 个） · n/] 下一个 · p/[ 上一个 · 关窗退出\n")


def run_interactive(start: int = 0) -> None:
    """开一个窗口，键盘切换所有场景（macOS 会自动改用 mjpython 重启）。"""
    common.relaunch_under_mjpython()
    import mujoco.viewer

    idx = start
    print(f"\n交互 viewer 启动（共 {len(SCENES)} 个场景）")
    while True:
        scene = SCENES[idx]
        cfg = scene.build()
        model, data = cfg["model"], cfg["data"]
        brighten(model)
        request = {"next": None}

        def key_callback(keycode, _idx=idx, _req=request):
            ch = chr(keycode) if 32 <= keycode < 127 else ""
            if ch.isdigit():
                n = int(ch)
                target = 9 if n == 0 else n - 1
                if 0 <= target < len(SCENES):
                    _req["next"] = target
            elif ch in ("n", "]", "."):
                _req["next"] = (_idx + 1) % len(SCENES)
            elif ch in ("p", "[", ","):
                _req["next"] = (_idx - 1) % len(SCENES)

        print_scene_header(idx)
        t0 = time.time()
        t = 0.0
        with mujoco.viewer.launch_passive(model, data,
                                          key_callback=key_callback) as v:
            c = cfg["camera"]
            v.cam.azimuth = c.azimuth
            v.cam.elevation = c.elevation
            v.cam.distance = c.distance
            v.cam.lookat[:] = c.lookat
            if cfg["vis"] is not None:
                v.opt.flags[:] = cfg["vis"].flags
            while v.is_running():
                if cfg["controller"] is not None:
                    cfg["controller"](model, data, t)
                mujoco.mj_step(model, data)
                v.sync()
                t += model.opt.timestep
                remain = t0 + t - time.time()
                if remain > 0:
                    time.sleep(remain)
                if request["next"] is not None:
                    break

        if request["next"] is None:
            print("窗口已关闭，退出。")
            return
        idx = request["next"]


def main() -> None:
    parser = argparse.ArgumentParser(description="一个 viewer 看遍所有脚本的场景")
    parser.add_argument("scene", nargs="?", type=int,
                        help="直接打开第 N 个场景（1 开始）")
    parser.add_argument("--list", action="store_true", help="列出所有场景")
    parser.add_argument("--gallery", action="store_true",
                        help="生成全部场景的缩略图墙 out/gallery.png")
    parser.add_argument("--png", type=int, metavar="N",
                        help="只渲染第 N 个场景为 PNG")
    args = parser.parse_args()

    if args.list:
        list_scenes()
        return
    if args.gallery:
        render_gallery()
        return
    if args.png:
        idx = max(0, min(args.png, len(SCENES)) - 1)
        frame = render_scene_png(idx)
        common.save_png(frame, f"scene_{idx + 1:02d}")
        return

    if args.scene:
        idx = max(0, min(args.scene, len(SCENES)) - 1)
    else:
        list_scenes()
        try:
            raw = input(f"\n选择场景（1-{len(SCENES)}，回车 = 1）：").strip()
            idx = (int(raw) - 1) if raw else 0
            idx = max(0, min(idx, len(SCENES) - 1))
        except (EOFError, ValueError):
            idx = 0
    run_interactive(idx)


if __name__ == "__main__":
    main()
