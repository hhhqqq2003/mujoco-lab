"""
view_xml.py —— 通用 MJCF 查看器：改完 XML 立刻看效果

这是"学 MuJoCo 最舒服的工作流"：编辑 XML → 保存 → 看窗口，不用改任何 Python 代码。

用法：
  python3 view_xml.py                      # 看默认的 01_model.xml（第 1 课的模型）
  python3 view_xml.py 我的机器人.xml         # 看任意 MJCF 文件
  python3 view_xml.py robot.xml --png      # 只存一张截图（不用开窗）
  python3 view_xml.py robot.xml --gif 4    # 录 4 秒 GIF（不用开窗）
  python3 view_xml.py robot.xml --info     # 打印模型结构（学习 XML 元素对应哪些字段）
  python3 view_xml.py robot.xml --contacts # 开窗并显示接触点/接触力
  python3 view_xml.py robot.xml --no-physics  # 只看静态模型（适合纯外观检查）

相机默认自动取景（按模型的几何包围盒算），也可以用 --az --el --dist --lookat 手动指定。

macOS 注意：交互窗口必须由 mjpython 启动，本脚本会自动切换（不用你操心）。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

import common

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_XML = HERE / "01_model.xml"


# --------------------------------------------------------------------------- #
# 加载与自动取景
# --------------------------------------------------------------------------- #
def resolve_path(arg: str) -> pathlib.Path:
    """先按当前目录找，再按脚本目录找，都找不到就报错。"""
    p = pathlib.Path(arg).expanduser()
    for cand in (p, HERE / p):
        if cand.is_file():
            return cand.resolve()
    print(f"[!] 找不到文件：{arg}")
    print(f"    （找过：{p} 和 {HERE / p}）")
    raise SystemExit(1)


def auto_camera(model, data, azimuth: float = 135.0,
                elevation: float = -18.0) -> mujoco.MjvCamera:
    """
    按几何体的包围盒自动取景。

    跳过 plane（无限大，会把包围盒撑爆），用剩下的 geom 中心和半径算相机距离。
    """
    mujoco.mj_forward(model, data)
    pts, radii = [], []
    for i in range(model.ngeom):
        gtype = int(np.ravel(model.geom(i).type)[0])
        if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
            # 地面参与定中心（否则相机只盯着空中的物体），但不参与算半径
            pts.append(np.array(data.geom(i).xpos, dtype=float))
            continue
        pts.append(np.array(data.geom(i).xpos, dtype=float))
        radii.append(float(np.ravel(model.geom(i).rbound)[0]))
    cam = mujoco.MjvCamera()
    cam.azimuth = azimuth
    cam.elevation = elevation
    if pts:
        pts = np.array(pts)
        radii = np.array(radii)
        center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        half_extent = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) / 2
        cam.lookat[:] = center
        cam.distance = float(max(1.0, 2.6 * (half_extent + radii.max())))
    else:
        cam.lookat[:] = (0.0, 0.0, 0.5)
        cam.distance = 3.0
    return cam


def load_model(path: pathlib.Path):
    """加载模型；出错时把 MuJoCo 的报错原样打出来（写 XML 时最常看这个）。"""
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        print(f"[!] 加载失败：{path}")
        print(f"    {type(exc).__name__}: {exc}")
        print("\n提示：XML 报错会指出行号和元素名，按行号回去改就行。")
        raise SystemExit(1)
    return model


# --------------------------------------------------------------------------- #
# 模式 1：打印模型结构（学 XML 时最有用 —— XML 元素 ↔ 模型字段）
# --------------------------------------------------------------------------- #
def model_display_name(model) -> str:
    """MuJoCo 3.17 的 MjModel 没有 .name，这里从 names 缓存里取第一个字符串。"""
    raw = getattr(model, "names", b"") or b""
    if not raw:
        return "(未命名)"
    first = raw.split(b"\x00")[0]
    return first.decode("utf-8", errors="replace") or "(未命名)"


def print_info(model, data) -> None:
    common.banner(f"模型结构：{model_display_name(model)}")
    print(f"  nbody={model.nbody} njnt={model.njnt} nq={model.nq} nv={model.nv} "
          f"nu={model.nu} ngeom={model.ngeom} nsensor={model.nsensor} "
          f"nmaterial={model.nmat} nkey={model.nkey}")
    print(f"  timestep={model.opt.timestep} "
          f"integrator={mujoco.mjtIntegrator(int(model.opt.integrator)).name} "
          f"gravity={np.round(model.opt.gravity, 2)}")
    print(f"  总质量 = {model.body_mass.sum():.3f} kg")

    common.section("body（刚体树）")
    for i in range(model.nbody):
        b = model.body(i)
        print(f"  [{i}] {b.name or '(无名)':<12} mass={float(np.ravel(b.mass)[0]):7.3f} "
              f"pos={np.round(b.pos, 3)}")

    common.section("joint（关节）")
    if model.njnt == 0:
        print("  （没有关节：这个模型是纯静态装饰）")
    for i in range(model.njnt):
        j = model.joint(i)
        print(f"  [{i}] {j.name or '(无名)':<12} "
              f"type={mujoco.mjtJoint(int(np.ravel(j.type)[0])).name.replace('mjJNT_', ''):<8} "
              f"qposadr={int(np.ravel(j.qposadr)[0])} dofadr={int(np.ravel(j.dofadr)[0])}")

    common.section("geom（几何体）")
    for i in range(model.ngeom):
        g = model.geom(i)
        print(f"  [{i}] {g.name or '(无名)':<12} "
              f"type={mujoco.mjtGeom(int(np.ravel(g.type)[0])).name.replace('mjGEOM_', ''):<8} "
              f"size={np.round(g.size, 3)}")

    common.section("actuator（执行器）")
    for i in range(model.nu):
        a = model.actuator(i)
        print(f"  [{i}] {a.name or '(无名)':<12} ctrlrange={np.round(a.ctrlrange, 2)}")

    common.section("sensor（传感器）")
    for i in range(model.nsensor):
        s = model.sensor(i)
        print(f"  [{i}] {s.name or '(无名)':<12} needstage={int(np.ravel(s.needstage)[0])}")

    common.section("camera / light（相机与灯光）")
    for i in range(model.ncam):
        c = model.camera(i)
        print(f"  camera[{i}] {c.name or '(无名)'} pos={np.round(c.pos, 3)} fovy={float(np.ravel(c.fovy)[0])}")
    for i in range(model.nlight):
        lt = model.light(i)
        print(f"  light[{i}]  {lt.name or '(无名)'} pos={np.round(lt.pos, 3)}")

    common.section("当前状态（MjData）")
    print(f"  qpos = {np.round(data.qpos, 3)}")
    print(f"  qvel = {np.round(data.qvel, 3)}")
    print("  → 想改初始状态，改 XML 里的 pos / quat / axis，而不是改这里的数值。")

    common.section("怎么改")
    print("  · 改数值：直接在 XML 里改 pos / size / mass / kp / timestep，保存后重跑本脚本")
    print("  · 加元素：复制一个 <body> 或 <geom> 块，改 name（name 不能重复）")
    print("  · 报错时：报错信息会给行号，对照着看那一段")


# --------------------------------------------------------------------------- #
# 模式 2：离屏出图 / GIF（不开窗，后台也能跑）
# --------------------------------------------------------------------------- #
def ensure_offscreen(model, width: int, height: int) -> None:
    """XML 里 <visual><global offwidth/offheight> 限制了离屏渲染尺寸，不够就调大。"""
    gw = int(np.ravel(model.vis.global_.offwidth)[0])
    gh = int(np.ravel(model.vis.global_.offheight)[0])
    if width > gw or height > gh:
        if width > gw:
            model.vis.global_.offwidth = width
        if height > gh:
            model.vis.global_.offheight = height
        print(f"  [i] 离屏帧缓冲 {gw}x{gh} 不够，已调大到 {width}x{height}")


def render_png(model, data, cam, path: pathlib.Path,
               width: int = 800, height: int = 600,
               scene_option=None) -> None:
    ensure_offscreen(model, width, height)
    frame = common.render_once(model, data, width=width, height=height,
                               camera=cam, scene_option=scene_option)
    out = common.OUT / f"{path.stem}.png"
    common.save_png(frame, path.stem)
    print(f"  模型：{path.name}  相机：az={cam.azimuth:.0f}° el={cam.elevation:.0f}° "
          f"dist={cam.distance:.2f}")


def render_gif(model, data, cam, path: pathlib.Path, seconds: float,
               scene_option=None) -> None:
    fps = 30
    steps_per_frame = max(1, round((1.0 / fps) / model.opt.timestep))
    n_frames = max(1, int(seconds * fps))
    print(f"  录制 {seconds:.1f} 秒 = {n_frames} 帧（每帧 {steps_per_frame} 个物理步）…")
    frames = common.render_frames(
        model, data, n_frames=n_frames, steps_per_frame=steps_per_frame,
        camera=cam, width=480, height=360, scene_option=scene_option)
    common.save_gif(frames, path.stem, fps=fps)


# --------------------------------------------------------------------------- #
# 模式 3：交互窗口
# --------------------------------------------------------------------------- #
def open_viewer(model, data, cam, *, contacts: bool, no_physics: bool) -> None:
    common.relaunch_under_mjpython()      # macOS：自动改用 mjpython 启动
    import mujoco.viewer
    import time

    print("""
  ============== 操作 ==============
  左键拖动 = 旋转     右键拖动 = 平移
  滚轮 = 缩放         双击 = 选中物体
  空格 = 暂停         Backspace = 重置
  左侧 UI = 接触力/关节/质心等开关
  ==================================
  关闭窗口即退出。
""")
    t0 = time.time()
    t = 0.0
    try:
        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.azimuth = cam.azimuth
            v.cam.elevation = cam.elevation
            v.cam.distance = cam.distance
            v.cam.lookat[:] = cam.lookat
            if contacts:
                opt = common.vis_option(contact_point=True, contact_force=True,
                                        joint=True, com=True)
                v.opt.flags[:] = opt.flags
            while v.is_running():
                if not no_physics:
                    mujoco.mj_step(model, data)
                    t += model.opt.timestep
                v.sync()
                remain = t0 + t - time.time()
                if remain > 0:
                    time.sleep(remain)
    except Exception as exc:      # pragma: no cover - 无 GUI 环境
        print(f"[!] 无法打开窗口：{type(exc).__name__}: {exc}")
        if "mjpython" in str(exc):
            print("    macOS 需要 mjpython 启动，手动命令：")
            print(f"    {common.mjpython_tip()}")
        print("    没有窗口也能看：加 --png 存图，或 --gif 3 录 GIF。")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="通用 MJCF 查看器：改完 XML 立刻看效果")
    parser.add_argument("xml", nargs="?", default=str(DEFAULT_XML),
                        help="MJCF 文件路径（默认 01_model.xml）")
    parser.add_argument("--info", action="store_true", help="只打印模型结构")
    parser.add_argument("--png", action="store_true", help="渲染一张 PNG（不开窗）")
    parser.add_argument("--gif", type=float, metavar="SECONDS",
                        help="录制 GIF，例如 --gif 4（不开窗）")
    parser.add_argument("--contacts", action="store_true",
                        help="开窗/出图时显示接触点/接触力/关节/质心")
    parser.add_argument("--t", type=float, default=0.0, metavar="SECONDS",
                        help="渲染前先仿真多少秒（例如 --t 1 让箱子先落地）")
    parser.add_argument("--no-physics", action="store_true",
                        help="只看静态模型，不推进物理")
    parser.add_argument("--az", type=float, default=135.0, help="相机水平角（度）")
    parser.add_argument("--el", type=float, default=-18.0, help="相机俯仰角（度）")
    parser.add_argument("--dist", type=float, help="相机距离（默认自动取景）")
    parser.add_argument("--lookat", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="相机注视点（默认自动取景）")
    args = parser.parse_args()

    path = resolve_path(args.xml)
    model = load_model(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam = auto_camera(model, data, azimuth=args.az, elevation=args.el)
    if args.dist is not None:
        cam.distance = args.dist
    if args.lookat is not None:
        cam.lookat[:] = args.lookat

    scene_option = None
    if args.contacts and (args.png or args.gif):
        scene_option = common.vis_option(contact_point=True, contact_force=True,
                                         joint=True, com=True)

    if args.info:
        print_info(model, data)
        return
    if args.t > 0:
        n = int(args.t / model.opt.timestep)
        for _ in range(n):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        print(f"  [i] 先仿真 {args.t:.2f} s（{n} 步），当前 t={data.time:.2f}s, "
              f"接触数 ncon={data.ncon}")
    if args.png:
        render_png(model, data, cam, path, scene_option=scene_option)
        return
    if args.gif:
        render_gif(model, data, cam, path, args.gif, scene_option=scene_option)
        return
    open_viewer(model, data, cam, contacts=args.contacts,
                no_physics=args.no_physics)


if __name__ == "__main__":
    main()
