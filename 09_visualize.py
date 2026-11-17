"""
09 —— 可视化手册：把 MuJoCo 仿真“看见”的四种方式

同一个仿真，可以有不同层级的可视化手段。这一课把它们一次讲清：

  A. 交互式窗口（实时看、拖拽视角、点开可视化开关）—— 需要你在桌面的 Terminal 里跑
  B. 离屏渲染出单帧 / 多机位拼图 / 跟踪相机 GIF —— 不需要窗口，后台也能出图
  C. 深度图 + 分割图 —— 做感知/视觉 RL 时的“传感器图像”
  D. 可视化叠加层：接触点、接触力、关节坐标、质心 —— 排查问题的利器
  E. 第一人称相机（用 MjSpec 在运行时给机器人装摄像头）

跑法：
  python3 09_visualize.py            # 跑 B/C/D/E 全部离屏演示（约 6 秒）
  python3 09_visualize.py --viewer   # 打开交互窗口（需要桌面环境）
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib

import mujoco
import numpy as np
from PIL import Image

import common

HERE = pathlib.Path(__file__).resolve().parent


def load_lesson(filename: str):
    """加载 06_quadruped_gait.py 这种以数字开头的模块。"""
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""),
                                                  HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_scene():
    """用第 6 课的四足场景：有接触、有多个几何体，最适合演示可视化。"""
    L6 = load_lesson("06_quadruped_gait.py")
    model, data = L6.make_data(L6.KNEE0)
    L6.settle(model, data, L6.KNEE0)
    return L6, model, data


def look_quat(direction) -> np.ndarray:
    """
    返回一个四元数，让相机的"视线"（相机坐标系 -z 轴）指向 direction（世界系）。

    MuJoCo 相机默认看向自己的 -z 方向；这里用 look-at 的方式构造旋转矩阵后转四元数。
    注意 mju_mat2Quat 需要的是「行主序」展平的 3×3 矩阵。
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    z_cam = -d                                  # 相机 z 轴指向视线反方向
    x_cam = np.cross([0.0, 0.0, 1.0], z_cam)
    x_cam /= np.linalg.norm(x_cam)              # 相机 x 轴保持水平
    y_cam = np.cross(z_cam, x_cam)              # 相机 y 轴（画面"上"）
    R = np.column_stack([x_cam, y_cam, z_cam])  # 列 = 相机各轴在世界系的方向
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat


def advance(L6, model, data, seconds: float) -> None:
    """让四足原地 trot 一段时间，这样截出来的图里有动作、有接触。"""
    for _ in range(int(seconds / model.opt.timestep)):
        L6.trot_control(model, data, data.time - 0.4, L6.KNEE0)
        mujoco.mj_step(model, data)


# --------------------------------------------------------------------------- #
# B1. 最基础：离屏渲染一帧
# --------------------------------------------------------------------------- #
def demo_single_frame(L6, model, data) -> None:
    common.banner("B1. 离屏渲染一帧：mujoco.Renderer（不需要窗口）")
    frame = common.render_once(
        model, data, width=640, height=480,
        camera=common.make_camera(azimuth=120, elevation=-16,
                                  distance=0.95, lookat=(0, 0, 0.22)))
    common.save_png(frame, "09_frame")
    print("  三行核心代码就是：")
    print("    renderer = mujoco.Renderer(model, height, width)")
    print("    renderer.update_scene(data, camera=-1)   # -1 = 自由相机")
    print("    pixels = renderer.render()               # numpy (h, w, 3) uint8")


# --------------------------------------------------------------------------- #
# B2. 多机位拼图：同一时刻四个角度
# --------------------------------------------------------------------------- #
def demo_multicam(L6, model, data) -> None:
    common.banner("B2. 多机位：MjvCamera 就是 azimuth / elevation / distance / lookat")
    views = [("前", 0), ("右", 90), ("后", 180), ("左", 270)]
    tiles = []
    for label, az in views:
        cam = common.make_camera(azimuth=az, elevation=-18,
                                 distance=0.95, lookat=(0, 0, 0.22))
        tiles.append(common.render_once(model, data, width=480, height=360,
                                        camera=cam))
        print(f"  机位「{label}」: azimuth={az:>3}°  elevation=-18°")
    sheet = Image.new("RGB", (480 * 2, 360 * 2))
    for k, tile in enumerate(tiles):
        sheet.paste(Image.fromarray(tile), ((k % 2) * 480, (k // 2) * 360))
    path = common.OUT / "09_multicam.png"
    sheet.save(path)
    print(f"  [png] {path}  ← 四个角度拼成一张图")
    print("  提示：azimuth 每加 90° 就绕一圈 1/4，想要俯视就把 elevation 调到 -60°。")


# --------------------------------------------------------------------------- #
# B3. 跟踪相机：镜头始终跟着机器人
# --------------------------------------------------------------------------- #
def demo_tracking(L6, model, data) -> None:
    common.banner("B3. 跟踪相机：把相机绑在刚体上")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    cam.azimuth = 135
    cam.elevation = -18
    cam.distance = 0.95
    print(f"  跟踪 body「torso」(id={cam.trackbodyid}) —— 机器人走到哪镜头跟到哪")

    # 先跑起来，让机器人有点位移（原地 trot 也会前后漂一点点）
    frames = common.render_frames(
        model, data, n_frames=60, steps_per_frame=20, camera=cam,
        controller=lambda m, d, t: L6.trot_control(m, d, t - 0.4, L6.KNEE0),
        width=480, height=360)
    common.save_gif(frames, "09_tracking", fps=30)


# --------------------------------------------------------------------------- #
# C. 深度图 + 分割图：把仿真当“相机传感器”用
# --------------------------------------------------------------------------- #
def demo_depth_segmentation(L6, model, data) -> None:
    common.banner("C. 深度图与分割图：视觉感知的“真值标签”")
    cam = common.make_camera(azimuth=120, elevation=-25,
                             distance=1.1, lookat=(0, 0, 0.2))
    renderer = mujoco.Renderer(model, height=360, width=480)

    # 1) 普通 RGB
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()

    # 2) 深度图：每个像素到相机的距离（单位 m，float32）
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam)
    depth = renderer.render()
    renderer.disable_depth_rendering()

    # 3) 分割图：每个像素的 (物体 id, 物体类型)，int32
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    renderer.close()

    print(f"  RGB : shape={rgb.shape}, dtype={rgb.dtype}")
    print(f"  深度: shape={depth.shape}, dtype={depth.dtype}, "
          f"范围 {depth.min():.2f} ~ {depth.max():.2f} m")
    print(f"  分割: shape={seg.shape}，channel 0 = 物体 id，channel 1 = 物体类型")
    print("  分割图里出现过的物体（可直接当语义分割标签用）：")
    for obj_id, obj_type in np.unique(seg.reshape(-1, 2), axis=0):
        if obj_id < 0:
            continue
        name = mujoco.mj_id2name(model, int(obj_type), int(obj_id))
        if not name and int(obj_type) == int(mujoco.mjtObj.mjOBJ_GEOM):
            bid = int(model.geom(int(obj_id)).bodyid[0])
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            name = f"(属于 body {body})"
        tname = mujoco.mjtObj(int(obj_type)).name.replace("mjOBJ_", "").lower()
        print(f"    id={obj_id:>3} type={tname:<8} name={name or '(无名)'}")

    # 组合成一张对比图
    try:
        plt, font = common.setup_plot()
        tr = (lambda zh, en: zh) if font else (lambda zh, en: en)

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        im = axes[1].imshow(np.clip(depth, 0, 3.0), cmap="turbo")
        axes[1].set_title(tr("深度图 (m，裁剪到 3 m)", "depth (m, clipped to 3m)"))
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)

        # 给每个 (id, type) 一个随机颜色
        rng = np.random.default_rng(0)
        palette = {(-1, -1): np.zeros(3)}
        colors = np.zeros((*seg.shape[:2], 3), dtype=np.uint8)
        for obj_id, obj_type in np.unique(seg.reshape(-1, 2), axis=0):
            key = (int(obj_id), int(obj_type))
            if key not in palette:
                palette[key] = rng.integers(60, 255, size=3) / 255.0
            mask = np.all(seg == [obj_id, obj_type], axis=-1)
            colors[mask] = (palette[key] * 255).astype(np.uint8)
        axes[2].imshow(colors)
        axes[2].set_title(tr("分割图 (按物体上色)", "segmentation"))
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        path = common.OUT / "09_depth_segmentation.png"
        fig.savefig(path, dpi=120)
        print(f"  [png] {path}")
    except ImportError:
        print("  （没有 matplotlib，跳过组合图）")
    print("""
  这有什么用：
    * 深度图 → 点云重建、避障；分割图 → 实例分割的完美标签（仿真免费给真值）
    * 做视觉强化学习时，这两张图就是 policy 的输入，不需要手工标注""")

    return rgb, depth, seg


# --------------------------------------------------------------------------- #
# D. 可视化叠加层：接触点 / 接触力 / 关节 / 质心
# --------------------------------------------------------------------------- #
def demo_overlays(L6, model, data) -> None:
    common.banner("D. 叠加层：接触点 / 接触力 / 关节坐标 / 质心")
    opt = common.vis_option(contact_point=True, contact_force=True,
                            joint=True, com=True)
    print("  当前帧接触点数量 ncon =", data.ncon)

    # 拍近景，把脚部的接触力和接触点拍清楚
    cam = common.make_camera(azimuth=125, elevation=-12,
                             distance=0.8, lookat=(0.02, 0, 0.14))
    frame = common.render_once(model, data, width=640, height=480,
                               camera=cam, scene_option=opt)
    common.save_png(frame, "09_contacts")
    print("""
  这些开关就是 viewer 左侧 UI 里那些复选框，代码里通过 MjvOption 控制：
    opt = common.vis_option(contact_point=True, contact_force=True, joint=True, com=True)
    renderer.update_scene(data, camera=cam, scene_option=opt)

  可用的可视化项（对应 mjtVisFlag 去掉 mjVIS_ 前缀）：
    contact_point  接触点     contact_force  接触力箭头
    joint          关节轴     com            质心
    inertia        惯量椭球   camera         相机视锥
    activation     执行器激活  tendon         肌腱
    """
          "")


# --------------------------------------------------------------------------- #
# E. 第一人称相机：用 MjSpec 在运行时给机器人"装摄像头"
# --------------------------------------------------------------------------- #
def demo_fpv(L6) -> None:
    common.banner("E. 第一人称视角：MjSpec 运行时改模型（加相机 / 车头灯 / 障碍物）")
    spec = mujoco.MjSpec.from_string(L6.build_xml())

    # 1) 在场景里摆几个障碍物（纯视觉用的方块，机器人不会往前走太远）
    obstacles = [(0.9, 0.15, 0.12, (0.90, 0.30, 0.25)),
                 (1.6, -0.20, 0.20, (0.30, 0.70, 0.40)),
                 (2.4, 0.10, 0.30, (0.35, 0.50, 0.85))]
    for i, (x, y, size, rgb) in enumerate(obstacles):
        body = spec.worldbody.add_body()
        body.pos = np.array([x, y, size])
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = np.array([size, size, size])
        geom.rgba = np.array([*rgb, 1.0])
        geom.name = f"obstacle_{i}"

    # 2) 给躯干挂一个"眼睛"：位置在躯干前上方，朝前下方看
    torso = spec.body("torso")
    cam = torso.add_camera()
    cam.name = "fpv"
    cam.pos = np.array([0.24, 0.0, 0.10])
    cam.fovy = 80.0
    cam.quat = look_quat([1.0, 0.0, -0.35])

    # 3) 顺手装一个"车头灯"（聚光灯），否则第一人称画面只有黑地板
    lamp = torso.add_light()
    lamp.name = "headlamp"
    lamp.pos = np.array([0.20, 0.0, 0.04])
    lamp.dir = np.array([1.0, 0.0, -0.35]) / np.linalg.norm([1.0, 0.0, -0.35])
    lamp.type = mujoco.mjtLightType.mjLIGHT_SPOT
    lamp.cutoff = 70.0
    lamp.exponent = 1.5
    lamp.intensity = 4.0
    lamp.range = 8.0

    model = spec.compile()
    data = mujoco.MjData(model)
    knee0 = L6.KNEE0
    data.qpos[2] = L6.stand_height(knee0)
    for n in L6.LEGS:
        data.qpos[model.joint(f"knee_{n}_j").qposadr[0]] = knee0
    mujoco.mj_forward(model, data)
    L6.settle(model, data, knee0)
    print("  MjSpec 干的三件事：加 3 个障碍物、给 torso 装相机 fpv、装聚光灯 headlamp")

    frames = common.render_frames(
        model, data, n_frames=60, steps_per_frame=20, camera="fpv",
        controller=lambda m, d, t: L6.trot_control(m, d, t - 0.4, knee0),
        width=480, height=360)
    common.save_gif(frames, "09_fpv", fps=30)
    print("  → 画面随身体一起俯仰/晃动，前面能看到三个障碍物 —— 这就是机器人视角")


# --------------------------------------------------------------------------- #
# A. 交互式窗口
# --------------------------------------------------------------------------- #
def demo_viewer() -> None:
    L6, model, data = build_scene()
    common.relaunch_under_mjpython()   # macOS：自动改用 mjpython 启动
    import mujoco.viewer
    import time

    print("""
  ================== 交互式窗口操作指南 ==================
  鼠标左键拖动  旋转视角        鼠标右键拖动  平移
  滚轮          缩放            双击          选中物体（看局部坐标）
  空格          暂停 / 继续     Backspace     重置到初始状态
  左侧 UI       可视化开关（接触力、关节、质心、帧率……）
  右侧 UI       仿真设置（积分器、求解器、实时倍率……）
  ========================================================
""")
    t0 = time.time()
    t = 0.0
    try:
        with mujoco.viewer.launch_passive(model, data) as v:
            # 用代码直接打开几个可视化开关（等价于在左侧 UI 里打勾）
            opt = common.vis_option(contact_point=True, contact_force=True,
                                    joint=True, com=True)
            v.opt.flags[:] = opt.flags
            while v.is_running():
                L6.trot_control(model, data, t)
                mujoco.mj_step(model, data)
                v.sync()
                t += model.opt.timestep
                remain = t0 + t - time.time()
                if remain > 0:
                    time.sleep(remain)
    except Exception as exc:
        print(f"[!] 无法打开窗口：{type(exc).__name__}: {exc}")
        print("    请在你自己的桌面 Terminal 里运行：python3 09_visualize.py --viewer")
        print("    没有窗口也不影响：离屏渲染出的 PNG/GIF 在 out/ 目录里。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="打开交互式窗口")
    args = parser.parse_args()

    if args.viewer:
        demo_viewer()
        return

    common.banner("可视化手册：全部离屏演示")
    L6, model, data = build_scene()
    advance(L6, model, data, 1.0)     # 先让机器人动起来，截图才好看

    demo_single_frame(L6, model, data)
    demo_multicam(L6, model, data)
    demo_tracking(L6, model, data)
    demo_depth_segmentation(L6, model, data)
    demo_overlays(L6, model, data)
    demo_fpv(L6)

    common.banner("下一步")
    print("  1. 实时交互：python3 09_visualize.py --viewer（或 08_play.py）")
    print("  2. 在 Jupyter/ipython 里内嵌图像：")
    print("""
         import mujoco, io, base64
         from IPython.display import HTML, display
         from PIL import Image
         r = mujoco.Renderer(model, 240, 320)
         r.update_scene(data, camera=-1)
         buf = io.BytesIO(); Image.fromarray(r.render()).save(buf, format="PNG")
         display(HTML('<img src="data:image/png;base64,%s">'
                      % base64.b64encode(buf.getvalue()).decode()))
""")
    print("  3. 想要 mp4 而不是 GIF：装 imageio[ffmpeg]，把帧列表交给 imageio.mimsave('x.mp4')")
    print("  4. 想看所有产物：open out/")


if __name__ == "__main__":
    main()
