"""
01 —— 认识 MuJoCo 的两个核心对象：MjModel 和 MjData

这是整个 MuJoCo 学习里最重要的一课。90% 的困惑都来自没分清：
  MjModel = 机器人/场景"长什么样"（质量、关节、几何、参数）—— 编译后一般不动
  MjData  = 这一瞬间"是什么状态"（位置、速度、力、接触）    —— 每一步都在变

跑法：
  python3 01_model_and_data.py            # 打印模型/状态数据（本课主体）
  python3 01_model_and_data.py --viewer   # 打开交互窗口，看这个模型动起来
  python3 view_xml.py 01_model.xml        # 通用查看器（等价，且能看任意 XML）

模型 XML 在 01_model.xml，想改直接编辑它，保存后重跑即可。
"""

from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np

import common

import sys, time

# 模型文件（XML）就在同目录的 01_model.xml 里 —— 想改就改它，改完重跑本脚本
XML_PATH = pathlib.Path(__file__).with_name("01_model.xml")


def run_viewer() -> None:
    """交互窗口：箱子自由落体，门跟着正弦目标角来回摆（看 data.ctrl 的效果）。"""
    common.relaunch_under_mjpython()     # macOS：交互窗口必须 mjpython 启动
    import mujoco.viewer
    import time

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = common.make_camera(azimuth=135, elevation=-18,
                             distance=2.4, lookat=(-0.2, 0, 0.7))

    print(f"""
  模型文件：{XML_PATH.name}
  看到的三件事：
    1. 橙色箱子从 1.5 m 自由落体，落地后停住（freejoint + 接触力）
    2. 蓝色门板跟着正弦目标角来回摆 —— 因为下面这行代码在每步改 data.ctrl
         data.ctrl[0] = 0.9 * sin(2*pi*0.25*t)
    3. 左侧 UI 里可以勾选 contact point，看到箱子与地面的接触点

  操作：左键旋转 / 右键平移 / 滚轮缩放 / 空格暂停 / 关窗口退出。
""")
    t0 = time.time()
    t = 0.0
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            data.ctrl[0] = 0.9 * np.sin(2 * np.pi * 0.25 * t)   # 门的目标角
            mujoco.mj_step(model, data)
            v.sync()
            t += model.opt.timestep
            remain = t0 + t - time.time()
            if remain > 0:
                time.sleep(remain)


def main() -> None:
    parser = argparse.ArgumentParser(description="第 1 课：MjModel 与 MjData")
    parser.add_argument("--viewer", action="store_true",
                        help="打开交互窗口看 01_model.xml")
    args = parser.parse_args()
    if args.viewer:
        run_viewer()
        return

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # =================== 第一半：MjModel 是"静态蓝图" =================== #
    common.banner("1) MjModel —— 机器人长什么样（编译后不变）")

    common.section("规模计数（记住这几个缩写，看报错时会救命）")
    print(f"  nbody   = {model.nbody:3d}  刚体个数（含 world）")
    print(f"  njnt    = {model.njnt:3d}  关节个数")
    print(f"  nq      = {model.nq:3d}  位置变量维度（广义坐标）")
    print(f"  nv      = {model.nv:3d}  速度变量维度（= 自由度，模型上 nv ≤ nq）")
    print(f"  nu      = {model.nu:3d}  执行器个数（你能控制的输入维数）")
    print(f"  ngeom   = {model.ngeom:3d}  几何体个数（碰撞/渲染都靠它）")
    print(f"  nsensor = {model.nsensor:3d}  传感器个数")
    print(f"  nmat    = {model.nmat:3d}  材料个数")

    common.section("关节表：qpos 和 qvel 的『地址』在这里分配")
    print(f"  {'id':<4}{'name':<12}{'type':<10}{'qposadr':>8}{'dofadr':>8}")
    for j in range(model.njnt):
        jt = model.joint(j)
        jtype = mujoco.mjtJoint(int(np.ravel(jt.type)[0])).name.replace("mjJNT_", "")
        print(f"  {j:<4}{jt.name:<12}{jtype:<10}{jt.qposadr[0]:>8}{jt.dofadr[0]:>8}")

    common.section("刚体质量与几何")
    for i in range(model.nbody):
        b = model.body(i)
        print(f"  body[{i}] {b.name:<10} mass={b.mass[0]:7.3f}  pos={np.round(b.pos, 2)}")
    for i in range(model.ngeom):
        g = model.geom(i)
        gtype = mujoco.mjtGeom(int(np.ravel(g.type)[0])).name.replace("mjGEOM_", "")
        print(f"  geom[{i}] {g.name:<12} type={gtype:<8} size={np.round(g.size, 3)}")

    common.section("仿真参数（都在 model.opt 里，改它就能影响数值行为）")
    print(f"  timestep   = {model.opt.timestep}   s")
    print(f"  integrator = {mujoco.mjtIntegrator(model.opt.integrator).name}")
    print(f"  gravity    = {np.round(model.opt.gravity, 3)}")
    print(f"  solver     = {mujoco.mjtSolver(model.opt.solver).name}, "
          f"iterations={model.opt.iterations}, tolerance={model.opt.tolerance}")

    # =================== 第二半：MjData 是"这一瞬间的状态" =================== #
    common.banner("2) MjData —— 这一瞬间的状态（每步都被重写）")

    common.section("初始状态")
    print(f"  qpos      = {np.round(data.qpos, 4)}   (nq={model.nq})")
    print(f"  qvel      = {np.round(data.qvel, 4)}   (nv={model.nv})")
    print(f"  ctrl      = {np.round(data.ctrl, 4)}   (nu={model.nu})")
    print(f"  sensordata= {np.round(data.sensordata, 4)}")
    print("  注意 qpos 前 7 个数属于箱子 freejoint：")
    print("    [x, y, z, qw, qx, qy, qz]  — 位置 3 个 + 四元数 4 个")
    print("  后 1 个数属于门的 hinge：整一个角度。")
    print("  => 这就是 nq=8 而 nv=7 的原因：四元数有 4 个数但只有 3 个转动自由度。")

    common.section("mj_forward vs mj_step（新手最容易搞混的一对）")
    print("  mj_forward(model, data) : 只算运动学/动力学量（位置、雅可比、力），不推进时间")
    print("  mj_step(model, data)    : 先算力，再积分推进一个 timestep —— 真正的一步仿真")
    z_before = data.qpos[2]
    data.ctrl[0] = 0.5          # 让门电机转到 0.5 rad
    for _ in range(500):
        mujoco.mj_step(model, data)
    z_after = data.qpos[2]
    print(f"\n  仿真 500 步（= {500 * model.opt.timestep:.2f} s）后：")
    print(f"    箱子高度  {z_before:.3f} -> {z_after:.3f} m（落地了）")
    print(f"    门的角度  {data.sensor('door_angle').data[0]:+.3f} rad"
          f"（目标 0.5，位置执行器有稳态误差，很正常）")
    print(f"    传感器读数 sensordata = {np.round(data.sensordata, 3)}")
    print(f"    接触点数 ncon = {data.ncon}   <- 箱子碰到地面了")
    if data.ncon:
        c = data.contact[0]
        print(f"    接触[0]: geom{c.geom1} vs geom{c.geom2}, "
              f"穿透深度 {c.dist:+.4f} m")
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, 0, force)
        print(f"    接触力(局部坐标系) = {np.round(force[:3], 2)} N")

    common.banner("要点回顾")
    print("  1. XML -> MjModel（编译一次，只读）；仿真状态全在 MjData 里")
    print("  2. 你只能通过 data.ctrl 影响仿真（以及改 qpos/qvel 做重置）")
    print("  3. 一切力、接触、雅可比都从 MjData 里读，不要自己算动力学")
    print("\n  想看这个模型长什么样？")
    print("    python3 01_model_and_data.py --viewer     # 交互窗口")
    print("    python3 view_xml.py 01_model.xml          # 通用查看器（任意 XML 都能看）")
    print("    python3 view_xml.py 01_model.xml --png    # 只存一张图")
    print("\n练习题：")
    print("  (a) 把 timestep 改成 0.02，看箱子落地时会不会穿透/爆炸？为什么？")
    print("  (b) 把 position 执行器换成 motor，data.ctrl 就变成力矩了 —— 试着手动给 0.5 N·m。")
    print("  (c) 给门加 damping=0 再看，摆动多久才停？")
    print("  改完跑：python3 02_physics_intuition.py 会看到数值差别的量化结果。")


if __name__ == "__main__":
    main()
