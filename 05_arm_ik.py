"""
05 —— 机械臂逆运动学：用雅可比矩阵 + 阻尼最小二乘，让末端追一个移动目标

前四课是"1 个自由度"的玩具，这一课开始碰真正的机器人学：
  1. 用 mocap 刚体当"目标点"（不参与物理，纯粹用来标记位置）
  2. mj_jacSite 直接给你末端雅可比 J（3×nv），不用手推
  3. 阻尼最小二乘（DLS）迭代求解逆运动学：dq = Jᵀ(JJᵀ + λ²I)⁻¹ e
  4. 关节空间 PD + 重力前馈去跟踪 IK 算出的 q_des（第 3 课的知识直接用上）
  5. 【坑】MJCF 默认角度单位是度，写弧度要加 <compiler angle="radian"/>

跑法：python3 05_arm_ik.py [--viewer]
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

import common

XML = """
<mujoco model="planar_arm">
  <!-- 【坑】MJCF 里 range/ref 的默认角度单位是「度」！想用弧度必须显式声明 -->
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0.5 1 2.5"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.22 0.24 0.28 1"/>
    <body name="base" pos="0 0 0.15">
      <geom type="cylinder" size="0.06 0.06" contype="0" conaffinity="0"
            rgba="0.55 0.55 0.58 1"/>
      <!-- 大臂：绕 y 轴铰链，0 rad 时竖直向上 -->
      <body name="upper">
        <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.05"
               range="-2.6 2.6" limited="true"/>
        <geom name="upper_geom" type="capsule" fromto="0 0 0 0 0 0.4"
              size="0.035" mass="0.6" rgba="0.2 0.6 0.95 1"
              contype="0" conaffinity="0"/>
        <body name="fore" pos="0 0 0.4">
          <joint name="elbow" type="hinge" axis="0 1 0" damping="0.05"
                 range="-2.6 2.6" limited="true"/>
          <geom name="fore_geom" type="capsule" fromto="0 0 0 0 0 0.35"
                size="0.03" mass="0.35" rgba="0.15 0.45 0.8 1"
                contype="0" conaffinity="0"/>
          <geom name="hand" type="sphere" size="0.035" pos="0 0 0.38" mass="0.1"
                rgba="0.95 0.6 0.2 1" contype="0" conaffinity="0"/>
          <site name="ee" pos="0 0 0.38"/>
        </body>
      </body>
    </body>
    <!-- 目标点：mocap 刚体，位置由 data.mocap_pos 直接写，不参与物理 -->
    <body name="target" mocap="true" pos="0.3 0 0.6">
      <geom type="sphere" size="0.05" rgba="1 0.15 0.15 0.85"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="1" ctrlrange="-25 25"/>
    <motor name="elbow_motor" joint="elbow" gear="1" ctrlrange="-15 15"/>
  </actuator>
</mujoco>
"""

KP = np.array([120.0, 60.0])       # 关节空间 PD 增益
KD = np.array([12.0, 6.0])
TAU_LIMIT = np.array([25.0, 15.0])


def target_position(t: float) -> np.ndarray:
    """目标点在一个竖直圆上移动（相对底座，单位 m）。"""
    return np.array([0.20 + 0.10 * np.cos(1.0 * t),
                     0.0,
                     0.55 + 0.10 * np.sin(1.0 * t)])


def solve_ik(model, ik_data, target: np.ndarray, q_guess: np.ndarray,
             iters: int = 25, damping: float = 0.08, step_limit: float = 0.15):
    """
    阻尼最小二乘 IK。

    为什么需要"阻尼"？当手臂伸直（奇异位形）时，J 会降秩，
    普通伪逆 J⁺ = Jᵀ(JJᵀ)⁻¹ 会爆炸；加上 λ²I 相当于给关节速度加了"软限制"，
    保证解总是有界 —— 这就是工业机器人里最常用的 DLS / Levenberg-Marquardt 做法。

    注意这里用独立的 ik_data 试算前向运动学，绝不污染真实仿真状态。
    """
    ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    nv = model.nv
    jacp = np.zeros((3, nv))
    q = q_guess.copy()

    for _ in range(iters):
        ik_data.qpos[:] = q
        mujoco.mj_forward(model, ik_data)   # 前向运动学（jacSite 依赖 forward 的中间量）
        err = target - ik_data.site("ee").xpos
        if np.linalg.norm(err) < 1e-4:
            break
        jacp[:] = 0.0
        mujoco.mj_jacSite(model, ik_data, jacp, None, ee_site)
        J = jacp
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), err)
        q = q + np.clip(dq, -step_limit, step_limit)       # 限制单次迭代步长
    return q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="打开交互窗口实时看")
    args = parser.parse_args()

    if args.viewer:
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        ik_data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        q_des = data.qpos.copy()

        def control_loop(m, d, t):
            dj = d.mocap_pos.shape[0] if hasattr(d, "mocap_pos") else 1
            tgt = target_position(t) + np.array([0, 0, 0.15])
            d.mocap_pos[0] = tgt
            q_des[:] = solve_ik(m, ik_data, tgt, q_des)
            tau = d.qfrc_bias + KP * (q_des - d.qpos) - KD * d.qvel
            d.ctrl[:] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)

        print("打开交互窗口……关窗口结束。")
        common.run_viewer(model, data, controller=control_loop)
        return

    common.banner("机械臂跟踪移动目标：IK（雅可比）+ 关节空间 PD")

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q_des = data.qpos.copy()

    dt = model.opt.timestep
    duration, ctrl_every = 6.0, 2          # 500 Hz 控制

    ts, ee_pos, tgt_pos, q_traj, errs = [], [], [], [], []
    for step in range(int(duration / dt)):
        if step % ctrl_every == 0:
            tgt = target_position(data.time)
            q_des[:] = solve_ik(model, ik_data, tgt, q_des)
            tau = data.qfrc_bias + KP * (q_des - data.qpos) - KD * data.qvel
            data.ctrl[:] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)
            data.mocap_pos[0] = tgt
        mujoco.mj_step(model, data)

        ts.append(data.time)
        ee_pos.append(data.site("ee").xpos.copy())
        tgt_pos.append(data.mocap_pos[0].copy())
        q_traj.append(data.qpos.copy())
        errs.append(np.linalg.norm(data.site("ee").xpos - data.mocap_pos[0]))

    ts = np.array(ts)
    ee_pos, tgt_pos = np.array(ee_pos), np.array(tgt_pos)
    q_traj, errs = np.array(q_traj), np.array(errs)

    warm = ts > 0.5
    print(f"  跟踪误差（0.5 s 之后）：均值 {errs[warm].mean() * 1000:.1f} mm，"
          f"最大 {errs[warm].max() * 1000:.1f} mm")
    print(f"  肩关节角度范围：[{q_traj[:, 0].min():+.2f}, {q_traj[:, 0].max():+.2f}] rad")
    print(f"  肘关节角度范围：[{q_traj[:, 1].min():+.2f}, {q_traj[:, 1].max():+.2f}] rad")
    print("""
  解读：
    * IK 每一步只解"几何问题"（怎么让末端到目标），完全不管力、质量、重力
    * PD 负责"把关节开到 q_des"，重力前馈 qfrc_bias 负责抵消自重
    * 这两层分开是机器人控制的常见架构：任务空间规划 + 关节空间跟踪
    * 目标轨迹在圆的边缘时手臂接近伸直（奇异位形），注意误差会突然变大""")

    # ---------------- 出图 ----------------
    try:
        plt, font = common.setup_plot()
        tr = (lambda zh, en: zh) if font else (lambda zh, en: en)

        fig = plt.figure(figsize=(11, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(ee_pos[:, 0], ee_pos[:, 2], lw=2, color="#2b8bd8",
                 label=tr("末端轨迹", "end-effector"))
        ax1.plot(tgt_pos[:, 0], tgt_pos[:, 2], "r--", lw=1,
                 label=tr("目标轨迹", "target"))
        ax1.scatter([0], [0.15], c="k", s=40, marker="s")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("z (m)")
        ax1.set_title(tr("末端在竖直平面里追圆", "End-effector tracking a circle"))
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=9)
        ax1.set_aspect("equal")

        axes = fig.add_subplot(2, 2, 2)
        axes.plot(ts, q_traj[:, 0], lw=1.2, label="shoulder")
        axes.plot(ts, q_traj[:, 1], lw=1.2, label="elbow")
        axes.set_ylabel(tr("关节角 (rad)", "joint angle (rad)"))
        axes.set_title(tr("关节空间运动", "Joint-space motion"))
        axes.grid(alpha=0.3)
        axes.legend(fontsize=8)

        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(ts, errs * 1000, lw=1.2, color="#d85b2b")
        ax3.set_xlabel("time (s)")
        ax3.set_ylabel(tr("跟踪误差 (mm)", "tracking error (mm)"))
        ax3.set_title(tr("末端跟踪误差", "End-effector error"))
        ax3.grid(alpha=0.3)
        fig.tight_layout()
        path = common.OUT / "05_arm_ik.png"
        fig.savefig(path, dpi=120)
        print(f"\n  [png] {path}")
    except ImportError:
        print("\n  （没有 matplotlib，跳过曲线图）")

    # ---------------- GIF ----------------
    print("\n  正在渲染 GIF ...")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q_des = data.qpos.copy()
    model.opt.timestep = dt

    def set_ctrl(m, d, t):
        # 每帧只解一次 IK（渲染用，控制频率 100 Hz 足够顺滑）
        if int(t / dt) % ctrl_every == 0:
            tgt = target_position(t)
            q_des[:] = solve_ik(m, ik_data, tgt, q_des)
            d.mocap_pos[0] = tgt
            tau = d.qfrc_bias + KP * (q_des - d.qpos) - KD * d.qvel
            d.ctrl[:] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)

    frames = common.render_frames(
        model, data, n_frames=90, steps_per_frame=15,
        camera=common.make_camera(azimuth=150, elevation=-6,
                                  distance=1.5, lookat=(0.05, 0, 0.6)),
        controller=set_ctrl, width=480, height=360)
    common.save_gif(frames, "05_arm_ik", fps=30)

    common.banner("要点回顾")
    print("  1. mj_jacSite(model, data, jacp, None, site_id) 直接给末端雅可比")
    print("  2. DLS：dq = Jᵀ(JJᵀ + λ²I)⁻¹e，λ 是防止奇异位形爆炸的阻尼")
    print("  3. IK 只解几何；动力学靠 PD + 前馈在关节空间解决")
    print("  4. mocap 刚体是标记目标点的好工具，不参与物理")
    print("\n练习题：")
    print("  (a) 把目标轨迹半径从 0.10 改到 0.35，找到手臂伸直的奇异点，"
          "把 damping 从 0.08 调到 0.3 看区别。")
    print("  (b) 把 solve_ik 的 iters 从 25 改成 1，看跟踪误差怎么变（为什么还能动？）")
    print("  (c) 给末端加一个必须保持水平的约束（第二个任务），"
          "查一下零空间投影（null-space projection）怎么做。")


if __name__ == "__main__":
    main()
