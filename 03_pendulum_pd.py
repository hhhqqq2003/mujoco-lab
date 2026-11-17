"""
03 —— 单摆的 PD 控制：把"控制器"塞进 MuJoCo 的仿真回路

从"看物理"跨到"做控制"的第一步。会学到：
  1. 执行器 motor = 直接给力矩，data.ctrl 就是 N·m
  2. 控制回路和物理步长是两回事：控制 50 Hz，物理 500 Hz
  3. 只有 PD 会有稳态误差（重力把摆拉下来）；加前馈 data.qfrc_bias 才能精确跟踪
  4. 手写 PD 和 MuJoCo 内置 position 执行器是等价的
  5. 【踩坑】几何体意外重叠会生成“幽灵接触”，把关节卡死 —— 用 data.ncon 排查

跑法：python3 03_pendulum_pd.py [--viewer]
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

import common

XML = """
<mujoco model="pendulum_pd">
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0.5 1 3"/>
    <geom name="floor" type="plane" size="4 4 0.1" rgba="0.22 0.24 0.28 1"/>
    <body name="pivot" pos="0 0 1.5">
      <!-- 注意 contype/conaffinity = 0：这是纯视觉支架，不参与碰撞。
           忘了写的话，它会和摆杆重叠，生成一个把铰链卡死的接触约束。 -->
      <geom type="cylinder" size="0.03 0.05" contype="0" conaffinity="0"
            rgba="0.55 0.55 0.58 1"/>
      <body name="rod">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom name="rod_geom" type="capsule" fromto="0 0 0 0 0 -0.6"
              size="0.02" mass="0.3" rgba="0.2 0.6 0.95 1"/>
        <geom name="bob" type="sphere" size="0.06" pos="0 0 -0.6" mass="0.3"
              rgba="0.95 0.6 0.2 1"/>
        <site name="tip" pos="0 0 -0.6"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- motor：ctrl 直接就是力矩（乘以 gear） -->
    <motor name="torque" joint="hinge" gear="1" ctrlrange="-6 6"/>
  </actuator>
</mujoco>
"""

# 同一个模型，但把 motor 换成内置 position 执行器（内部就是 kp/kv 的 PD）
XML_POSITION = XML.replace(
    '<motor name="torque" joint="hinge" gear="1" ctrlrange="-6 6"/>',
    '<position name="servo" joint="hinge" kp="40" kv="4" ctrlrange="-6 6"/>'
).replace('model="pendulum_pd"', 'model="pendulum_servo"')

TAU_MAX = 6.0

# 参考轨迹时间线：(时间, 目标角)。0 rad = 自然下垂
_TIMELINE = [(0.0, 0.0), (0.6, 0.9), (1.8, 0.9), (2.2, -0.6), (4.5, -0.6)]


def _smoothstep(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def reference(t: float) -> float:
    """平滑阶跃：用 smoothstep 过渡，避免力矩瞬间饱和（工程上的标准做法）。"""
    for (t0, v0), (t1, v1) in zip(_TIMELINE, _TIMELINE[1:]):
        if t < t0:
            return v0
        if t < t1:
            return v0 + (v1 - v0) * _smoothstep((t - t0) / (t1 - t0))
    return _TIMELINE[-1][1]


class PDController:
    """最经典的位置控制：tau = kp * (q_des - q) - kd * qdot (+ 重力/科氏前馈)。"""

    def __init__(self, kp: float, kd: float, gravity_comp: bool = False):
        self.kp, self.kd, self.comp = kp, kd, gravity_comp

    def __call__(self, model, data, q_des: float) -> float:
        tau = self.kp * (q_des - data.qpos[0]) - self.kd * data.qvel[0]
        if self.comp:
            tau += data.qfrc_bias[0]     # 抵消重力 + 科氏力 + 离心力
        return float(np.clip(tau, -TAU_MAX, TAU_MAX))


def rollout(controller, duration: float = 4.5, ctrl_every: int = 10):
    """
    跑一次仿真。ctrl_every = 每多少个物理步算一次控制（10 步 × 2 ms = 50 Hz）。
    真实机器人控制器就是几十~几百 Hz，而物理引擎可以更细 —— 这个分离很重要。
    """
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)          # 让 qfrc_bias 有第一帧的值

    ts, angles, targets, taus = [], [], [], []
    for step in range(int(duration / model.opt.timestep)):
        if step % ctrl_every == 0:
            data.ctrl[0] = controller(model, data, reference(data.time))
        mujoco.mj_step(model, data)
        ts.append(data.time)
        angles.append(data.qpos[0])
        targets.append(reference(data.time))
        taus.append(data.ctrl[0])
    return (np.array(ts), np.array(angles), np.array(targets), np.array(taus))


def servo_rollout(duration: float = 4.5, ctrl_every: int = 10):
    """用内置 position 执行器跑同样的参考轨迹（ctrl 直接写期望角度）。"""
    model = mujoco.MjModel.from_xml_string(XML_POSITION)
    data = mujoco.MjData(model)
    ts, angles = [], []
    for step in range(int(duration / model.opt.timestep)):
        if step % ctrl_every == 0:
            data.ctrl[0] = reference(data.time)
        mujoco.mj_step(model, data)
        ts.append(data.time)
        angles.append(data.qpos[0])
    return np.array(ts), np.array(angles)


def hold_error(ts, angles, t0: float, t1: float) -> float:
    """稳态误差：某段时间内 |角度 - 目标| 的平均值。"""
    mask = (ts >= t0) & (ts < t1)
    target = np.array([reference(t) for t in ts[mask]])
    return float(np.mean(np.abs(angles[mask] - target)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="打开交互窗口实时看")
    args = parser.parse_args()

    if args.viewer:
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        ctrl = PDController(kp=40, kd=4.0, gravity_comp=True)

        def control_loop(m, d, t):
            d.ctrl[0] = ctrl(m, d, reference(t))

        print("打开交互窗口……鼠标拖动旋转，空格暂停，关窗口结束。")
        common.run_viewer(model, data, controller=control_loop, max_seconds=20)
        return

    common.banner("对比三种控制器：低增益 / 高增益 / 高增益 + 重力前馈")
    cases = [
        ("低增益 PD      (kp=8,  kd=0.8)", PDController(8, 0.8)),
        ("高增益 PD      (kp=40, kd=4.0)", PDController(40, 4.0)),
        ("高增益 + 前馈  (kp=40, kd=4.0)", PDController(40, 4.0, gravity_comp=True)),
    ]

    results = []
    print(f"  {'控制器':<30}{'保持 0.9 rad':>14}{'保持 -0.6 rad':>15}{'最大力矩':>10}")
    for name, controller in cases:
        ts, ang, tgt, tau = rollout(controller)
        e1 = hold_error(ts, ang, 1.2, 1.7)     # 稳定在 0.9 rad 的窗口
        e2 = hold_error(ts, ang, 3.0, 4.4)     # 稳定在 -0.6 rad 的窗口
        results.append((name, ts, ang, tgt, tau))
        print(f"  {name:<30}{e1:>14.4f}{e2:>15.4f}{np.abs(tau).max():>10.2f}")

    print("""
  解读：
    * 只要 PD 里没有重力项，摆最终停在“重力力矩 = kp × 误差”的平衡点
      → kp 越大误差越小，但永远不为 0（这就是稳态误差）
    * 加上前馈 qfrc_bias（重力 + 科氏力补偿）后，误差掉到 1e-4 量级以下
    * 前馈不是作弊：真机上也常用 model-based feedforward + 反馈修正的组合""")

    # ---------------- 内置 position 执行器等价性 ----------------
    ts_s, ang_s = servo_rollout()
    e_servo = hold_error(ts_s, ang_s, 1.2, 1.7)
    print(f"  对照：内置 position 执行器 (kp=40, kv=4) 的稳态误差 = {e_servo:.4f} rad")
    print("        —— 它内部做的就是和你手写 PD 完全一样的事，只是用 C 实现、更快。")

    # ---------------- 幽灵接触排查（教学点） ----------------
    common.section("踩坑：几何体重叠 → 幽灵接触 → 关节卡死")
    broken = XML.replace(
        '<geom type="cylinder" size="0.03 0.05" contype="0" conaffinity="0"',
        '<geom type="cylinder" size="0.03 0.05"')
    m_bad = mujoco.MjModel.from_xml_string(broken)
    d_bad = mujoco.MjData(m_bad)
    m_good = mujoco.MjModel.from_xml_string(XML)
    d_good = mujoco.MjData(m_good)
    for d in (d_bad, d_good):
        d.ctrl[0] = 3.0
    for _ in range(500):
        mujoco.mj_step(m_bad, d_bad)
        mujoco.mj_step(m_good, d_good)
    print(f"  支架参与碰撞时：ncon={d_bad.ncon}, 摆角={d_bad.qpos[0]:+.3f} rad "
          f"（3 N·m 推了 1 秒几乎没动 → 关节被卡住）")
    print(f"  支架纯视觉时  ：ncon={d_good.ncon}, 摆角={d_good.qpos[0]:+.3f} rad（正常）")
    print("  排查口诀：关节扳不动先看 data.ncon；装饰性几何一律 "
          "contype=0 conaffinity=0。")

    # ---------------- 出图 ----------------
    try:
        plt, font = common.setup_plot()
        tr = (lambda zh, en: zh) if font else (lambda zh, en: en)

        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        colors = ["#d85b2b", "#2b8bd8", "#27a35b"]
        for (name, ts, ang, tgt, tau), c in zip(results, colors):
            axes[0].plot(ts, ang, lw=1.3, color=c, label=name)
            axes[1].plot(ts, tau, lw=1.0, color=c, label=name)
            axes[2].plot(ts, ang - tgt, lw=1.3, color=c, label=name)
        axes[0].plot(ts, tgt, "k--", lw=1.5, label=tr("参考轨迹", "reference"))
        axes[0].set_ylabel(tr("角度 (rad)", "angle (rad)"))
        axes[0].set_title(tr("单摆 PD 位置控制：稳态误差从哪来",
                             "Pendulum PD control: where steady-state error comes from"))
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].set_ylabel(tr("力矩 (N·m)", "torque (N·m)"))
        axes[1].grid(alpha=0.3)
        axes[2].set_ylabel(tr("跟踪误差 (rad)", "tracking error (rad)"))
        axes[2].set_xlabel("time (s)")
        axes[2].grid(alpha=0.3)
        axes[2].set_ylim(-0.5, 0.5)
        fig.tight_layout()
        path = common.OUT / "03_pendulum_pd.png"
        fig.savefig(path, dpi=120)
        print(f"\n  [png] {path}")
    except ImportError:
        print("\n  （没有 matplotlib，跳过曲线图）")

    # ---------------- GIF ----------------
    print("\n  正在渲染 GIF（高增益 + 重力前馈）...")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctrl_best = PDController(40, 4.0, gravity_comp=True)

    def set_ctrl(m, d, t):
        d.ctrl[0] = ctrl_best(m, d, reference(t))

    frames = common.render_frames(
        model, data, n_frames=100, steps_per_frame=20,
        camera=common.make_camera(azimuth=120, elevation=-8,
                                  distance=3.0, lookat=(0, 0, 1.1)),
        controller=set_ctrl, width=480, height=360)
    common.save_gif(frames, "03_pendulum_pd", fps=30)

    common.banner("要点回顾")
    print("  1. data.ctrl 是唯一的控制入口；motor 写力矩，position 写目标角度")
    print("  2. 控制频率（50 Hz）和物理步长（500 Hz）分离，是标准做法")
    print("  3. PD 有稳态误差；前馈 qfrc_bias 消掉重力，误差降 2~3 个数量级")
    print("  4. 装饰性几何一定要设 contype=0 conaffinity=0，否则会产生幽灵接触")
    print("\n练习题：")
    print("  (a) 把 kp 从 40 调到 400，看轨迹会不会抖（振动）？再调 kd 试试。")
    print("  (b) 把 ctrl_every 从 10 改成 100（10 Hz 控制），看跟踪质量怎么退化。")
    print("  (c) 把参考轨迹换成正弦波 0.5*sin(2*pi*0.5*t)，比较三种控制器的相位延迟。")


if __name__ == "__main__":
    main()
