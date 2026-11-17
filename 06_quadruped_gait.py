"""
06 —— 四足机器人：站立 + 原地 trot 踏步 + 接触/步态分析

这是第一个"真正的机器人"模型，会学到：
  1. 用 MJCF 手搓一个 8 自由度四足（每条腿：髋 pitch + 膝 pitch）
  2. 【关键】前后腿的膝必须镜像，否则支撑多边形偏向一侧 → 站都站不住
  3. 步态的本质是"相位"：trot = 对角腿同相；用相位函数生成关节轨迹
  4. 用 data.contact 读出接触状态，画出"步态图"（gait diagram）
  5. 为什么原地踏步容易、往前走路难：2 自由度腿对 yaw/roll 没有控制能力

跑法：python3 06_quadruped_gait.py [--viewer]
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

import common

LEGS = ["FL", "FR", "RL", "RR"]           # 前左、前右、后左、后右
POS = {"FL": (0.11, 0.07), "FR": (0.11, -0.07),
       "RL": (-0.11, 0.07), "RR": (-0.11, -0.07)}
SIGN = {"FL": 1, "FR": 1, "RL": -1, "RR": -1}   # 前后腿的关节轴镜像
TROT = {"FL": 0.0, "RR": 0.0, "FR": np.pi, "RL": np.pi}  # 对角腿同相

THIGH, SHANK, FOOT_R = 0.14, 0.16, 0.025

# 调好的原地踏步参数
FREQ = 2.0     # Hz
KNEE0 = 0.8    # 站立时的膝角
LIFT = 0.4     # 摆动相膝角抬升幅度
DUTY = 0.5     # trot：单腿一半时间摆动


def build_xml(kp: float = 30.0, kv: float = 2.0, with_head: bool = False) -> str:
    legs = []
    for name, (x, y) in POS.items():
        s = SIGN[name]
        rgba = "0.25 0.6 0.95 1" if name in ("FL", "RR") else "0.2 0.5 0.85 1"
        legs.append(f"""
    <body name="hip_{name}" pos="{x} {y} -0.03">
      <joint name="hip_{name}_j" type="hinge" axis="0 {s} 0" range="-1.0 1.0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 -{THIGH}" size="0.018" mass="0.08"
            rgba="{rgba}" contype="0" conaffinity="0"/>
      <body name="knee_{name}" pos="0 0 -{THIGH}">
        <joint name="knee_{name}_j" type="hinge" axis="0 {s} 0" range="0 2.2"
               damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0 0 -{SHANK}" size="0.015" mass="0.06"
              rgba="{rgba}" contype="0" conaffinity="0"/>
        <geom name="foot_{name}" type="sphere" size="{FOOT_R}" pos="0 0 -{SHANK}"
              mass="0.05" rgba="0.15 0.15 0.18 1" condim="3" friction="1.5 0.1 0.1"/>
      </body>
    </body>""")
    head = ("" if not with_head else
            '<geom type="box" size="0.05 0.04 0.02" pos="0.14 0 0.02" mass="0.05"'
            ' rgba="0.85 0.35 0.1 1" contype="0" conaffinity="0"/>')
    acts = "\n".join(
        f'    <position joint="hip_{n}_j" kp="{kp}" kv="{kv}" ctrlrange="-1.0 1.0"/>\n'
        f'    <position joint="knee_{n}_j" kp="{kp}" kv="{kv}" ctrlrange="0 2.2"/>'
        for n in LEGS)
    return f"""<mujoco model="quadruped">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="10 10 0.1" rgba="0.22 0.24 0.28 1"
          friction="1.2 0.05 0.05"/>
    <body name="torso" pos="0 0 0.32">
      <freejoint/>
      <geom type="box" size="0.15 0.08 0.03" mass="1.2" rgba="0.9 0.45 0.15 1"
            contype="0" conaffinity="0"/>
      {head}
      {''.join(legs)}
    </body>
  </worldbody>
  <actuator>
{acts}
  </actuator>
</mujoco>"""


def stand_height(knee0: float) -> float:
    """膝关节弯曲 knee0 时，躯干中心离地的高度。"""
    return 0.03 + THIGH + SHANK * np.cos(knee0) + FOOT_R


def make_data(knee0: float = KNEE0, with_head: bool = False):
    model = mujoco.MjModel.from_xml_string(build_xml(with_head=with_head))
    data = mujoco.MjData(model)
    data.qpos[2] = stand_height(knee0)      # 直接摆好站立姿态，避免开局掉下来
    for n in LEGS:
        data.qpos[model.joint(f"knee_{n}_j").qposadr[0]] = knee0
    mujoco.mj_forward(model, data)
    return model, data


def settle(model, data, knee0: float = KNEE0, seconds: float = 0.4) -> None:
    for _ in range(int(seconds / model.opt.timestep)):
        for i, n in enumerate(LEGS):
            data.ctrl[2 * i] = 0.0
            data.ctrl[2 * i + 1] = knee0
        mujoco.mj_step(model, data)


def trot_control(model, data, t: float, knee0: float = KNEE0, f: float = FREQ,
                 lift: float = LIFT) -> None:
    """trot 相位生成器：对角腿同相，支撑相膝角不变，摆动相抬膝。"""
    for i, n in enumerate(LEGS):
        phase = 2 * np.pi * f * t + TROT[n]
        data.ctrl[2 * i] = 0.0                              # 髋：原地踏步不摆动
        data.ctrl[2 * i + 1] = knee0 + lift * max(0.0, -np.cos(phase))


def foot_contact_id(model):
    """把 foot_* 几何体的 id 收集起来，方便判断哪只脚着地。"""
    ids = {}
    for n in LEGS:
        ids[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{n}")] = n
    return ids


def measure(f: float, knee0: float, lift: float, seconds: float = 8.0,
            with_head: bool = False):
    """跑一段 trot，统计：接触模式、抬脚高度、躯干高度波动。"""
    model, data = make_data(knee0, with_head=with_head)
    settle(model, data, knee0)
    foot_ids = foot_contact_id(model)

    contacts = []      # 每步的 4 个布尔值
    zs, qw, foot_z = [], [], []
    for _ in range(int(seconds / model.opt.timestep)):
        trot_control(model, data, data.time - 0.4, knee0, f, lift)
        mujoco.mj_step(model, data)
        zs.append(data.qpos[2])
        qw.append(abs(data.qpos[3]))
        foot_z.append([data.geom(f"foot_{n}").xpos[2] for n in LEGS])
        in_contact = {n: False for n in LEGS}
        for ci in range(data.ncon):
            c = data.contact[ci]
            for g in (c.geom1, c.geom2):
                if g in foot_ids:
                    in_contact[foot_ids[g]] = True
        contacts.append([in_contact[n] for n in LEGS])

    contacts = np.array(contacts)
    foot_z = np.array(foot_z)
    zs = np.array(zs)
    return dict(
        model=model, data=data,
        contacts=contacts,
        n_support=contacts.sum(axis=1),
        z=zs,
        lift=np.round(foot_z.max(axis=0) - foot_z.min(axis=0), 3),
        stable=(zs.min() > 0.2 and min(qw) > 0.85),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="打开交互窗口实时看")
    args = parser.parse_args()

    if args.viewer:
        model, data = make_data(KNEE0)
        settle(model, data, KNEE0)
        print("打开交互窗口……这是原地 trot 踏步，关窗口结束。")
        common.run_viewer(
            model, data,
            controller=lambda m, d, t: trot_control(m, d, t - 0.4, KNEE0),
            max_seconds=30)
        return

    # ================= 1. 站立 ================= #
    common.banner("1) 先站住：镜像膝关节的作用")
    model, data = make_data(KNEE0)
    settle(model, data, KNEE0, seconds=1.0)
    print(f"  躯干高度 = {data.qpos[2]:.3f} m（理论值 {stand_height(KNEE0):.3f}）")
    print(f"  接触数 ncon = {data.ncon}（4 只脚都着地）")
    print(f"  姿态四元数 qw = {data.qpos[3]:+.3f}（≈1 表示没有翻倒）")

    # 反例：不镜像膝关节会怎样
    broken = build_xml().replace('axis="0 -1 0"', 'axis="0 1 0"')
    m_bad = mujoco.MjModel.from_xml_string(broken)
    d_bad = mujoco.MjData(m_bad)
    d_bad.qpos[2] = stand_height(KNEE0)
    for n in LEGS:
        d_bad.qpos[m_bad.joint(f"knee_{n}_j").qposadr[0]] = KNEE0
    mujoco.mj_forward(m_bad, d_bad)
    for _ in range(500):
        d_bad.ctrl[:] = [KNEE0 if i % 2 else 0.0 for i in range(m_bad.nu)]
        mujoco.mj_step(m_bad, d_bad)
    print(f"  【反例】如果后腿膝轴不镜像：1 s 后 qw = {d_bad.qpos[3]:+.3f}、"
          f"高度 = {d_bad.qpos[2]:+.3f} m → 已经翻倒")
    print("  → 四条腿同向弯曲会让支撑多边形整体偏移，重心跑到支撑区外，必倒。")

    # 重心实验：给躯干前端加 50 g 配重
    st_head = measure(FREQ, KNEE0, LIFT, seconds=6.0, with_head=True)
    print(f"  【配重实验】躯干前端加 50 g 配重后：躯干高度范围 "
          f"[{st_head['z'].min():.2f}, {st_head['z'].max():.2f}] m，"
          f"{'稳定' if st_head['stable'] else '翻倒'}")
    print("  → 只把重心前移不到 4 mm 就足以让动态步态崩掉："
          "trot 的双足支撑只是一条对角线，对重心极其敏感。")

    # ================= 2. 步态参数扫描 ================= #
    common.banner("2) 原地 trot 扫参：频率和抬腿幅度怎么影响稳定性")
    print(f"  {'频率 Hz':>8}{'抬膝 rad':>10}{'四足支撑%':>10}{'双足支撑%':>10}"
          f"{'躯干高度波动':>13}{'抬脚 cm':>9}  判定")
    best = None
    for f in (1.5, 2.0, 2.5):
        for lift in (0.2, 0.4, 0.6):
            st = measure(f, KNEE0, lift, seconds=5.0)
            support = st["n_support"]
            four = float((support == 4).mean() * 100)
            two = float((support <= 2).mean() * 100)
            zrange = float(st["z"].max() - st["z"].min())
            tag = "稳定" if st["stable"] else "翻倒"
            print(f"  {f:>8.1f}{lift:>10.1f}{four:>10.0f}{two:>10.0f}"
                  f"{zrange:>13.3f}{st['lift'].mean() * 100:>9.1f}  {tag}")
            if st["stable"] and (best is None or two > best[0]):
                best = (two, f, lift, st)

    if best is None:
        print("  （没有稳定配置，使用默认参数）")
        best = (0, FREQ, LIFT, measure(FREQ, KNEE0, LIFT))
    _, f_best, lift_best, st = best
    print(f"\n  选中的参数：f = {f_best} Hz, 抬膝 = {lift_best} rad"
          f"（双足支撑时间占比 {st['n_support'].mean() / 4 * 100:.0f}% 左右）")
    print("""
  解读：
    * trot 是"动态步态"：任何时刻只有对角的两条腿着地，靠动量维持
    * 频率太高 → 落地冲击大、躯干晃；太低 → 单脚支撑时间过长容易侧倾
    * "抬脚高度"来自摆动相膝角：抬得太高会重心上下起伏过大""")

    # ================= 3. 步态图 ================= #
    common.banner("3) 步态图（gait diagram）：看懂接触序列")
    st = measure(f_best, KNEE0, lift_best, seconds=3.0)
    contacts = st["contacts"]
    print("  每个字母表示该腿着地，. 表示腾空：")
    step = max(1, len(contacts) // 60)
    print("        " + "".join(str(i // 10 % 10) for i in range(60)))
    for k, n in enumerate(LEGS):
        row = "".join("█" if contacts[i][k] else "·" for i in range(0, len(contacts), step))
        print(f"  {n}    {row}")
    print(f"\n  对角线 FL↔RR、FR↔RL 严格同相 —— 这就是 trot 的定义。")

    try:
        plt, font = common.setup_plot()
        tr = (lambda zh, en: zh) if font else (lambda zh, en: en)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                       gridspec_kw={"height_ratios": [1, 1.2]})
        t = np.arange(len(contacts)) * st["model"].opt.timestep
        ax1.imshow(contacts.T, aspect="auto", cmap="Greys", interpolation="nearest",
                   extent=[t[0], t[-1], 3.5, -0.5])
        ax1.set_yticks(range(4))
        ax1.set_yticklabels(LEGS)
        ax1.set_title(tr("步态图：黑 = 着地，白 = 腾空",
                          "Gait diagram: dark = stance, light = swing"))
        ax2.plot(t, st["z"], lw=1.0, color="#2b8bd8")
        ax2.set_ylabel(tr("躯干高度 (m)", "torso height (m)"))
        ax2.set_xlabel("time (s)")
        ax2.set_title(tr("躯干高度：trot 的上下起伏",
                          "Torso height oscillation"))
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        path = common.OUT / "06_gait_diagram.png"
        fig.savefig(path, dpi=120)
        print(f"\n  [png] {path}")
    except ImportError:
        print("\n  （没有 matplotlib，跳过步态图）")

    # ================= 4. GIF ================= #
    print("\n  正在渲染 GIF（原地 trot）...")
    model, data = make_data(KNEE0)
    settle(model, data, KNEE0)
    frames = common.render_frames(
        model, data, n_frames=100, steps_per_frame=20,
        camera=common.make_camera(azimuth=120, elevation=-16,
                                  distance=0.95, lookat=(0, 0, 0.22)),
        controller=lambda m, d, tt: trot_control(m, d, tt - 0.4),
        width=480, height=360)
    common.save_gif(frames, "06_quadruped_trot", fps=30)

    common.banner("要点回顾 & 为什么不能直接让它往前走")
    print("  1. 相位函数生成关节轨迹，是开环步态最简单的实现方式")
    print("  2. 前后腿膝轴镜像、髋轴镜像，是四条腿能对称支撑的前提")
    print("  3. data.contact 能读出每一步谁在接触 → 画步态图、算占空比")
    print("  4. 但只靠开环，往前走会失败：2 自由度腿无法产生 yaw/roll 修正力矩")
    print("     → 实测：加髋摆动后，机器人 1 秒内就会偏航打转并摔倒")
    print("\n想让它真走起来，下一步是：")
    print("  (a) 每条腿加一个髋外摆关节（3 自由度 × 4），用 IMU 反馈横滚/偏航")
    print("  (b) 或者直接上强化学习（第 7 课的 CEM 思路，或 PPO/MJX）")
    print("\n练习题：")
    print("  (a) 把 DUTY 从 0.5 改成 0.75（三足支撑的 walk 步态），看还能不能站住。")
    print("  (b) 给 trot 加一个正弦的髋关节前后摆动（swing），看机器人朝哪边漂。")
    print("  (c) 把躯干质量从 1.2 kg 改到 0.3 kg，看步态频率需要怎么调整。")


if __name__ == "__main__":
    main()
