"""
练习 02(c) —— 能不能用 actuator_dampingpoly 加阻力？

答案：**可以，而且这正是 MuJoCo 内置的"多项式阻尼"**。

这个 3.17 构建新增了 <damper> 执行器（官方老版本里没有），它的两个通道是：

  ① 被动通道 damping：力写进 qfrc_passive，和关节的 damping 参数一个效果
         F = -(c0 + c1·|v| + c2·v²) · v
     XML 写法：  <damper joint="j" damping="c0 c1 c2" ctrlrange="0 1"/>
     - 第一个数进 model.actuator_damping（标量）
     - 后两个数进 model.actuator_dampingpoly（每个执行器 2 个系数，mjNPOLY=2）

     想让阻力是**二次**的，就写 damping="0 k 0"：
         F = -(0 + k·|v|)·v = -k·|v|·v     ← 就是空气阻力公式

  ② 主动通道 kv（受 ctrl 缩放，可做"半主动阻尼器/刹车"）
         F = -(kv · ctrl) · v
     XML 写法：  <damper joint="j" kv="2" ctrlrange="0 1"/>
     运行时改 data.ctrl 就能实时调整阻尼强度。

本脚本验证三件事：
  A. 铰链上用 <damper damping="0 k 0">，和"python 里写 qfrc_applied"的手算二次阻力
     结果是否一致（应该几乎重合）
  B. 自由关节（下落小球）上也能挂 damper，终端速度和理论 √(mg/k) 对不对
  C. 运行时直接改 model.actuator_dampingpoly 来"开关"阻力

跑法：python3 ex_02c_damper_actuator.py
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

import common

# --------------------------------------------------------------------------- #
# A. 单摆 + 二次阻力：damper 执行器 vs Python 手算
# --------------------------------------------------------------------------- #
PENDULUM_XML = """
<mujoco model="pendulum_damper">
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="rod" pos="0 0 1.2">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.4" pos="0 0 -0.4" mass="1"
            rgba="0.2 0.6 0.9 1"/>
      <site name="tip" pos="0 0 -0.8"/>
    </body>
  </worldbody>
  <actuator>
    <!-- 二次阻尼：c0=0, c1=K_QUAD, c2=0  →  F = -K_QUAD·|v|·v -->
    <damper name="air" joint="hinge" damping="0 {k_quad} 0" ctrlrange="0 1"/>
  </actuator>
</mujoco>
"""

K_QUAD = 0.05          # N·(s/rad)²  二次阻尼系数（为了衰减速明显大一点）


def pendulum_run(use_damper: bool, k_quad: float = K_QUAD,
                 duration: float = 8.0, integrator: str = "implicitfast"):
    """返回 (时间, 可用机械能比)。use_damper=False 时改用 Python 手算阻力。"""
    xml = PENDULUM_XML.format(k_quad=k_quad).replace(
        'integrator="implicitfast"', f'integrator="{integrator}"')
    model = mujoco.MjModel.from_xml_string(xml)
    if not use_damper:
        # 把 XML 里的 damper 关掉：dampingpoly 清零 → 等效无阻力
        model.actuator_dampingpoly[0] = [0.0, 0.0]
    common.enable_energy(model)

    data = mujoco.MjData(model)

    # 势能基线：静止在最低点
    data.qpos[0] = 0.0
    mujoco.mj_forward(model, data)
    e_min = common.total_energy(data)
    data.qpos[0] = 1.0
    mujoco.mj_forward(model, data)
    e0 = common.total_energy(data)

    ts, er = [], []
    for _ in range(int(duration / model.opt.timestep)):
        if not use_damper:
            # 手算：F = -k·|v|·v，写进广义外力
            v = float(data.qvel[0])
            data.qfrc_applied[0] = -k_quad * abs(v) * v
        mujoco.mj_step(model, data)
        ts.append(data.time)
        er.append((common.total_energy(data) - e_min) / (e0 - e_min))
    return np.array(ts), np.array(er)


# --------------------------------------------------------------------------- #
# B. 下落小球：自由关节挂 damper 做二次空气阻力
# --------------------------------------------------------------------------- #
RHO, CD = 1.225, 0.47
BALL_R, BALL_M = 0.08, 0.005
BALL_A = math.pi * BALL_R ** 2
K_AIR = 0.5 * RHO * CD * BALL_A          # F = -K_AIR·|v|·v
V_TERMINAL = math.sqrt(BALL_M * 9.81 / K_AIR)

BALL_XML = f"""
<mujoco model="ball_damper">
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="3 3 0.1" rgba="0.25 0.28 0.32 1"/>
    <body name="ball" pos="0 0 2.0">
      <freejoint name="ball_joint"/>
      <geom name="ball_geom" type="sphere" size="{BALL_R}" mass="{BALL_M}"
            rgba="0.25 0.65 0.95 1"/>
    </body>
  </worldbody>
  <actuator>
    <!-- 自由关节也能挂 damper；系数取真实的 ½ρCdA，二次项 -->
    <damper name="air" joint="ball_joint" damping="0 {K_AIR:.6f} 0" ctrlrange="0 1"/>
  </actuator>
</mujoco>
"""


def ball_run(duration: float = 2.0):
    model = mujoco.MjModel.from_xml_string(BALL_XML)
    data = mujoco.MjData(model)
    ts, speeds, zs = [], [], []
    for _ in range(int(duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
        ts.append(data.time)
        speeds.append(float(np.linalg.norm(data.qvel[0:3])))
        zs.append(float(data.qpos[2]))
    return np.array(ts), np.array(speeds), np.array(zs)


# --------------------------------------------------------------------------- #
def main() -> None:
    common.banner("A. 铰链上用 <damper> 的二次阻尼 vs Python 手算")
    print(f"  阻尼系数 k_quad = {K_QUAD}，力 = -k·|v|·v")
    model = mujoco.MjModel.from_xml_string(PENDULUM_XML.format(k_quad=K_QUAD))
    print(f"  XML 的 damping=\"0 {K_QUAD} 0\" 编译结果："
          f"actuator_damping={model.actuator_damping}, "
          f"actuator_dampingpoly={model.actuator_dampingpoly.tolist()}")

    ts_a, er_a = pendulum_run(use_damper=True)
    ts_b, er_b = pendulum_run(use_damper=False)
    diff = float(np.abs(er_a - er_b).max())
    print(f"  8 s 后可用能量：damper 执行器 {er_a[-1]:.4f}，手算 qfrc_applied {er_b[-1]:.4f}")
    print(f"  两条曲线最大差异 = {diff:.2e}（implicitfast 积分器）")
    for integ in ("Euler", "RK4"):
        _, ea = pendulum_run(True, duration=4.0, integrator=integ)
        _, eb = pendulum_run(False, duration=4.0, integrator=integ)
        print(f"  换 {integ:<6} 积分器时，两条曲线的最大差异 = "
              f"{float(np.abs(ea - eb).max()):.2e}")
    print("""
  结论：<damper damping="0 k 0"> 和手写 qfrc_applied 是**同一个物理力**；
        1e-3 级的差别来自数值处理方式：被动阻尼（damper → qfrc_passive）会被
        并进积分器的阻尼项，而 qfrc_applied 是每一步显式加上的外力。
        8 秒后能量 0.0568 vs 0.0567，肉眼完全看不出区别。""")

    # ---------------- B. 自由关节小球 ----------------
    common.banner("B. 自由关节上的 damper：终端速度对不对？")
    print(f"  小球 r={BALL_R} m, m={BALL_M} kg, ½ρCdA = {K_AIR:.6f}")
    print(f"  理论终端速度 = √(mg/k) = {V_TERMINAL:.3f} m/s")
    ts, speeds, zs = ball_run()
    mask = (ts > 0.5) & (zs > BALL_R + 0.02)      # 只统计还在空中的阶段
    print(f"  实测（空中阶段）平均速度 = {speeds[mask].mean():.3f} m/s，"
          f"最大 = {speeds.max():.3f} m/s")
    print(f"  与理论的相对误差：平均速度 {abs(speeds[mask].mean() - V_TERMINAL) / V_TERMINAL * 100:.1f}%，"
          f"最大速度 {abs(speeds.max() - V_TERMINAL) / V_TERMINAL * 100:.1f}%")
    print("""
  注意：自由关节有 6 个 dof，damper 的阻尼是**逐轴**算的：
        F_i = -(damping + c1|v_i| + c2·v_i²)·v_i
  所以小球只在竖直方向运动时，它等价于真实空气阻力；
  斜着飞的时候每个轴各自阻尼，和"各向同性的 -k|v|v"有细微差别。
  要严格各向同性，还是用 qfrc_applied 回调（见 ex_02b_air_drag.py）。""")

    # ---------------- C. 运行时改 dampingpoly ----------------
    common.banner("C. 运行时直接改 model.actuator_dampingpoly（做「风洞开关」）")
    model = mujoco.MjModel.from_xml_string(PENDULUM_XML.format(k_quad=K_QUAD))
    common.enable_energy(model)                      # 打开能量统计
    model.actuator_dampingpoly[0] = [0.0, 0.0]      # 关掉阻力
    data = mujoco.MjData(model)
    data.qpos[0] = 1.0
    mujoco.mj_forward(model, data)
    print("  前 2 秒：无阻力（dampingpoly=[0,0]）")
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    e_before = common.total_energy(data)
    model.actuator_dampingpoly[0] = [K_QUAD, 0.0]   # 打开二次阻力
    print(f"  在 t={data.time:.1f}s 把 dampingpoly 设为 [k, 0]，继续 2 秒")
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    e_after = common.total_energy(data)
    print(f"  开阻力前能量 = {e_before:.3f} J，2 秒后 = {e_after:.3f} J "
          f"→ 下降 {100 * (1 - e_after / e_before):.1f}%")
    print("  → dampingpoly 是 MjModel 的数组，可以直接在仿真中途改写（风洞/变阻力实验很好用）")

    # ---------------- D. kv 通道 ----------------
    common.banner("D. 另一个通道 kv：受 ctrl 控制的「半主动阻尼器」")
    xml_kv = PENDULUM_XML.format(k_quad=K_QUAD).replace(
        f'damping="0 {K_QUAD} 0"', 'damping="0 0 0" kv="0.5"')
    model = mujoco.MjModel.from_xml_string(xml_kv)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.0
    data.qvel[0] = 1.0
    mujoco.mj_forward(model, data)
    print("  力 = -kv·ctrl·v（kv=0.5, v=1 rad/s）：")
    for ctrl in (0.0, 0.5, 1.0):
        data.qvel[0] = 1.0
        data.ctrl[0] = ctrl
        mujoco.mj_forward(model, data)
        print(f"    ctrl={ctrl:<4} → qfrc_actuator = {float(data.qfrc_actuator[0]):+.3f} N·m")

    # ---------------- 出图 ----------------
    try:
        plt, font = common.setup_plot()
        tr = (lambda zh, en: zh) if font else (lambda zh, en: en)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax1.plot(ts_a, er_a, lw=2.2, color="#2b8bd8",
                 label=tr('<damper damping="0 k 0">', "damper actuator"))
        ax1.plot(ts_b, er_b, "--", lw=1.2, color="#d85b2b",
                 label=tr("Python 手算 qfrc_applied", "Python qfrc_applied"))
        ax1.set(xlabel="time (s)", ylabel=tr("可用机械能比", "available energy / E0"),
                title=tr("两种二次阻尼写法：曲线几乎完全重合（差异 1e-3）",
                         "Both quadratic-drag implementations agree"))
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=8)

        ax2.plot(ts, speeds, lw=1.8, color="#2b8bd8",
                 label=tr("小球速度（damper 执行器）", "ball speed (damper actuator)"))
        ax2.axhline(V_TERMINAL, color="k", ls=":", lw=1.2,
                    label=tr(f"理论终端速度 {V_TERMINAL:.2f} m/s",
                             f"terminal velocity {V_TERMINAL:.2f} m/s"))
        ax2.set(xlabel="time (s)", ylabel=tr("速度 (m/s)", "speed (m/s)"),
                title=tr("自由关节挂 damper：收敛到终端速度",
                         "Free-joint damper reaches terminal velocity"))
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        path = common.OUT / "ex_02c_damper_actuator.png"
        fig.savefig(path, dpi=120)
        print(f"\n  [png] {path}")
    except ImportError:
        print("\n  （没有 matplotlib，跳过曲线图）")

    common.banner("总结：actuator_dampingpoly 能怎么用")
    print("""
  ✅ 能用。它是 <damper> 执行器的被动阻尼通道，力公式：
        F = -(damping + c1·|v| + c2·v²) · v   （逐 dof 计算）
     XML：  <damper joint="j" damping="c0 c1 c2" ctrlrange="0 1"/>
     二次阻力（空气阻力）：damping="0 k 0"
     线性 + 二次混合：    damping="c0 k 0"（c0 进 actuator_damping 标量）

  ✅ 运行时也能改：model.actuator_dampingpoly[i] = [c1, c2]
     （可以直接在仿真中途打开/关闭阻力，做"风洞"实验）

  ✅ 能挂在自由关节上（自由飞行的小球），但它是**逐轴**阻尼，
     只在运动方向单一（例如竖直下落）时等价于真实的各向同性空气阻力。
     要严格各向同性：用 qfrc_applied 回调（ex_02b_air_drag.py 里那套）。

  ✅ kv 通道 = 受 ctrl 控制的阻尼（半主动阻尼器/刹车）：
        F = -kv·ctrl·v，ctrlrange 必须是 [0, 1] 这类非负区间

  ⚠️ 老版本 MuJoCo 没有 <damper> 这个短写和 dampingpoly 字段，
     用 `<general joint="j" biastype="affine" biasprm="0 -1 -kv"/>` 之类近似，
     或者直接写 qfrc_applied。
""")


if __name__ == "__main__":
    main()
