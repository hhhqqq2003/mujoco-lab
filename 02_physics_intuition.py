"""
02 —— 物理直觉：积分器精度、能量守恒、接触"弹性"

MuJoCo 是物理引擎，但不是"自动正确"。这一课用两个最小实验量化三件事：
  1. timestep / 积分器如何影响精度（用无接触的单摆看能量漂移，干扰最少）
  2. MuJoCo 没有 restitution 系数！弹性来自接触模型的阻尼比 solref 第二个数
  3. 接触刚度和 timestep 必须匹配，否则会数值爆炸（能量凭空增长）

跑法：python3 02_physics_intuition.py
"""

from __future__ import annotations

import mujoco
import numpy as np

import common

# --------------------------------------------------------------------------- #
# 实验 A：无接触单摆 —— 干净的"数值精度"测试台
# --------------------------------------------------------------------------- #
PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="{dt}" integrator="{integ}"/>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="rod" pos="0 0 1.2">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" size="0.02 0.4" pos="0 0 -0.4" mass="1" rgba="0.2 0.6 0.9 1"/>
      <site name="tip" pos="0 0 -0.8"/>
    </body>
  </worldbody>
</mujoco>
"""

# --------------------------------------------------------------------------- #
# 实验 B：小球落在弹性地面上
# --------------------------------------------------------------------------- #
BALL_XML = """
<mujoco model="bouncing_ball">
  <option timestep="{dt}" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="3 3 0.1" solref="{solref}"/>
    <body name="ball" pos="0 0 1.2">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.1" mass="1"
            rgba="0.2 0.55 0.95 1" solref="{solref}" condim="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def simulate_pendulum(integ: str, dt: float, duration: float = 10.0):
    """单摆从水平位置释放（无阻尼、无接触）。能量守恒 → 数值误差 = 能量漂移。"""
    model = mujoco.MjModel.from_xml_string(
        PENDULUM_XML.format(dt=dt, integ=integ))
    common.enable_energy(model)
    data = mujoco.MjData(model)
    data.qpos[0] = 1.0          # 初始摆角 1 rad
    mujoco.mj_forward(model, data)

    ts, ang, e_ratio = [], [], []
    e0 = None
    for _ in range(int(duration / dt)):
        mujoco.mj_step(model, data)
        e = common.total_energy(data)
        e0 = e0 if e0 is not None else e
        ts.append(data.time)
        ang.append(data.qpos[0])
        e_ratio.append(e / e0)
    return np.array(ts), np.array(ang), np.array(e_ratio), data.time


def drop_ball(sr: str, dt: float = 0.002, duration: float = 2.0):
    """小球从 1.2 m 自由落下，返回 (各次反弹最高点列表, 末态能量比, 是否数值爆炸)。"""
    model = mujoco.MjModel.from_xml_string(BALL_XML.format(dt=dt, solref=sr))
    common.enable_energy(model)
    data = mujoco.MjData(model)
    zs = []
    for _ in range(int(duration / dt)):
        mujoco.mj_step(model, data)
        zs.append(data.qpos[2])
    zs = np.array(zs)
    peaks = [zs[i] for i in range(1, len(zs) - 1)
             if zs[i] > zs[i - 1] and zs[i] >= zs[i + 1] and zs[i] > 0.12]
    e_ratio = common.total_energy(data) / (9.81 * 1.2)   # 初始势能 = m*g*h
    exploded = e_ratio > 2 or (peaks and max(peaks) > 1.3)
    return [round(float(v), 3) for v in peaks[:4]], round(float(e_ratio), 3), exploded


def main() -> None:
    # ==================== 实验 A：积分器 ==================== #
    common.banner("实验 A：积分器精度 —— 无接触单摆，理论上能量永远守恒")

    runs = [("Euler", 0.02), ("Euler", 0.01), ("Euler", 0.005), ("Euler", 0.0005),
            ("RK4", 0.02), ("RK4", 0.005), ("implicitfast", 0.02)]
    print(f"  {'integrator':<13}{'dt (s)':>8}{'仿真步数':>10}{'能量漂移':>12}")
    curve = []
    for integ, dt in runs:
        ts, ang, er, t_end = simulate_pendulum(integ, dt)
        drift = (er[-1] - 1.0) * 100
        curve.append((integ, dt, ts, er))
        print(f"  {integ:<13}{dt:>8.4f}{len(ts):>10d}{drift:>+11.4f}%")

    print("""
  解读：
    * Euler（和 implicitfast）是一阶法：漂移几乎正比于 dt，10 秒能漂 0.7%
    * RK4 是四阶法：dt=0.02 时漂移仍然 < 0.001%，但每步要算 4 次导数
    * implicitfast 是 MuJoCo 的默认选择：对接触稳定，代价是一点数值阻尼
    * 实机做强化学习时，通常 dt=0.002~0.005 + implicitfast 就够了""")

    # ==================== 实验 B：接触弹性 ==================== #
    common.banner("实验 B：MuJoCo 没有 restitution 系数，弹性来自 solref 阻尼比")

    configs = [
        ("0.02 1", "临界阻尼：完全不弹，快速吸能（默认手感）"),
        ("0.02 0.3", "欠阻尼：轻微弹跳"),
        ("0.02 0.1", "欠阻尼：明显反弹"),
        ("0.02 0.05", "强欠阻尼：接近皮球质感"),
    ]
    a_dt = 0.001   # 表 B 用更小的 dt，保证强弹跳配置也稳定
    print(f"  dt = {a_dt} s")
    print(f"  {'solref':<12}{'反弹最高点 (m)':<30}{'末态 E/E0':>10}  说明")
    for sr, desc in configs:
        peaks, e_ratio, exploded = drop_ball(sr, dt=a_dt)
        print(f"  {sr:<12}{str(list(peaks)):<28}{e_ratio:>10.3f}  {desc}")

    print("""
  解读：
    * MuJoCo 的接触是"弹簧 + 阻尼"模型，没有 restitution 这个参数
    * solref = "时间常数 阻尼比"：阻尼比 < 1 → 欠阻尼 → 会弹
    * 从 1.2 m 落下，solref="0.02 0.1" 只弹到约 0.7 m —— 因为弹性能量被接触阻尼吃掉一部分
    * 想要"几乎无损"的弹跳，实际工程里会在接触后再手动翻转法向速度（自己写 restitution）""")

    # ==================== 实验 C：接触刚度 vs 时间步 ==================== #
    common.banner("实验 C：接触刚度和 timestep 必须匹配，否则能量凭空爆炸")

    print(f"  {'solref':<12}{'dt (s)':<10}{'反弹最高点':<28}{'判定'}")
    for sr in ("0.02 0.1", "0.01 0.05"):
        for dt in (0.002, 0.0005):
            peaks, e_ratio, exploded = drop_ball(sr, dt=dt)
            tag = "💥 数值爆炸（能量暴增）" if exploded else "稳定"
            print(f"  {sr:<12}{dt:<10.4f}{str(list(peaks)):<28}{tag}")
    print("""
  解读：
    * 接触"刚度" ≈ 1/时间常数²。时间常数 0.01 s 的硬接触需要极小 dt 才稳
    * dt 相对接触固有周期过大 → 显式积分失稳 → 能量被注入系统（越弹越高）
    * 经验：先用默认 solref="0.02 1"，需要弹跳再调小阻尼比并同步减小 dt""")

    # ==================== 出图 ==================== #
    try:
        plt, font = common.setup_plot()
        tr = (lambda zh, en: zh) if font else (lambda zh, en: en)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for integ, dt, ts, er in curve:
            ax1.plot(ts, er, lw=1.4, label=f"{integ} dt={dt}")
        ax1.set(xlabel="time (s)", ylabel="E / E0",
                title=tr("单摆能量比：理想值恒为 1", "Pendulum energy ratio (ideal = 1)"))
        ax1.axhline(1.0, color="k", ls="--", lw=0.8)
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=7)

        # 反弹高度对比
        durs = np.arange(0, 3.0, a_dt)
        for sr, _ in configs:
            m = mujoco.MjModel.from_xml_string(BALL_XML.format(dt=a_dt, solref=sr))
            d = mujoco.MjData(m)
            hs = []
            for _ in range(len(durs)):
                mujoco.mj_step(m, d)
                hs.append(d.qpos[2])
            ax2.plot(durs, hs, lw=1.2, label=f"solref={sr}")
        ax2.set_ylim(-0.05, 1.6)
        ax2.set(xlabel="time (s)", ylabel="ball height z (m)",
                title=tr("弹跳过程：solref 阻尼比控制弹性",
                         "Bounce: damping ratio controls elasticity"))
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        path = common.OUT / "02_integrator_and_contact.png"
        fig.savefig(path, dpi=120)
        print(f"\n  [png] {path}")
    except ImportError:
        print("\n  （没有 matplotlib，跳过曲线图）")

    # ==================== GIF ==================== #
    print("\n  正在渲染皮球 GIF ...")
    model = mujoco.MjModel.from_xml_string(
        BALL_XML.format(dt=0.002, solref="0.02 0.1"))
    data = mujoco.MjData(model)
    frames = common.render_frames(
        model, data, n_frames=90, steps_per_frame=20,
        camera=common.make_camera(azimuth=120, elevation=-10,
                                  distance=2.4, lookat=(0, 0, 0.6)),
        width=480, height=360)
    common.save_gif(frames, "02_bouncing_ball", fps=30)

    common.banner("要点回顾")
    print("  1. dt 是精度/速度旋钮；接触越硬，越需要小 dt")
    print("  2. 开 mjENBL_ENERGY 看 data.energy，是最好用的数值体检工具")
    print("  3. 弹性不是一个系数，而是接触模型（solref/solimp）的涌现行为")
    print("\n练习题：")
    print("  (a) 把单摆的阻尼 damping 设成 0.05，能量还守恒吗？（应该单调下降）")
    print("  (b) 把实验 C 里 solref 的第二个数固定 0.1，只把时间常数从 0.02 改到 0.005，"
          "找到让 dt=0.002 爆炸的临界值。")
    print("  (c) 给球加一个随速度变化的额外力（data.qfrc_applied），模拟空气阻力。")


if __name__ == "__main__":
    main()
