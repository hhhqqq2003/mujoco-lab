"""
练习 02(a) —— 给单摆加阻尼 damping="0.05"，能量还守恒吗？

先给答案：**不守恒，而且一定单调下降**。
原因：MuJoCo 的关节阻尼是一个与速度成正比的力矩
      tau = -c * qvel
它对系统做的功率是
      P = tau * qvel = -c * qvel²  ≤ 0
功率恒为负（最多等于 0），机械能只可能单向流出 → 单调下降。
能量最终变成热耗散掉，摆停在最低点。

本脚本做三件事：
  1. 扫几个阻尼值，量化「能量掉了多少」
  2. 检查每一步的能量变化是否真的单调（有没有数值抖动）
  3. 用 I/c 预测能量衰减时间常数，和实测对比（I 从 MjData 的惯性矩阵里读）

⚠️ 一个容易踩的坑：data.energy 的势能有零点偏移（不是"到底部=0"）。
   摆停到底部时 E/E0 会收敛到 0.81 而不是 0，看起来"没耗散多少"。
   所以这里先算「底部静止能量 E_min」，用 (E - E_min) 作为可用机械能来归一化。

跑法：python3 ex_02a_damping.py
"""

from __future__ import annotations

import mujoco
import numpy as np

import common

# 和 02_physics_intuition.py 里同一个单摆，只是多了一个 {damping} 占位符
PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="{dt}" integrator="{integ}"/>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="rod" pos="0 0 1.2">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="{damping}"/>
      <geom type="capsule" size="0.02 0.4" pos="0 0 -0.4" mass="1" rgba="0.2 0.6 0.9 1"/>
      <site name="tip" pos="0 0 -0.8"/>
    </body>
  </worldbody>
</mujoco>
"""


def build(damping: float, integ: str = "implicitfast", dt: float = 0.002):
    model = mujoco.MjModel.from_xml_string(
        PENDULUM_XML.format(damping=damping, dt=dt, integ=integ))
    common.enable_energy(model)
    return model, mujoco.MjData(model)


def rest_energy(damping: float, integ: str = "implicitfast", dt: float = 0.002):
    """摆静止在最低点时的总能量（势能零点基线），同时返回转动惯量 I。"""
    model, data = build(damping, integ, dt)
    data.qpos[0] = 0.0        # 自然下垂
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, M)      # MuJoCo ≥3.1: mj_fullM(m, d, dst)
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)   # 老版本: mj_fullM(m, M, M_sparse)
    return common.total_energy(data), float(M[0, 0])


def simulate(damping: float, integ: str = "implicitfast", dt: float = 0.002,
             duration: float = 10.0):
    """
    从 1 rad（水平）释放，返回：
      ts      时间
      excess  可用机械能占比 (E - E_min)/(E0 - E_min)，从 1 单调趋向 0
      angles  摆角
      inertia 转动惯量 I
    """
    model, data = build(damping, integ, dt)
    e_min, inertia = rest_energy(damping, integ, dt)

    data.qpos[0] = 1.0
    mujoco.mj_forward(model, data)
    e0 = common.total_energy(data)

    ts, excess, angles = [], [], []
    for _ in range(int(duration / dt)):
        mujoco.mj_step(model, data)
        ts.append(data.time)
        excess.append((common.total_energy(data) - e_min) / (e0 - e_min))
        angles.append(float(data.qpos[0]))
    return np.array(ts), np.array(excess), np.array(angles), inertia


def main() -> None:
    common.banner("练习 02(a)：阻尼 damping 对能量守恒的影响")
    print("  模型：单摆从 1 rad（水平）释放，初始速度 0")
    print("  阻尼力矩 tau = -damping × qvel  →  功率 P = -damping × qvel² ≤ 0")
    print("  （能量已减去摆静止在最低点的基线，因此从 1 衰减到 0）\n")

    print(f"  {'damping':>8}{'I(kg·m²)':>11}{'10s后可用能量':>15}"
          f"{'能量增加步数':>13}{'最大单步ΔE':>13}  判定")
    results = []
    for damping in (0.0, 0.01, 0.05, 0.2):
        ts, ex, ang, inertia = simulate(damping)
        de = np.diff(ex)
        n_up = int((de > 1e-12).sum())
        max_up = float(de.max())
        results.append(dict(damping=damping, ts=ts, excess=ex, angles=ang,
                            inertia=inertia))
        if damping == 0:
            drift = abs(1.0 - ex[-1])
            verdict = (f"✅ 物理守恒（数值漂移 {drift * 100:.2f}%）"
                       if drift < 0.01 else f"⚠️ 数值漂移 {drift * 100:.1f}%")
        else:
            verdict = "❌ 物理耗散（单调下降）"
        print(f"  {damping:>8.2f}{inertia:>11.4f}{ex[-1]:>15.4f}"
              f"{n_up:>13d}{max_up:>13.1e}  {verdict}")

    print("""
  读表：
    * damping=0：能量基本不变，只剩积分器带来的微小数值漂移，物理上守恒
    * damping>0：可用能量单调下降；「能量增加步数=0」说明每一步都在减少
    * damping 越大掉得越快：0.2 的摆 10 秒后几乎完全停下""")

    # ---------------- 理论 vs 实测 ----------------
    common.section("为什么是指数衰减？用 I/c 预测")
    r = results[2]                            # damping = 0.05
    damping, inertia = r["damping"], r["inertia"]
    tau_theory = inertia / damping
    mask = (r["ts"] > 2.0) & (r["excess"] > 1e-6)   # 跳过起始瞬态
    slope, _ = np.polyfit(r["ts"][mask], np.log(r["excess"][mask]), 1)
    tau_measured = -1.0 / slope
    print(f"  damping = {damping}, I = {inertia:.4f} kg·m²")
    print(f"  理论能量时间常数  tau = I/c = {tau_theory:.2f} s")
    print(f"  实测拟合时间常数       = {tau_measured:.2f} s  "
          f"（误差 {abs(tau_measured - tau_theory) / tau_theory * 100:.1f}%）")
    print(f"  10 s 后理论值 = {np.exp(-10 / tau_theory):.3f}，"
          f"实测值 = {r['excess'][-1]:.3f}")
    print("""
  推导（轻阻尼近似）：
    E = ½ I θ̇² + ½ k θ²,        dE/dt = -c θ̇²
    简谐振动里 ⟨θ̇²⟩ = (k/I)⟨θ²⟩，且 E ≈ k⟨θ²⟩
    → ⟨dE/dt⟩ = -(c/I)·E     ⇒   E(t) = E0 · exp(-t·c/I)
  所以能量按 tau = I/c 指数衰减（振幅按 2I/c 衰减，包络衰减慢一倍）。""")

    # ---------------- 数值 vs 物理 ----------------
    common.section("别把「物理耗散」和「数值误差」搞混")
    for integ, dt in (("implicitfast", 0.002), ("Euler", 0.02)):
        ts2, ex2, _, _ = simulate(0.05, integ=integ, dt=dt, duration=6.0)
        de = np.diff(ex2)
        print(f"  damping=0.05, {integ:<13} dt={dt:<6} → 6s 后可用能量 "
              f"{ex2[-1]:.3f}，能量增加步数={int((de > 1e-12).sum())}")
    print("""
  即使换成粗糙的 Euler + 大 dt，只要阻尼存在，能量依然整体下降：
  耗散是物理事实，数值误差在这里只是小扰动。
  反过来，damping=0 时同样的 Euler + dt=0.02 会**凭空涨能量**
  （这是第 2 课实验 A 的结论：没有物理耗散时，数值误差就暴露出来了）。""")

    # ---------------- 出图 ----------------
    try:
        plt, font = common.setup_plot()
        tr = (lambda zh, en: zh) if font else (lambda zh, en: en)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for r in results:
            label = f"damping={r['damping']}"
            ax1.plot(r["ts"], r["excess"], lw=1.5, label=label)
            ax2.semilogy(r["ts"], np.maximum(r["excess"], 1e-4), lw=1.5, label=label)

        r = results[2]
        tt = np.linspace(0, r["ts"][-1], 300)
        ax2.semilogy(tt, np.exp(-tt / (r["inertia"] / 0.05)), "k--", lw=1,
                     label=tr("理论 exp(-t·c/I)", "theory exp(-t·c/I)"))

        ax1.set(xlabel="time (s)",
                ylabel=tr("可用机械能比", "available energy / E0"),
                title=tr("加阻尼后能量单调下降",
                         "Damping makes energy decay monotonically"))
        ax2.set(xlabel="time (s)",
                ylabel=tr("可用机械能比 (log)", "energy ratio (log)"),
                title=tr("半对数坐标：直线 = 指数衰减",
                         "Log scale: straight line = exponential decay"))
        common.plain_log_ticks(ax2)      # 对数刻度用纯文本，避免中文字体缺字形
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = common.OUT / "ex_02a_damping.png"
        fig.savefig(path, dpi=120)
        print(f"\n  [png] {path}")
    except ImportError:
        print("\n  （没有 matplotlib，跳过曲线图）")


if __name__ == "__main__":
    main()
