"""
04 —— 小车倒立摆：用 MuJoCo 做线性化 + LQR 平衡

这是控制课的"hello world"，也是具身智能里 model-based 控制的起点。会学到：
  1. mjd_transitionFD：MuJoCo 内置的有限差分，直接给你 A、B 矩阵（不用手推导数）
  2. 线性化必须发生在平衡点；倒立摆的平衡点是"立着不动"这个不稳定点
  3. 解 Riccati 方程拿 LQR 增益 K，u = -Kx 就能立住
  4. 有反馈控制后，倒立摆能扛住外部扰动（随机推）

跑法：python3 04_cartpole_lqr.py [--viewer]
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

import common

XML = """
<mujoco model="cartpole">
  <!-- 注意：mjd_transitionFD 不支持 RK4，只支持 Euler/implicit/implicitfast -->
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 4"/>
    <!-- 导轨：纯视觉，不参与碰撞 -->
    <geom name="rail" type="box" size="2.5 0.02 0.02" pos="0 0 0.14"
          contype="0" conaffinity="0" rgba="0.35 0.35 0.4 1"/>
    <body name="cart" pos="0 0 0.2">
      <joint name="slider" type="slide" axis="1 0 0" damping="0.05"/>
      <geom name="cart_geom" type="box" size="0.12 0.08 0.06" mass="1.0"
            rgba="0.9 0.5 0.2 1" contype="0" conaffinity="0"/>
      <geom name="wheel1" type="sphere" size="0.04" pos="-0.08 0.09 -0.06" mass="0.02"
            rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      <geom name="wheel2" type="sphere" size="0.04" pos="-0.08 -0.09 -0.06" mass="0.02"
            rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      <geom name="wheel3" type="sphere" size="0.04" pos="0.08 0.09 -0.06" mass="0.02"
            rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      <geom name="wheel4" type="sphere" size="0.04" pos="0.08 -0.09 -0.06" mass="0.02"
            rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      <body name="pole" pos="0 0 0.06">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.002"/>
        <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.02"
              mass="0.2" rgba="0.2 0.6 0.95 1" contype="0" conaffinity="0"/>
        <geom name="pole_tip" type="sphere" size="0.05" pos="0 0 0.6" mass="0.1"
              rgba="0.95 0.6 0.2 1" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- 给小车施加水平力（牛顿） -->
    <motor name="force" joint="slider" gear="1" ctrlrange="-15 15"/>
  </actuator>
</mujoco>
"""

# 状态量：x = [小车位置, 摆角, 小车速度, 摆角速度]；输入 u = [水平力]
X_LABELS = ["x_cart", "theta", "v_cart", "omega"]
U_MAX = 15.0


def linearize(model, data):
    """
    在 data 当前状态处做有限差分线性化，返回连续时间 A、B。

    mjd_transitionFD 给的是"走一步之后"的离散雅可比：
        x_{k+1} ≈ A_d x_k + B_d u_k
    转成连续时间：A_c = (A_d - I) / dt,  B_c = B_d / dt
    """
    mujoco.mj_forward(model, data)
    nv, nu = model.nv, model.nu
    A_d = np.zeros((2 * nv, 2 * nv))
    B_d = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data, 1e-5, True, A_d, B_d, None, None)

    dt = model.opt.timestep
    return (A_d - np.eye(2 * nv)) / dt, B_d / dt


def lqr_gain(A, B, Q, R):
    """解连续时间 Riccati 方程：AᵀP + PA - PBR⁻¹BᵀP + Q = 0，然后 K = R⁻¹BᵀP。"""
    from scipy.linalg import solve_continuous_are
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


def state_of(data) -> np.ndarray:
    """cartpole 的 nq = nv = 2，所以状态就是 qpos ++ qvel。"""
    return np.concatenate([data.qpos, data.qvel])


def simulate(K, duration: float = 6.0, theta0: float = 0.2,
             push_times=(3.0,), seed: int = 0):
    """从 theta0 扰动出发，用 u = -Kx 稳定。push_times 里给随机横向推力。"""
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[1] = theta0
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(seed)
    ts, states, forces = [], [], []
    for _ in range(int(duration / model.opt.timestep)):
        x = state_of(data)
        u = 0.0 if K is None else float(np.clip(float(-K[0] @ x), -U_MAX, U_MAX))
        data.ctrl[0] = u
        # 在指定时刻给一个 20 N 的横向推力，持续 0.1 s
        data.qfrc_applied[0] = 0.0
        if any(abs(data.time - tp) < 0.05 for tp in push_times):
            data.qfrc_applied[0] = 20.0 * rng.choice([-1.0, 1.0])
        mujoco.mj_step(model, data)
        ts.append(data.time)
        states.append(state_of(data))
        forces.append(u)
    return np.array(ts), np.array(states), np.array(forces)


def settle_time(ts, states) -> float:
    """摆角进入 ±0.05 rad 并且不再出来的时刻。"""
    ok = np.abs(states[:, 1]) < 0.05
    idx = np.where(~ok)[0]
    return float(ts[idx[-1] + 1]) if len(idx) and idx[-1] + 1 < len(ts) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="打开交互窗口实时看")
    args = parser.parse_args()

    # ---------------- 线性化 ----------------
    common.banner("第 1 步：在平衡点做线性化（MuJoCo 直接给 A、B）")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)          # qpos = qvel = 0 就是"立着不动"的平衡点
    mujoco.mj_forward(model, data)

    A, B = linearize(model, data)
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print("  状态顺序：", X_LABELS)
    print("\n  连续时间 A =\n", A)
    print("\n  连续时间 B =\n", B)
    print(f"\n  A 的特征值 = {np.round(np.linalg.eigvals(A), 3)}")
    print("  → 有一对 ± 的实特征值，说明平衡点是不稳定的（摆会倒）。")

    # ---------------- LQR ----------------
    common.banner("第 2 步：LQR —— 求解 Q、R 权衡下的最优增益")
    Q = np.diag([5.0, 50.0, 0.5, 0.5])   # 更在意摆角（theta）和位置
    R = np.array([[1.0]])                # 力越大越"贵"
    K = lqr_gain(A, B, Q, R)
    print(f"  Q = diag({np.diag(Q)}),  R = {R[0, 0]}")
    print(f"\n  LQR 增益 K = {np.round(K, 3)}")
    print("  控制律 u = -Kx = "
          f"{-K[0, 0]:+.2f}·x {-K[0, 1]:+.2f}·theta "
          f"{-K[0, 2]:+.2f}·v {-K[0, 3]:+.2f}·omega")
    poles = np.linalg.eigvals(A - B @ K)
    print(f"\n  闭环极点的实部 = {np.round(poles.real, 3)}  （全为负 → 稳定）")

    # ---------------- 对比：不控制 vs LQR ----------------
    common.banner("第 3 步：对比 —— 不控制会倒，LQR 能立住并扛住推力")
    ts0, st0, _ = simulate(None)
    ts1, st1, u1 = simulate(K, push_times=(3.0, 5.0))

    def rmse(states):
        return float(np.sqrt(np.mean(states[:, 1] ** 2)))

    print(f"  不控制：6 s 后摆角 = {st0[-1, 1]:+.3f} rad（已经倒下），"
          f"摆角 RMS = {rmse(st0):.3f}")
    print(f"  LQR   ：6 s 后摆角 = {st1[-1, 1]:+.3f} rad，摆角 RMS = {rmse(st1):.3f}，"
          f"最大力 = {np.abs(u1).max():.1f} N")
    print(f"  → 3 s 和 5 s 各来一次 20 N 的推力，LQR 都恢复了")
    print(f"  → 稳定时间（摆角进入 ±0.05 rad）= {settle_time(ts1, st1):.2f} s")

    print("\n  参考：把 Q 里的 theta 权重从 50 改到 500，K 的摆角项会明显变大，"
          "\n        相当于更用力地扶正 —— 这就是 Q/R 权重的直觉。")

    if args.viewer:
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        data.qpos[1] = 0.15
        mujoco.mj_forward(model, data)

        def control_loop(m, d, t):
            u = float(np.clip(-K @ state_of(d), -U_MAX, U_MAX))
            d.ctrl[0] = u
            d.qfrc_applied[0] = 0.0
            if abs(t - 3.0) < 0.05 or abs(t - 6.0) < 0.05:
                d.qfrc_applied[0] = 20.0

        print("\n打开交互窗口（3 s、6 s 各推一下）……关窗口结束。")
        common.run_viewer(model, data, controller=control_loop, max_seconds=12)
        return

    # ---------------- 出图 ----------------
    try:
        plt, font = common.setup_plot()
        tr = (lambda zh, en: zh) if font else (lambda zh, en: en)

        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(ts1, st1[:, 1], lw=1.5, color="#27a35b", label="LQR")
        axes[0].plot(ts0, st0[:, 1], lw=1.2, color="#d85b2b", alpha=0.8,
                     label=tr("不控制", "no control"))
        axes[0].axhline(0.05, color="k", ls=":", lw=0.8)
        axes[0].axhline(-0.05, color="k", ls=":", lw=0.8)
        axes[0].set_ylabel(tr("摆角 (rad)", "pole angle (rad)"))
        axes[0].set_title(tr("小车倒立摆：LQR 稳定 + 抗扰动",
                             "Cart-pole: LQR stabilization and disturbance rejection"))
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=9)

        for tp in (3.0, 5.0):
            axes[1].axvline(tp, color="#d85b2b", ls="--", lw=0.8, alpha=0.7)
        axes[1].plot(ts1, st1[:, 0], lw=1.4, color="#2b8bd8")
        axes[1].set_ylabel(tr("小车位置 (m)", "cart position (m)"))
        axes[1].grid(alpha=0.3)

        axes[2].plot(ts1, u1, lw=1.0, color="#7a51c8")
        axes[2].set_ylabel(tr("控制力 (N)", "force (N)"))
        axes[2].set_xlabel("time (s)")
        axes[2].grid(alpha=0.3)
        fig.tight_layout()
        path = common.OUT / "04_cartpole_lqr.png"
        fig.savefig(path, dpi=120)
        print(f"\n  [png] {path}")
    except ImportError:
        print("\n  （没有 matplotlib，跳过曲线图）")

    # ---------------- GIF ----------------
    print("\n  正在渲染 GIF（LQR 抗扰动）...")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[1] = 0.2
    mujoco.mj_forward(model, data)

    def set_ctrl(m, d, t):
        d.ctrl[0] = float(np.clip(float(-K[0] @ state_of(d)), -U_MAX, U_MAX))
        d.qfrc_applied[0] = 20.0 if (abs(t - 1.5) < 0.05 or abs(t - 2.5) < 0.05) else 0.0

    frames = common.render_frames(
        model, data, n_frames=110, steps_per_frame=20,
        camera=common.make_camera(azimuth=100, elevation=-8,
                                  distance=3.2, lookat=(0, 0, 0.5)),
        controller=set_ctrl, width=480, height=360)
    common.save_gif(frames, "04_cartpole_lqr", fps=30)

    common.banner("要点回顾")
    print("  1. mjd_transitionFD 一句话拿到 A、B —— 不用自己推导动力学方程")
    print("  2. LQR 解的是「Q 惩罚状态偏差、R 惩罚控制量」的二次型最优问题")
    print("  3. 线性控制器只在线性化点附近可靠；摆角太大（比如 1 rad）就会失效")
    print("\n练习题：")
    print("  (a) 把 Q 的 theta 权重从 50 改成 500，K 的摆角项变了多少？稳定性变好了没有？")
    print("  (b) 把 R 从 1 改到 0.1（控制更便宜），看最大控制力变大多少。")
    print("  (c) 把初始摆角改成 0.8 rad，LQR 还能救回来吗？如果不能，想想为什么。")


if __name__ == "__main__":
    main()
