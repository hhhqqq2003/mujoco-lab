"""
07 —— 不用梯度、不推公式：用 CEM 从零“学”出倒立摆控制器

这一课把前面所有的东西串起来，也是通向强化学习的桥梁。会学到：
  1. 交叉熵方法（CEM）：采样一批参数 → 挑最好的 → 更新分布 → 重复
  2. 把控制器写成参数化策略 u = -Kx，让优化器自己找 K（不用解 Riccati 方程）
  3. 【关键】参数缩放：直接搜 K 的 4 个数很难收敛，按物理量纲缩放后才好学
  4. 与第 4 课的 LQR 对比：学到的 K 有多接近解析最优？为什么还可能更好？
  5. 这就是 model-free 的雏形：优化器只看到“跑一遍得多少分”，完全不懂动力学

跑法：python3 07_cem_learning.py
（纯 numpy + MuJoCo，大约 10 秒跑完）
"""

from __future__ import annotations

import mujoco
import numpy as np

import common

XML = """
<mujoco model="cartpole_cem">
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 4"/>
    <geom name="rail" type="box" size="2.5 0.02 0.02" pos="0 0 0.14"
          contype="0" conaffinity="0" rgba="0.35 0.35 0.4 1"/>
    <body name="cart" pos="0 0 0.2">
      <joint name="slider" type="slide" axis="1 0 0" damping="0.05"/>
      <geom name="cart_geom" type="box" size="0.12 0.08 0.06" mass="1.0"
            rgba="0.9 0.5 0.2 1" contype="0" conaffinity="0"/>
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
    <motor name="force" joint="slider" gear="1" ctrlrange="-15 15"/>
  </actuator>
</mujoco>
"""

U_MAX = 15.0
EPISODE_TIME = 2.0          # 每个 episode 2 秒
FALL_ANGLE = 0.6            # 摆角超过这个就判失败

# 策略参数的"物理量纲"缩放：z 是优化器搜的归一化参数，K = z * SCALES
# 这样 z 大致都在 [-2, 2] 之间，CEM 才学得动（直接搜 K 的话，摆角增益要到 -40 量级）
SCALES = np.array([10.0, 40.0, 10.0, 10.0])


def make_model():
    return mujoco.MjModel.from_xml_string(XML)


def run_episode(K, theta0: float = 0.1, duration: float = EPISODE_TIME,
                record: bool = False):
    """
    用反馈律 u = -K @ x 跑一个 episode，返回 (总代价, 轨迹记录)。

    这里的“控制律”就是参数化策略，K 是待优化的 4 个参数。
    这是最朴素的 policy optimization：策略 → rollout → 累计代价。
    """
    model = make_model()
    data = mujoco.MjData(model)
    data.qpos[1] = theta0
    mujoco.mj_forward(model, data)

    K = np.asarray(K, dtype=float).ravel()
    cost = 0.0
    traj = {"t": [], "theta": [], "x": [], "u": []}
    for _ in range(int(duration / model.opt.timestep)):
        state = np.concatenate([data.qpos, data.qvel])
        u = float(np.clip(float(-K @ state), -U_MAX, U_MAX))
        data.ctrl[0] = u
        mujoco.mj_step(model, data)

        theta = float(data.qpos[1])
        # 代价：主要惩罚摆角，其次小车位置，最后控制能量
        cost += theta ** 2 + 0.05 * float(data.qpos[0]) ** 2 + 0.001 * u ** 2

        if record:
            traj["t"].append(data.time)
            traj["theta"].append(theta)
            traj["x"].append(float(data.qpos[0]))
            traj["u"].append(u)

        if abs(theta) > FALL_ANGLE:     # 倒了：按剩余时间加倍惩罚
            cost += 50.0 * (duration - data.time)
            break
    return cost, traj


def cem(iterations: int = 16, n_samples: int = 24, n_elite: int = 6,
        init_std: float = 1.0, seed: int = 0, verbose: bool = True):
    """最标准的 CEM 循环：在缩放后的参数空间里搜索。"""
    rng = np.random.default_rng(seed)
    mean = np.zeros(4)                  # z 的均值
    std = np.full(4, init_std)          # z 的标准差
    init_thetas = np.array([0.12, -0.12, 0.06])   # 每个样本测 3 个初始摆角，降低方差
    history = []

    for it in range(iterations):
        z = rng.normal(mean, std, size=(n_samples, 4))
        # 1) 采样 → 2) 评估：每个 z 跑 3 个 episode 取平均
        costs = np.array([
            np.mean([run_episode(zz * SCALES, theta0=th)[0] for th in init_thetas])
            for zz in z])
        # 3) 选精英 → 4) 平滑更新分布
        idx = np.argsort(costs)[:n_elite]
        elite_z = z[idx]
        elite_mean, elite_std = elite_z.mean(axis=0), elite_z.std(axis=0) + 1e-3
        mean = 0.7 * mean + 0.3 * elite_mean      # 平滑系数，避免分布塌缩太快
        std = 0.7 * std + 0.3 * elite_std
        history.append((float(costs.min()), float(costs[idx].mean()), mean * SCALES))

        if verbose and (it % 4 == 0 or it == iterations - 1):
            print(f"  第 {it:2d} 代：本代最好代价 = {costs.min():6.2f}，"
                  f"精英平均 = {costs[idx].mean():6.2f}")
            print(f"          K ≈ {np.round(mean * SCALES, 1)}")

    return mean * SCALES, history


def lqr_gain() -> np.ndarray:
    """第 4 课的方法：线性化 + 解 Riccati 方程，作为“参考答案”。"""
    from scipy.linalg import solve_continuous_are

    model = make_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    nv, nu = model.nv, model.nu
    A_d = np.zeros((2 * nv, 2 * nv))
    B_d = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data, 1e-5, True, A_d, B_d, None, None)
    dt = model.opt.timestep
    A, B = (A_d - np.eye(2 * nv)) / dt, B_d / dt
    Q, R = np.diag([5.0, 50.0, 0.5, 0.5]), np.array([[1.0]])
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P).ravel()


def main() -> None:
    common.banner("问题设定")
    print("  小车倒立摆：状态 x = [小车位置, 摆角, 小车速度, 摆角速度]")
    print("  策略形式：u = -K·x，K 是 4 个待优化的数")
    print(f"  代价：Σ (摆角² + 0.05·位置² + 0.001·力²)，episode = {EPISODE_TIME} s")
    print(f"  初值：摆角在 ±0.15 rad 随机，|摆角| > {FALL_ANGLE} 直接判失败")
    print("\n  优化器完全不懂动力学方程 —— 它只看得到每个 K 跑出多少分。")

    cost_zero, _ = run_episode(np.zeros(4))
    print(f"\n  基线：K = 0（不控制）代价 = {cost_zero:.1f}（开局就倒）")

    common.banner("开始 CEM：采样 → 打分 → 选精英 → 更新分布")
    K_cem, history = cem()
    K_lqr = lqr_gain()

    common.banner("对比：CEM 学到的 K vs LQR 的解析解")
    print(f"  CEM 学到的 K = {np.round(K_cem, 2)}")
    print(f"  LQR 解析的 K = {np.round(K_lqr, 2)}")
    print(f"  两个 K 的夹角/相对差异 = "
          f"{np.linalg.norm(K_cem - K_lqr) / np.linalg.norm(K_lqr) * 100:.0f}%")

    thetas = np.linspace(-0.15, 0.15, 7)
    cost_cem = np.mean([run_episode(K_cem, theta0=th)[0] for th in thetas])
    cost_lqr = np.mean([run_episode(K_lqr, theta0=th)[0] for th in thetas])
    print(f"\n  同一批初始状态下的平均代价：CEM = {cost_cem:.2f}，LQR = {cost_lqr:.2f}")

    print("""
  解读：
    * CEM 完全不知道 A、B 矩阵，也不知道 Riccati 方程，却学出了和 LQR 同量级的增益
    * 它甚至可能"更好"—— 因为我们评估的代价函数（摆角² + 0.05x² + 0.001u²）
      和 LQR 用的 Q/R（diag(5,50,0.5,0.5)、R=1）不是同一个目标
      → 优化器总是针对"你写的那个代价"优化，这也是 RL 里 reward 设计如此重要的原因
    * 代价：CEM 花了约 1000 次 rollout；LQR 一次矩阵求解。样本效率差 3 个数量级
    * 好处：CEM 不需要模型可微、不需要线性化，换个机器人（四足、机械臂）照样能用""")

    # ---------------- 出图 ----------------
    try:
        plt, font = common.setup_plot()
        tr = (lambda zh, en: zh) if font else (lambda zh, en: en)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        iters = np.arange(1, len(history) + 1)
        ax1.plot(iters, [h[0] for h in history], "o-", lw=1.5, ms=4,
                 label=tr("本代最好", "best in generation"))
        ax1.plot(iters, [h[1] for h in history], "s--", lw=1.2, ms=4,
                 label=tr("精英平均", "elite mean"))
        ax1.axhline(cost_lqr, color="#27a35b", ls=":", lw=1.5, label="LQR")
        ax1.set_xlabel(tr("CEM 代数", "CEM generation"))
        ax1.set_ylabel(tr("episode 代价", "episode cost"))
        ax1.set_title(tr("CEM 学习曲线（对数坐标）", "CEM learning curve (log)"))
        ax1.set_yscale("log")
        ax1.grid(alpha=0.3, which="both")
        ax1.legend(fontsize=9)

        _, traj_cem = run_episode(K_cem, theta0=0.15, record=True)
        _, traj_lqr = run_episode(K_lqr, theta0=0.15, record=True)
        ax2.plot(traj_cem["t"], traj_cem["theta"], lw=1.6,
                 label=tr("CEM 策略", "CEM policy"))
        ax2.plot(traj_lqr["t"], traj_lqr["theta"], "--", lw=1.2, label="LQR")
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel(tr("摆角 (rad)", "pole angle (rad)"))
        ax2.set_title(tr("学到的策略同样能把摆扶正", "Learned policy balances too"))
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9)
        fig.tight_layout()
        path = common.OUT / "07_cem_learning.png"
        fig.savefig(path, dpi=120)
        print(f"\n  [png] {path}")
    except ImportError:
        print("\n  （没有 matplotlib，跳过曲线图）")

    # ---------------- GIF ----------------
    print("\n  正在渲染 GIF（CEM 学到的策略）...")
    model = make_model()
    data = mujoco.MjData(model)
    data.qpos[1] = 0.15
    mujoco.mj_forward(model, data)

    def policy(m, d, t):
        state = np.concatenate([d.qpos, d.qvel])
        d.ctrl[0] = float(np.clip(float(-K_cem @ state), -U_MAX, U_MAX))

    frames = common.render_frames(
        model, data, n_frames=100, steps_per_frame=20,
        camera=common.make_camera(azimuth=100, elevation=-8,
                                  distance=3.2, lookat=(0, 0, 0.5)),
        controller=policy, width=480, height=360)
    common.save_gif(frames, "07_cem_cartpole", fps=30)

    common.banner("要点回顾")
    print("  1. CEM = 采样参数 → 按回报排序 → 用精英更新高斯分布，循环往复")
    print("  2. 参数缩放（把 K 换成 z·SCALES）是让 CEM 能收敛的关键工程技巧")
    print("  3. 只要能参数化控制律，就能用优化代替推导（model-free policy search）")
    print("  4. 代价/reward 的设计比优化器本身更重要：优化器只会忠实地优化你写的东西")
    print("\n练习题：")
    print("  (a) 把 n_samples 从 24 降到 8，CEM 还能收敛吗？需要多少代？")
    print("  (b) 把代价里的位置权重 0.05 改成 5，学出来的 K 有什么变化？")
    print("  (c) 去掉 SCALES 缩放（直接搜 K），看 CEM 会不会卡在初始代价附近。")
    print("  (d) 把策略换成两层小神经网络（numpy 手写前向），这就是进化策略（ES）。")


if __name__ == "__main__":
    main()
