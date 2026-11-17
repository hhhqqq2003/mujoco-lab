"""
练习 02(b) —— 怎么给小球加空气阻力？

MuJoCo 里没有现成的「空气阻力」开关（这个版本也不支持 fluidshape），
但有三条路，从粗糙到物理：

  ① 关节阻尼（一行搞定，线性阻力 / Stokes）
        <joint type="free" damping="0.03"/>
     力 = -c·v，各方向同性，简单但**不是真实空气阻力**（真实阻力 ∝ v²）。

  ② 二次空气阻力（推荐，物理正确）
        F_drag = -½·ρ·Cd·A·|v|·v
     通过每步写 data.qfrc_applied 施加；可以挂在控制回调里，
     viewer 里也能自动生效。

  ③ 被动力回调 set_mjcb_passive
     和 ② 等价，但属于"被动力"，在求解器算约束前就并进去，更正规。

本脚本让三个一模一样的球（x = -0.6 / 0 / +0.6）从 2 m 高同时下落：
    左：无阻力   中：线性阻尼 c=0.03   右：二次空气阻力（真实公式）
比较落地时间、最大速度、能量，并生成对比 GIF。

跑法：
  python3 ex_02b_air_drag.py            # 跑数值对比 + 出 GIF
  python3 ex_02b_air_drag.py --viewer   # 打开窗口看三球下落
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

import common

# --------------------------------------------------------------------------- #
# 物理参数（标准空气）
# --------------------------------------------------------------------------- #
RHO_AIR = 1.225          # kg/m³  空气密度
CD_SPHERE = 0.47         # 球体的阻力系数
BALL_R = 0.08            # m      球半径
BALL_M = 0.005           # kg     球质量（很轻，像气球，让阻力效果明显）
GRAVITY = 9.81

BALL_AREA = math.pi * BALL_R ** 2          # 迎风面积 m²
K_QUAD = 0.5 * RHO_AIR * CD_SPHERE * BALL_AREA   # F = -K_QUAD·|v|·v
K_LINEAR = 0.017        # 线性阻尼 c：取 mg/v_t，让终端速度与二次阻力一致，方便对比
V_TERMINAL = math.sqrt(BALL_M * GRAVITY / K_QUAD)

# 三个球：名字 / x 位置 / 说明
BALLS = [
    ("ball_free", -0.6, "无阻力（真空）"),
    ("ball_linear", 0.0, f"线性阻尼 c={K_LINEAR}"),
    ("ball_drag", 0.6, "二次空气阻力 0.5ρCdA|v|v"),
]

XML = f"""
<mujoco model="air_drag">
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 4"/>
    <geom name="ground" type="plane" size="3 3 0.1"
          rgba="0.25 0.28 0.32 1" friction="0.7 0.1 0.1"/>

    <!-- ① 无阻力：标准 freejoint -->
    <body name="ball_free" pos="-0.6 0 2.0">
      <freejoint name="ball_free_joint"/>
      <geom name="geom_free" type="sphere" size="{BALL_R}" mass="{BALL_M}"
            rgba="0.90 0.35 0.25 1"/>
    </body>

    <!-- ② 线性阻尼：把 freejoint 写成 joint type="free" 并加 damping -->
    <body name="ball_linear" pos="0 0 2.0">
      <joint type="free" name="ball_linear_joint" damping="{K_LINEAR}"/>
      <geom name="geom_linear" type="sphere" size="{BALL_R}" mass="{BALL_M}"
            rgba="0.95 0.70 0.20 1"/>
    </body>

    <!-- ③ 二次空气阻力：XML 里不写，交给 Python 回调 -->
    <body name="ball_drag" pos="0.6 0 2.0">
      <freejoint name="ball_drag_joint"/>
      <geom name="geom_drag" type="sphere" size="{BALL_R}" mass="{BALL_M}"
            rgba="0.25 0.65 0.95 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def build():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def dof_address(model, joint_name: str) -> int:
    """取某个关节在 qvel 里的起始下标（free joint 占 6 个）。"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(np.ravel(model.joint(jid).dofadr)[0])


def apply_quadratic_drag(model, data) -> None:
    """
    二次空气阻力：F = -½·ρ·Cd·A·|v|·v

    写进 data.qfrc_applied（广义外力）。free joint 的前 3 个 dof 是平移，
    所以把力直接写进 [dof, dof+1, dof+2]，方向与速度相反、大小 ∝ v²。
    """
    dof = dof_address(model, "ball_drag_joint")
    v = data.qvel[dof:dof + 3]
    speed = float(np.linalg.norm(v))
    data.qfrc_applied[dof:dof + 3] = -K_QUAD * speed * v


def install_drag_callback(model) -> None:
    """
    把阻力挂成控制回调（mj_step 里每步自动调用）。

    好处：你不用自己写仿真循环，mujoco.viewer 里也会自动生效。
    """
    def _callback(m, d):
        apply_quadratic_drag(m, d)

    mujoco.set_mjcb_control(_callback)


def simulate(duration: float = 2.5, verbose: bool = True):
    """跑一段仿真，返回每个球的轨迹数据。"""
    model, data = build()
    install_drag_callback(model)

    dofs = {name: dof_address(model, f"{name}_joint") for name, _, _ in BALLS}
    ts, z, speed = {n: [] for n, _, _ in BALLS}, {n: [] for n, _, _ in BALLS}, \
        {n: [] for n, _, _ in BALLS}
    landing = {n: None for n, _, _ in BALLS}          # 落地时刻
    landing_idx = {n: None for n, _, _ in BALLS}      # 落地发生在第几步

    for _ in range(int(duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
        for name, _, _ in BALLS:
            v = data.qvel[dofs[name]:dofs[name] + 3]
            ts[name].append(data.time)
            z[name].append(float(data.body(name).xpos[2]))
            speed[name].append(float(np.linalg.norm(v)))
            if landing[name] is None and data.body(name).xpos[2] <= BALL_R + 1e-3:
                landing[name] = data.time
                landing_idx[name] = len(speed[name]) - 1   # 接触前的最后一帧

    mujoco.set_mjcb_control(None)      # 清理回调，避免影响后续实验
    return model, data, ts, z, speed, landing, landing_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="开窗实时看三球下落")
    args = parser.parse_args()

    if args.viewer:
        model, data = build()
        install_drag_callback(model)
        common.relaunch_under_mjpython()
        print("三个球同时从 2 m 下落：红=无阻力，黄=线性阻尼，蓝=二次空气阻力")
        print("（二次阻力已通过 set_mjcb_control 挂钩，窗口里会自动生效）\n")
        common.run_viewer(model, data, max_seconds=10)
        return

    common.banner("空气阻力对比实验")
    print(f"  球：半径 {BALL_R} m，质量 {BALL_M} kg，迎风面积 A = {BALL_AREA:.5f} m²")
    print(f"  空气：ρ = {RHO_AIR} kg/m³，Cd = {CD_SPHERE}")
    print(f"  二次阻力系数 k = ½ρCdA = {K_QUAD:.5f} N/(m/s)²")
    print(f"  → 理论终端速度 v_t = √(mg/k) = {V_TERMINAL:.2f} m/s")
    print(f"  → 线性阻尼取 c = {K_LINEAR} N/(m/s)（终端速度与二次阻力相同，方便对比）\n")

    model, data, ts, z, speed, landing, landing_idx = simulate()

    print(f"  {'阻力模型':<26}{'落地时间(s)':>13}{'最大速度':>11}"
          f"{'落地速度':>11}{'落地动能/初始势能':>17}")
    for name, _, desc in BALLS:
        t_land = landing[name]
        v_max = max(speed[name])
        # 撞击前一帧的速度（接触那一步速度会被接触力改掉）
        i = landing_idx[name]
        v_end = speed[name][max(i - 1, 0)] if i is not None else speed[name][-1]
        ke_ratio = (0.5 * BALL_M * v_end ** 2) / (BALL_M * GRAVITY * 2.0)
        print(f"  {desc:<26}{(t_land if t_land else float('nan')):>13.3f}"
              f"{v_max:>11.2f}{v_end:>11.2f}{ke_ratio:>17.3f}")

    v_ideal = math.sqrt(2 * GRAVITY * (2.0 - BALL_R))   # 球心从 2 m 落到 0.08 m
    print(f"""
  解读：
    * 无阻力：自由落体，落地速度 √(2gh) ≈ {v_ideal:.2f} m/s，
      落地动能 ≈ 初始势能的 96%，最快落地（{landing["ball_free"]:.2f} s）
    * 线性阻尼 c·v：终端速度 = mg/c = {BALL_M * GRAVITY / K_LINEAR:.2f} m/s，
      但它是"指数逼近"，落地时（{landing["ball_linear"]:.2f} s）才达到 {speed["ball_linear"][landing_idx["ball_linear"]-1]:.2f} m/s
    * 二次阻力 ½ρCdA·v²：落地时 {speed["ball_drag"][landing_idx["ball_drag"]-1]:.2f} m/s，
      几乎正好是理论终端速度 {V_TERMINAL:.2f} m/s —— 雨滴、跳伞、纸片"匀速下落"就是这个机制
    * 为什么二次阻力反而比线性阻尼更早落地（{landing["ball_drag"]:.2f} s vs {landing["ball_linear"]:.2f} s）？
      因为低速时 v² 很小、阻力很弱 → 前段加速更快；高速时才被强阻力刹住 → 更接近真实空气
    * 空气阻力把 80% 的机械能变成了热 → 落地动能只剩约 20%""")

    # ---------------- 出图 ----------------
    try:
        plt, font = common.setup_plot()
        tr = (lambda zh, en: zh) if font else (lambda zh, en: en)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        colors = {"ball_free": "#d85b2b", "ball_linear": "#c9a227",
                  "ball_drag": "#2b8bd8"}
        for name, _, desc in BALLS:
            ax1.plot(ts[name], z[name], lw=1.8, color=colors[name],
                     label=tr(desc, name))
            ax2.plot(ts[name], speed[name], lw=1.8, color=colors[name],
                     label=tr(desc, name))
        ax2.axhline(V_TERMINAL, color="k", ls=":", lw=1.2,
                    label=tr(f"理论终端速度 {V_TERMINAL:.1f} m/s",
                             f"terminal velocity {V_TERMINAL:.1f} m/s"))
        ax1.set(xlabel="time (s)", ylabel=tr("高度 (m)", "height (m)"),
                title=tr("从 2 m 同时下落", "Dropped from 2 m"))
        ax2.set(xlabel="time (s)", ylabel=tr("速度 (m/s)", "speed (m/s)"),
                title=tr("速度曲线：二次阻力趋向终端速度",
                         "Speed: quadratic drag approaches terminal velocity"))
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = common.OUT / "ex_02b_air_drag.png"
        fig.savefig(path, dpi=120)
        print(f"\n  [png] {path}")
    except ImportError:
        print("\n  （没有 matplotlib，跳过曲线图）")

    # ---------------- GIF：三球同场下落 ----------------
    print("\n  正在渲染 GIF（三球同场下落）...")
    model, data = build()
    install_drag_callback(model)
    frames = common.render_frames(
        model, data, n_frames=60, steps_per_frame=20,
        camera=common.make_camera(azimuth=120, elevation=-10,
                                  distance=4.4, lookat=(0, 0, 1.05)),
        width=640, height=400)
    common.save_gif(frames, "ex_02b_air_drag", fps=30)

    common.banner("怎么用到自己的模型里")
    print("""
  ① 只想要"有点阻力"的近似（一行）：
         <joint type="free" name="ball_free" damping="0.03"/>
     （注意：这个版本 <freejoint> 不支持 damping 属性，要写成 <joint type="free">）

  ② 要物理正确的二次阻力（复制这段即可）：
         RHO, Cd, A = 1.225, 0.47, math.pi * r ** 2
         K = 0.5 * RHO * Cd * A

         def air_drag(model, data):
             dof = model.joint("ball_free_joint").dofadr[0]      # free joint 的平移 dof
             v = data.qvel[dof:dof + 3]
             data.qfrc_applied[dof:dof + 3] = -K * np.linalg.norm(v) * v

         mujoco.set_mjcb_control(lambda m, d: air_drag(m, d))    # 每步自动执行
         # 或者在你的仿真循环里，每次 mj_step 之前手动调用 air_drag(model, data)

  ③ 想更正规（被动力，求解器约束前并入）：用 mujoco.set_mjcb_passive(回调)，写法同上。

  别忘了：qfrc_applied 是广义外力，单位是"力矩/力"，free joint 前 3 个直接对应 xyz 力。
""")


if __name__ == "__main__":
    main()
