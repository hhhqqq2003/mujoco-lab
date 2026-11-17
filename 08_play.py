"""
08 —— 交互式 playground：在 MuJoCo 原生窗口里亲手玩前面所有场景

前七课都是"跑脚本 → 看 GIF/曲线"。这一课把窗口打开：
  鼠标左键拖动 = 旋转视角
  鼠标右键拖动 = 平移
  滚轮          = 缩放
  空格          = 暂停/继续
  双击          = 选中物体（可看局部坐标）

运行前请确认你是在**自己的桌面 Terminal** 里执行（而不是 SSH/后台），
否则会提示无法创建窗口 —— 这不影响前面所有课程的 GIF 输出。

macOS 提示：交互窗口必须由 mjpython 启动（python3 会报 RuntimeError）。
本脚本已内置自动切换，直接 python3 08_play.py 即可；手动等价命令是
    mjpython 08_play.py

跑法：python3 08_play.py            # 菜单选择场景
     python3 08_play.py 2          # 直接启动第 2 个场景
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import mujoco
import numpy as np

import common

HERE = pathlib.Path(__file__).resolve().parent


def load_lesson(filename: str):
    """从 03_xxx.py 这样的文件名里加载模块（文件名以数字开头，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""),
                                                  HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 每个场景返回 (model, data, controller_or_None, 说明)
# controller(model, data, t) 会被每步调用
# --------------------------------------------------------------------------- #
def scene_pendulum():
    L3 = load_lesson("03_pendulum_pd.py")
    model = mujoco.MjModel.from_xml_string(L3.XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctrl = L3.PDController(kp=40, kd=4.0, gravity_comp=True)

    def controller(m, d, t):
        d.ctrl[0] = ctrl(m, d, L3.reference(t))

    return model, data, controller, \
        "单摆 PD 跟踪：0.9 rad → -0.6 rad，前馈补偿重力（第 3 课）"


def scene_cartpole():
    L4 = load_lesson("04_cartpole_lqr.py")
    model = mujoco.MjModel.from_xml_string(L4.XML)
    data = mujoco.MjData(model)
    data.qpos[1] = 0.15
    mujoco.mj_forward(model, data)

    A, B = L4.linearize(model, data)
    K = L4.lqr_gain(A, B, np.diag([5.0, 50.0, 0.5, 0.5]), np.array([[1.0]]))
    rng = np.random.default_rng(0)

    def controller(m, d, t):
        d.ctrl[0] = float(np.clip(float(-K[0] @ L4.state_of(d)), -L4.U_MAX, L4.U_MAX))
        # 每 3 秒随机推一下，感受 LQR 的恢复能力
        d.qfrc_applied[0] = 18.0 * (1 if int(t / 3) % 2 == 0 else -1) \
            if (t % 3.0) < 0.08 else 0.0

    return model, data, controller, \
        "小车倒立摆 LQR：每 3 秒被随机推一下，看它怎么恢复（第 4 课）"


def scene_arm():
    L5 = load_lesson("05_arm_ik.py")
    model = mujoco.MjModel.from_xml_string(L5.XML)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q_des = data.qpos.copy()

    def controller(m, d, t):
        if int(t / m.opt.timestep) % 2 == 0:
            tgt = L5.target_position(t)
            q_des[:] = L5.solve_ik(m, ik_data, tgt, q_des)
            d.mocap_pos[0] = tgt
            tau = d.qfrc_bias + L5.KP * (q_des - d.qpos) - L5.KD * d.qvel
            d.ctrl[:] = np.clip(tau, -L5.TAU_LIMIT, L5.TAU_LIMIT)

    return model, data, controller, \
        "机械臂逆运动学：红球是 mocap 目标点，末端用雅可比跟踪它（第 5 课）"


def scene_quadruped():
    L6 = load_lesson("06_quadruped_gait.py")
    model, data = L6.make_data(L6.KNEE0)
    L6.settle(model, data, L6.KNEE0)

    def controller(m, d, t):
        L6.trot_control(m, d, t, L6.KNEE0)

    return model, data, controller, \
        "四足原地 trot：对角腿同相，观察四只脚的接触顺序（第 6 课）"


SCENES = {
    1: ("单摆 PD", scene_pendulum),
    2: ("小车倒立摆 LQR", scene_cartpole),
    3: ("机械臂 IK", scene_arm),
    4: ("四足 trot", scene_quadruped),
}


def main() -> None:
    if len(sys.argv) > 1:
        choice = int(sys.argv[1])
    else:
        print("=" * 60)
        print("MuJoCo Lab · 交互式 playground")
        print("=" * 60)
        for k, (name, _) in SCENES.items():
            print(f"  {k}. {name}")
        try:
            choice = int(input("\n选择场景（回车 = 2 小车倒立摆）：") or "2")
        except (EOFError, ValueError):
            choice = 2
    if choice not in SCENES:
        print(f"没有场景 {choice}，可选：{list(SCENES)}")
        return

    name, builder = SCENES[choice]
    model, data, controller, desc = builder()
    print(f"\n场景：{name}")
    print(f"  {desc}")
    print("  操作：左键旋转 / 右键平移 / 滚轮缩放 / 空格暂停 / 双击选中")
    print("  关掉窗口即退出。\n")

    common.relaunch_under_mjpython()   # macOS：交互窗口需要 mjpython，自动切换
    import mujoco.viewer   # 延迟导入，避免无 GUI 环境直接报错
    import time

    dt = model.opt.timestep
    t = 0.0
    t0 = time.time()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if controller is not None:
                    controller(model, data, t)
                mujoco.mj_step(model, data)
                t += dt
                viewer.sync()
                remain = t0 + t - time.time()
                if remain > 0:
                    time.sleep(remain)
    except Exception as exc:    # pragma: no cover
        print(f"\n[!] 无法打开窗口：{type(exc).__name__}: {exc}")
        print("    请在你自己的桌面上、用 Terminal 直接运行：python3 08_play.py")
        print("    没有图形界面也没关系：前面每一课的 out/*.gif 就是仿真录屏。")


if __name__ == "__main__":
    main()
