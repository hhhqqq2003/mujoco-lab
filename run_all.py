"""
run_all.py —— 一键跑完全部课程，顺便检查你的机器上一切正常。

用法：python3 run_all.py            # 全部跑一遍
     python3 run_all.py 1 3 4     # 只跑第 1、3、4 课
"""

from __future__ import annotations

import subprocess
import sys
import time

LESSONS = [
    ("01_model_and_data.py", "认识 MjModel / MjData"),
    ("02_physics_intuition.py", "积分器 / 接触 / 能量"),
    ("03_pendulum_pd.py", "单摆 PD 控制 + 重力前馈"),
    ("04_cartpole_lqr.py", "倒立摆线性化 + LQR"),
    ("05_arm_ik.py", "机械臂雅可比逆运动学"),
    ("06_quadruped_gait.py", "四足 trot 步态 + 接触分析"),
    ("07_cem_learning.py", "CEM 学控制"),
    ("09_visualize.py", "可视化手册（出图/GIF/深度/分割/FPV）"),
]


def main() -> int:
    want = {int(a) for a in sys.argv[1:] if a.isdigit()}
    failed = []
    t_all = time.time()
    for i, (script, title) in enumerate(LESSONS, start=1):
        if want and i not in want:
            continue
        print(f"\n{'=' * 70}\n>>> [{i}/{len(LESSONS)}] {script}  ——  {title}\n{'=' * 70}", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, script], cwd=__file__.rsplit("/", 1)[0])
        dt = time.time() - t0
        status = "✅" if proc.returncode == 0 else "❌"
        print(f"{status} {script} 完成，用时 {dt:.1f}s", flush=True)
        if proc.returncode != 0:
            failed.append(script)

    print(f"\n{'=' * 70}")
    print(f"总计用时 {time.time() - t_all:.1f}s")
    if failed:
        print(f"❌ 失败：{failed}")
        return 1
    print("✅ 全部课程跑通，产物在 mujoco-lab/out/ 目录")
    print("   交互式体验：python3 08_play.py  （需要桌面环境）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
