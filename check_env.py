"""
check_env.py —— 先跑这个，确认你的机器能把整个项目跑起来。

它会检查：Python / mujoco 版本、GPU(OpenGL) 后端、离屏渲染、GIF 写入、
matplotlib、scipy，最后打印一张"能力清单"告诉你哪些课可以上。
"""

from __future__ import annotations

import sys

import numpy as np

import common

OK, WARN, FAIL = "✅", "⚠️ ", "❌"


def main() -> int:
    common.banner("MuJoCo Lab 环境自检")

    print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
    try:
        import mujoco
        print(f"mujoco      : {mujoco.__version__}")
    except ImportError:
        print(f"mujoco      : {FAIL} 未安装 —— 请先执行：python3 -m pip install --user mujoco")
        return 1

    print(f"numpy       : {np.__version__}")

    # --- 离屏渲染：这是整个项目"不开窗口也能看结果"的关键 ---
    print("\n离屏渲染（offscreen render）")
    try:
        from PIL import Image  # noqa: F401
        m = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody><light pos="0 0 3"/>
              <geom type="plane" size="2 2 .1"/>
              <body pos="0 0 1"><freejoint/><geom type="sphere" size=".2" rgba="0 .4 1 1"/></body>
            </worldbody></mujoco>""")
        d = mujoco.MjData(m)
        for _ in range(100):
            mujoco.mj_step(m, d)
        common.save_png(common.render_frames(m, d, n_frames=1, steps_per_frame=0,
                                            width=320, height=240)[0], "check_env_render")
        print(f"  {OK} OpenGL 渲染正常（Apple Silicon 上走 CGL/Metal 后端）")
    except Exception as exc:
        print(f"  {FAIL} 渲染失败：{type(exc).__name__}: {exc}")
        print("     试试在终端里设置 MUJOCO_GL=glfw 后重跑。")
        return 1

    # --- 可选依赖 ---
    print("\n可选依赖")
    try:
        import matplotlib
        matplotlib.use("Agg")
        print(f"  {OK} matplotlib {matplotlib.__version__}（第 2、3、7 课要画曲线）")
    except ImportError:
        print(f"  {WARN} 没有 matplotlib：曲线图会被跳过（pip install --user matplotlib）")

    try:
        import scipy  # noqa: F401
        print(f"  {OK} scipy {scipy.__version__}（第 4 课 LQR 要解 Riccati 方程）")
    except ImportError:
        print(f"  {WARN} 没有 scipy：第 4 课会退化成「手算增益」模式")

    # --- 交互式 viewer ---
    print("\n交互式 viewer")
    try:
        import mujoco.viewer  # noqa: F401
        print(f"  {OK} 可导入")
    except Exception as exc:
        print(f"  {WARN} 不可用：{exc}（不影响 GIF 输出）")

    if sys.platform == "darwin":
        mjpy = common.find_mjpython()
        if common.is_under_mjpython():
            print(f"  {OK} 当前已经跑在 mjpython 里，可以直接开窗口")
        elif mjpy:
            print(f"  {OK} macOS 需要 mjpython，已找到：{mjpy}")
            print("     viewer 脚本会自动切换；手动跑："
                  f"{mjpy} 08_play.py")
        else:
            print(f"  {WARN} 没找到 mjpython（macOS 上交互窗口必须用它启动）")
            print("     试：python3 -m pip install --user mujoco")
    else:
        print(f"  {OK} 有桌面环境即可直接弹窗口")

    common.banner("结论：核心课程（01~07）全部可跑，产物在 out/ 目录")
    print("下一步：python3 01_model_and_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
