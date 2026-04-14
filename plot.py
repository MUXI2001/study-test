import matplotlib.pyplot as plt
import math
import numpy as np
from code_combine.source_domain import source_load
from code_combine.data import dataset2

# 定义初始域
INIT = [[-1 / 15 * math.pi, 1 / 15 * math.pi],
        [-1 / 15 * math.pi, 1 / 15 * math.pi]]

# 定义不安全区域
unsafe_areas = [
    [[-1 / 4 * math.pi, -1 / 6 * math.pi], [-1 / 4 * math.pi, 1 / 4 * math.pi]],  # 区域1
    [[1 / 6 * math.pi, 1 / 4 * math.pi], [-1 / 4 * math.pi, 1 / 4 * math.pi]],  # 区域2
    [[-1 / 4 * math.pi, 1 / 4 * math.pi], [-1 / 4 * math.pi, -1 / 6 * math.pi]],  # 区域3
    [[-1 / 4 * math.pi, 1 / 4 * math.pi], [1 / 6 * math.pi, 1 / 4 * math.pi]],  # 区域4
]

# 设置绘图
fig, ax = plt.subplots(figsize=(8, 8))

# 填充不安全区域为红色
for unsafe in unsafe_areas:
    x1, x2 = unsafe[0][0], unsafe[0][1]  # x 的范围
    y1, y2 = unsafe[1][0], unsafe[1][1]  # y 的范围

    # 填充每个不安全区域
    ax.fill_betweenx([y1, y2], x1, x2, color='red', alpha=0.5)

# 绘制初始区域的边框
init_x = [INIT[0][0], INIT[0][1], INIT[0][1], INIT[0][0], INIT[0][0]]
init_y = [INIT[1][0], INIT[1][0], INIT[1][1], INIT[1][1], INIT[1][0]]
ax.plot(init_x, init_y, color='black', linewidth=2, label='INIT')  # 黑色边框

# 假设 track 是一个包含 (x, y) 坐标的列表
# track = dataset2.get_source_dataset_test(2000, "s_1", 2.5 ,2.5)  # 获取轨迹数据
track = dataset2.get_source_dataset_test_tf(2000)
# 绘制轨迹数据的点
if track:  # 确保 track 列表不为空
    track_array = np.array(track)  # 转换为 numpy 数组
    ax.plot(track_array[:, 0], track_array[:, 1], color='blue', marker='o', label='Track')  # 绘制数据点


# 设置坐标轴范围
ax.set_xlim(-1 / 4 * math.pi, 1 / 4 * math.pi)  # x轴范围
ax.set_ylim(-1 / 4 * math.pi, 1 / 4 * math.pi)  # y轴范围

# 设置主要刻度
major_ticks = [-1 / 4 * math.pi, -1 / 6 * math.pi, -1 / 15 * math.pi, 0, 1 / 15 * math.pi, 1 / 6 * math.pi,
               1 / 4 * math.pi]
ax.set_xticks(major_ticks)  # x轴刻度
ax.set_yticks(major_ticks)  # y轴刻度

# 更新刻度标签
ax.set_xticklabels([f"{round(t / math.pi, 2)}\u03C0" for t in major_ticks])
ax.set_yticklabels([f"{round(t / math.pi, 2)}\u03C0" for t in major_ticks])

# 设置坐标轴标签和标题
ax.set_xlabel(r'$X$ (radians)')
ax.set_ylabel(r'$Y$ (radians)')
ax.set_title('Unsafe Regions and Initial Region Visualization with Track Points')
ax.axhline(0, color='black', linewidth=0.5, ls='--')
ax.axvline(0, color='black', linewidth=0.5, ls='--')
ax.grid(color='gray', linestyle='--', linewidth=0.5)

# 显示图例
ax.legend()
# 显示图形
plt.show()
