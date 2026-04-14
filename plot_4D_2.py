import matplotlib.pyplot as plt
import math
import numpy as np
from code_combine.source_domain import source_load
from case_4D_2.source.data import dataset2
import torch
import torch.nn as nn

# 定义初始域
INIT = [[-0.3, 0.3], [-0.3, 0.3]]

# 定义不安全区域
# unsafe_areas = [
#     [[1.5, 2.0], [-1.0, 2.0]],
#     [[-1.5, 1.5], [1.5, 2.0]]
# ]
unsafe_areas = [
    [[1.5, 2.0], [-1.0, 2.0]],
    [[-1.0, 1.5], [1.5, 2.0]],
    [[-1.0, 2.0], [-1.5, -1.0]]
]


# 设置绘图
fig, ax = plt.subplots(figsize=(5, 4))

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
# track = dataset2.get_source_dataset_test(2000, 1)  # 获取轨迹数据
# 获取 30 条轨迹数据
tracks = [dataset2.get_source_dataset_test_tf_GAN(500) for _ in range(10)]
# tracks = [dataset2.get_source_dataset_test(500, 1) for _ in range(100)]
# tracks = [[track[0], track[2]] for track in tracks]

# 定义颜色列表
colors = [
    'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'red', 'pink', 'lime',
    'teal', 'gold', 'navy', 'violet', 'coral', 'olive', 'gray', 'maroon', 'chocolate', 'black',
    'khaki', 'turquoise', 'indigo', 'lavender', 'crimson', 'darkgreen', 'darkblue', 'salmon', 'peru', 'slateblue'
]
# 绘制轨迹数据的点

# track_array = np.array(track)  # 转换为 numpy 数组

# 绘制每一条轨迹的点
for i, track in enumerate(tracks):
    if track:  # 确保 track 列表不为空
        track_array = np.array(track)  # 转换为 numpy 数组
        ax.plot(
            track_array[:, 0],
            track_array[:, 1],
            color=colors[i % len(colors)],
            marker='o',  # 显示轨迹点
            markersize=1,  # 点的大小
            label=f'Track {i + 1}'
        )


# 设置坐标轴范围
ax.set_xlim(-3, 3)  # x轴范围
ax.set_ylim(-3, 3)  # y轴范围





# # 设置主要刻度
# major_ticks = [-3, -2, -0.3, 0, 0.3, 2, 3]
# ax.set_xticks(major_ticks)  # x轴刻度
# ax.set_yticks(major_ticks)  # y轴刻度
#
# # 更新刻度标签
# ax.set_xticklabels([f"{round(t)}" for t in major_ticks])
# ax.set_yticklabels([f"{round(t)}" for t in major_ticks])

# 设置主要刻度为均匀分布
x_ticks = np.linspace(-3, 3, 7)  # 生成 x 轴 7 个均匀分布的刻度
y_ticks = np.linspace(-3, 3, 7)  # 生成 y 轴 7 个均匀分布的刻度

# 应用均匀刻度
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

# 更新刻度标签
ax.set_xticklabels([f"{round(t, 1)}" for t in x_ticks])  # 格式化为小数点一位
ax.set_yticklabels([f"{round(t, 1)}" for t in y_ticks])


# 设置坐标轴标签和标题
# ax.set_xlabel(r'$X$ (radians)')
# ax.set_ylabel(r'$Y$ (radians)')
# ax.set_title('Unsafe Regions and Initial Region Visualization with Track Points')
ax.axhline(0, color='black', linewidth=0.5, ls='--')
ax.axvline(0, color='black', linewidth=0.5, ls='--')
ax.grid(color='gray', linestyle='--', linewidth=0.5)

# 显示图例
# ax.legend()
# 显示图形
plt.savefig('Fig4_db.jpg', dpi=600, bbox_inches='tight', format='jpg')
plt.show()
