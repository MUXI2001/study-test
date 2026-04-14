import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from case_3D.source.data import dataset2

# 定义初始域
INIT = [[-1.5, -0.5], [-1.5, -0.5], [-2.5, -1.5]]

# 定义不安全区域的函数
def unsafe_area(x, y, z):
    return (x + 2.5) ** 2 + (y + 2.5) ** 2 + (z + 0.5) ** 2 <= 0.25

# 设置三维绘图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制初始域（立方体）
x = [INIT[0][0], INIT[0][1], INIT[0][1], INIT[0][0], INIT[0][0], INIT[0][1], INIT[0][1], INIT[0][0]]
y = [INIT[1][0], INIT[1][0], INIT[1][1], INIT[1][1], INIT[1][0], INIT[1][0], INIT[1][1], INIT[1][1]]
z = [INIT[2][0], INIT[2][0], INIT[2][0], INIT[2][0], INIT[2][1], INIT[2][1], INIT[2][1], INIT[2][1]]

# 定义立方体的边
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
    (0, 4), (1, 5), (2, 6), (3, 7)   # 侧面
]

# 绘制立方体边
for edge in edges:
    ax.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], [z[edge[0]], z[edge[1]]], color='black')

# 绘制不安全区域（球形）
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
sphere_x = 0.5 * np.outer(np.cos(u), np.sin(v)) - 2.5
sphere_y = 0.5 * np.outer(np.sin(u), np.sin(v)) - 2.5
sphere_z = 0.5 * np.outer(np.ones(np.size(u)), np.cos(v)) - 0.5
ax.plot_surface(sphere_x, sphere_y, sphere_z, color='red', alpha=0.5)

# 获取轨迹数据
tracks = [dataset2.get_source_dataset_test_tf_GAN(500) for _ in range(25)]
colors = [
    'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'red', 'pink', 'lime',
    'teal', 'gold', 'navy', 'violet', 'coral', 'olive', 'gray', 'maroon', 'chocolate', 'black',
    'khaki', 'turquoise', 'indigo', 'lavender', 'crimson', 'darkgreen', 'darkblue', 'salmon', 'peru', 'slateblue'
]

# 绘制轨迹
for i, track in enumerate(tracks):
    if track:
        track_array = np.array(track)
        ax.plot(track_array[:, 0], track_array[:, 1], track_array[:, 2], color=colors[i % len(colors)], marker='o', markersize=4)

# 加载屏障证书模型
barr_model = torch.load(r'D:\PyCharm_Project\transfer-barr\case_3D\source\domain\barr.pth')

# 定义网格点
x_range = np.linspace(-3, 0, 30)
y_range = np.linspace(-3, 0, 30)
z_range = np.linspace(-3, 0, 30)
X, Y, Z = np.meshgrid(x_range, y_range, z_range)
B_values = np.zeros_like(X)

device = next(barr_model.parameters()).device
dtype = next(barr_model.parameters()).dtype

# 计算屏障证书 B(x, y, z)
for i, x_val in enumerate(x_range):
    for j, y_val in enumerate(y_range):
        for k, z_val in enumerate(z_range):
            state = torch.tensor([x_val, y_val, z_val], dtype=dtype, device=device)
            B_values[i, j, k] = barr_model(state).item()

# 只绘制 B(x, y, z) = 0 的等值面
verts, faces, _, _ = measure.marching_cubes(B_values, level=0)

# 将索引转换为实际坐标
verts[:, 0] = x_range[0] + verts[:, 0] * (x_range[-1] - x_range[0]) / (len(x_range) - 1)
verts[:, 1] = y_range[0] + verts[:, 1] * (y_range[-1] - y_range[0]) / (len(y_range) - 1)
verts[:, 2] = z_range[0] + verts[:, 2] * (z_range[-1] - z_range[0]) / (len(z_range) - 1)

ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap="viridis", alpha=0.5)

# 设置坐标轴范围
ax.set_xlim(-3, 0)
ax.set_ylim(-3, 0)
ax.set_zlim(-3, 0)

# 保存图像
plt.savefig('Fig3_db.jpg', dpi=600, bbox_inches='tight', format='jpg')
plt.show()