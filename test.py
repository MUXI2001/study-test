import random
import os
import sys
import torch
import tensorflow as tf
import torch.nn as nn
import numpy as np
from case_2D import prob_2D
import numpy.linalg as LA
from NN.ann import CustomNetOne

B_net = torch.load('D:\PyCharm_Project\\transfer-barr\\trans\\barr_nn')
C_net = torch.load('D:\PyCharm_Project\\transfer-barr\\trans\ctrl_nn')
# # 初始域 b(x) < -eta
in_eps = 9e-4
in_xs1 = torch.arange(-np.pi/15, np.pi/15, in_eps, dtype=torch.double)
in_xs2 = torch.arange(-np.pi/15, np.pi/15, in_eps, dtype=torch.double)
points = torch.cartesian_prod(in_xs1, in_xs2)
print(len(points))

# 不安全域 （条件2）b(x) > eta
u_eps = 7e-3  # 步长
u_xs1 = torch.arange(-np.pi/4, np.pi/4, u_eps, dtype=torch.double)
u_xs2 = torch.arange(-np.pi/4, np.pi/4, u_eps, dtype=torch.double)
# 生成笛卡尔积
points_all = torch.cartesian_prod(u_xs1, u_xs2)
# 定义不安全区域 [-π/6, π/6]
exclude_lower = -np.pi / 6
exclude_upper = np.pi / 6
# 筛选不在不安全区域的点
mask = ~((exclude_lower <= points_all[:, 0]) & (points_all[:, 0] <= exclude_upper) &
         (exclude_lower <= points_all[:, 1]) & (points_all[:, 1] <= exclude_upper))
points_unsafe = points_all[mask]
print(len(points_unsafe))

# 第三个条件
points_all_next = prob_2D.vector_field(torch.tensor(points_all), C_net(torch.tensor(points_all))[:, 0], 0).detach().numpy()


B_ini = B_net(points).detach().numpy()  # 得到屏障证书值
init_B_value = np.max(B_ini)  # 取最大值

B_u = B_net(points_unsafe).detach().numpy()  # 得到屏障证书值
u_B_value = np.min(B_u)  # 取最小值

B_third = B_net(torch.tensor(points_all_next)).detach().numpy() - B_net(points_all).detach().numpy()
third_B_value = np.max(B_third)

print('初始域max：', init_B_value)
print('不安全域min：', u_B_value)
print('第三个条件：', third_B_value)

# model = tf.keras.models.load_model('D:\PyCharm_Project\\transfer-barr\case_2D_2\\result\\tf_result\\target_encoder.keras', safe_mode=False)
# lip = 1
# for i in range(len(B_net.weights)):
#     lip*=np.max(np.abs(B_net.weights[i]))
# print(lip)


# def calculate_lip2(m):
#     # 提取神经网络所有的权重
#     weights = []
#     for name, param in m.named_parameters():
#         if 'weight' in name:  # 只提取权重，不包含偏置
#             weights.append(param.detach())  # detach 避免影响计算图
#
#     nm = len(weights)  # 权重层数
#     lip = 0  # 初始化 Lipschitz 值
#
#     # 第一个权重矩阵
#     test = weights[0]
#
#     for i in range(1, nm):
#         # 矩阵连乘
#         # test = torch.matmul(test, weights[i].T)
#         test = torch.matmul(test.T, weights[i].T)
#
#         if i != nm - 1:
#             # 计算剩余权重矩阵的特征值最大值
#             test2 = weights[i]
#             for j in range(i + 1, nm):
#                 # test2 = torch.matmul(test2.T, weights[j].T)
#                 test2 = torch.matmul(test2, weights[j])
#
#             eigenvalues2 = torch.linalg.eigvalsh(torch.matmul(test2.T, test2))
#             max_eigenvalue2 = torch.max(eigenvalues2).sqrt()
#         else:
#             max_eigenvalue2 = 1  # 最后一层没有后续的权重
#
#         # 计算当前矩阵的特征值最大值
#         eigenvalues1 = torch.linalg.eigvalsh(torch.matmul(test.T, test))
#         max_eigenvalue1 = torch.max(eigenvalues1).sqrt()
#
#         # 累加 Lipschitz 项
#         lip += max_eigenvalue1 * max_eigenvalue2
#
#     # 归一化
#     return lip / (2 ** (nm - 1))
# # print(list(B_net.named_parameters()))
# def calculate_lip_torch(m):
#     """
#     正确估算神经网络的 Lipschitz 常数。
#     :param m: 神经网络模型
#     :return: Lipschitz 常数
#     """
#     lip = 1.0  # 初始化 Lipschitz 常数
#     for name, param in m.named_parameters():
#         if 'weight' in name:  # 只处理权重矩阵
#             weight = param.detach()  # 获取权重矩阵
#             # 计算权重矩阵的最大奇异值
#             max_singular_value = torch.linalg.svdvals(weight).max()
#             lip *= max_singular_value  # 逐层相乘
#
#     return lip
#
# print(calculate_lip_torch(B_net))
# print(calculate_lip_torch(C_net))
# 计算李普希兹常数
# def compute_lipschitz_constant(model):
#     lipschitz_constant = 1
#     for layer in model.modules():
#         if isinstance(layer, nn.Linear):
#             # 获取权重矩阵
#             weights = layer.weight.data
#             # 计算谱范数（最大奇异值）
#             singular_values = torch.linalg.svd(weights, full_matrices=False).S
#             spectral_norm = singular_values.max().item()
#             lipschitz_constant *= spectral_norm
#         elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Hardtanh):
#             # ReLU 和 Hardtanh 的李普希兹常数均为 1
#             lipschitz_constant *= 1
#     return lipschitz_constant

# 计算并输出
# lipschitz_constant = compute_lipschitz_constant(B_net)
# lip = calculate_lip_torch(B_net)
# print(f"Total Lipschitz constant: {lipschitz_constant}")

#
# print(f"lip: {lip}")
# print(calculate_lip2(B_net))


# # 给定参数
# g = 9.8  # 重力加速度
# l = 1.0  # 摆杆长度
# m = 1.0  # 质量
# tau = 0.01  # 时间步长
#
# # 假设 k'(x1) = 0，代入雅可比矩阵
# # 计算雅可比矩阵 J_x
# x1 = 0.1  # 假设 x1 的值
# J_x = np.array([
#     [1, tau],
#     [g * tau / l * np.cos(x1), 1]
# ])
#
# # 使用 SVD 计算奇异值
# U, S, Vt = np.linalg.svd(J_x)
#
# # 最大奇异值 L_x
# L_x = np.max(S)
#
# # 计算 L_u (根据前面的矩阵推导)
# J_u = np.array([
#     [tau / l],
#     [0]
# ])
#
# L_u = np.linalg.norm(J_u, ord=2)  # 欧几里得范数
#
# print(L_x, L_u)
