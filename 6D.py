import tensorflow as tf
from tensorflow.keras import layers
import torch
import torch.nn as nn
import numpy as np
import time
import numpy.linalg as LA
import random
from case_6D import prob_6D
tf.config.list_physical_devices('GPU')

seed = 19
random.seed(seed)
np.random.seed(seed + 1)
tf.random.set_seed(seed + 2)

barr = torch.load('D:\PyCharm_Project\\transfer-barr\case_6D\source\domain\\barr.pth')
ctrl_nn = torch.load('D:\PyCharm_Project\\transfer-barr\case_6D\source\domain\ctrl.pth')
encoder_save_path = "D:\PyCharm_Project\\transfer-barr\case_6D\\result\\ctrl_6D.keras"
barr.eval()
ctrl_nn.eval()
dist = 1
eps = 0.14
xs = torch.arange(0, 1, eps, dtype=torch.double)
zs = torch.cartesian_prod(xs, xs, xs, xs, xs, xs)
xu = zs.detach().numpy()
xpt = prob_6D.vector_field(torch.tensor(zs),ctrl_nn(torch.tensor(zs)), 0).detach().numpy()
xxpt = np.hstack((xu,xpt))
bv = barr(torch.tensor(xxpt[:, 0:6]))
ind1 = np.where(bv < 0)
xxpt = xxpt[ind1[0], :]

model = tf.keras.Sequential()
model.add(layers.Dense(200, use_bias=True, activation='relu', input_shape=(6,), kernel_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(layers.Dense(200, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(layers.Dense(200, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(layers.Dense(200, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(layers.Dense(1, use_bias=True, activation='linear', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(tf.keras.layers.Lambda(lambda x: x * 10))
# model.load_weights('D:\PyCharm_Project\\transfer-barr\model.weights.h5')
del(xs, zs)

def calculate_lip_tf(m):
    test = m.weights[0].numpy()
    nm = int(len(m.weights) / 2)
    lip = 0

    for i in range(2, nm * 2, 2):
        test = np.matmul(test, m.weights[i].numpy())

        if i != nm * 2 - 2:
            test2 = m.weights[i].numpy()
            for j in range(i + 2, len(m.weights), 2):
                test2 = np.matmul(test2, m.weights[j].numpy())

            eigenvalues2, _ = LA.eig(np.matmul(test2.T, test2))
        else:
            eigenvalues2 = 1

        eigenvalues1, _ = LA.eig(np.matmul(test.T, test))

        lip += np.sqrt(np.max(eigenvalues1)) * np.sqrt(np.max(eigenvalues2))

    return lip / (np.power(2, nm - 1))

def calculate_lip_torch(m):
    # 提取神经网络所有的权重
    weights = []
    for name, param in m.named_parameters():
        if 'weight' in name:  # 只提取权重，不包含偏置
            weights.append(param.detach())  # detach 避免影响计算图

    nm = len(weights)  # 权重层数
    lip = 0  # 初始化 Lipschitz 值

    # 第一个权重矩阵
    test = weights[0]
    test_ = weights[0].T

    for i in range(1, nm):
        # 矩阵连乘
        # test = torch.matmul(test, weights[i].T)
        print(test.shape)
        print(weights[i].shape)
        if i == 1:
            test = torch.matmul(test.T, weights[i].T)
        else:
            test = torch.matmul(test_, weights[i].T)

        if i != nm - 1:
            # 计算剩余权重矩阵的特征值最大值
            test2 = weights[i]
            for j in range(i + 1, nm):
                # test2 = torch.matmul(test2.T, weights[j].T)
                test2 = torch.matmul(test2, weights[j].T)

            eigenvalues2 = torch.linalg.eigvalsh(torch.matmul(test2.T, test2))
            max_eigenvalue2 = torch.max(eigenvalues2).sqrt()
        else:
            max_eigenvalue2 = 1  # 最后一层没有后续的权重

        # 计算当前矩阵的特征值最大值
        eigenvalues1 = torch.linalg.eigvalsh(torch.matmul(test.T, test))
        max_eigenvalue1 = torch.max(eigenvalues1).sqrt()

        # 累加 Lipschitz 项
        lip += max_eigenvalue1 * max_eigenvalue2

    # 归一化
    return lip / (2 ** (nm - 1))

def calculate_condition(e, mae, LB, LX, LU, LK, LX_T, LU_T, LK_T):
    L_ = LX + LU * LK + LX_T + LU_T * LK_T
    lip = LB * (L_ * e / 2 + mae)
    return lip

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=1500,
    decay_rate=0.9,
    staircase=True)
opt = tf.keras.optimizers.Adam(lr_schedule)

# Training, timing the cell for Table 1

epochs = 16000
er = 1
er2 = 1
# Weight and length of the target system:
a1 = -1.5
a2 = 4.5

tau = 0.1

Lb = calculate_lip_torch(barr)
Lk = calculate_lip_torch(ctrl_nn)
print(Lb, Lk)
Lx = 1.81
Lu = 0.1
# target
Lx_t = 1.92
Lu_t = 0.1

# Verbose:
Ver = True
t1 = time.perf_counter()
for i in range(epochs):

    Lk_t = calculate_lip_tf(model)
    lip = float(calculate_condition(0.099, er2, Lb, Lx, Lu, Lk, Lx_t, Lu_t, Lk_t))

    if lip < 1.95:
        print(f'mse:{er},mae:{er2},epoch:{i},lip:{lip}')
        break
    if i % 100 == 0 and Ver:
        print(f'mse:{er},mae:{er2},epoch:{i},lip:{lip}')
    with tf.GradientTape() as id_tape:

        u = model(xxpt[:, 0:6], training=True)
        y_true = xxpt[:, 6].reshape(xxpt.shape[0], 1)
        # Vector field of the target system:
        # x[:, 0] + time_step * (a1 * x[:, 0]**3 + a2 * x[:, 1]**3 + u[:, 0])
        x2 = xxpt[:, 0].reshape(xxpt.shape[0], 1) + tau * (a1 * (xxpt[:, 0].reshape(xxpt.shape[0], 1)) ** 3 + a2 * (
            xxpt[:, 1].reshape(xxpt.shape[0], 1)) ** 3 + (u[:]))
        # print(y_true[:5, :])
        # print('----------------------')
        # print(x2[:5, :])
        loss = tf.reduce_mean(tf.square(tf.subtract(y_true, x2))) + tf.math.reduce_max(
            (tf.abs(tf.subtract(y_true, x2))))

        er = np.copy(tf.reduce_mean(tf.square(tf.subtract(y_true, x2))))
        er2 = np.copy(tf.math.reduce_max((tf.abs(tf.subtract(y_true, x2)))))
    grads = id_tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
t2 = time.perf_counter()
tot = t2 - t1

model.save(encoder_save_path)
print('save')
print(f'Total runtime:{tot}s')