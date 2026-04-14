import tensorflow as tf
from tensorflow.keras import layers
import torch
import torch.nn as nn
import numpy as np
import time
import numpy.linalg as LA
from case_4D_2 import prob_4D_2
import random

tf.config.list_physical_devices('GPU')
seed = 19
random.seed(seed)
np.random.seed(seed + 1)
tf.random.set_seed(seed + 2)

barr = torch.load('D:\PyCharm_Project\\transfer-barr\case_4D_2\source\domain\\1\\barr.pth')
ctrl_nn = torch.load('D:\PyCharm_Project\\transfer-barr\case_4D_2\source\domain\\1\ctrl.pth')

encoder_save_path = "D:\PyCharm_Project\\transfer-barr\case_4D_2\\result\\ctrl_4D_2.keras"

barr.eval()
ctrl_nn.eval()

dist = 1
eps = 0.29
xs = torch.arange(-3, 3, eps, dtype=torch.double)
zs = torch.cartesian_prod(xs, xs, xs, xs)
xu = zs.detach().numpy()
xpt = prob_4D_2.vector_field(torch.tensor(zs),ctrl_nn(torch.tensor(zs)),0).detach().numpy()
xxpt = np.hstack((xu,xpt))

model = tf.keras.Sequential()
model.add(layers.Dense(200, use_bias=True, activation='relu', input_shape=(4,),kernel_regularizer=tf.keras.regularizers.L1(0.01)))
model.add(layers.Dense(200, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.01)))
model.add(layers.Dense(200, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.01)))
model.add(layers.Dense(200, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.01)))
model.add(layers.Dense(2, use_bias=True, activation='linear', kernel_regularizer=tf.keras.regularizers.L1(0.01)))
model.add(tf.keras.layers.Lambda(lambda x: x * 10))
# model.load_weights('D:\PyCharm_Project\\transfer-barr\model.weights.h5')





del(xs, zs)
bv = barr(torch.tensor(xxpt[:,0:4]))
ind1 = np.where(bv<0)
xxpt = xxpt[ind1[0],:]
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

def calculate_lip_torch(model):
    lipschitz_constant = 1
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            # 获取权重矩阵
            weights = layer.weight.data
            # 计算谱范数（最大奇异值）
            singular_values = torch.linalg.svd(weights, full_matrices=False).S
            spectral_norm = singular_values.max().item()
            lipschitz_constant *= spectral_norm
        elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Hardtanh):
            # ReLU 和 Hardtanh 的李普希兹常数均为 1
            lipschitz_constant *= 1
    return lipschitz_constant

def calculate_condition(e, mae, LB, LX, LU, LK, LX_T, LU_T, LK_T):
    L_ = LX + LU * LK + LX_T + LU_T * LK_T
    lip = LB * (L_ * e / 2 + mae)
    return lip

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=2300,
    decay_rate=0.85,
    staircase=True)
opt = tf.keras.optimizers.Adam(lr_schedule)

# Training, timing the cell for Table 1

epochs = 10100
er = 1
er2 = 1
# Weight and length of the target system:
a1 = -1.0

tau = 0.01

Lb = calculate_lip_torch(barr)
Lk = calculate_lip_torch(ctrl_nn)
print(Lb, Lk)
Lx = 1.005
Lu = 0.01
# target -1.0
Lx_t = 1.005
Lu_t = 0.01

# Verbose:
Ver = True
t1 = time.perf_counter()
for i in range(epochs):

    Lk_t = calculate_lip_tf(model)
    lip = float(calculate_condition(0.23, er2, Lb, Lx, Lu, Lk, Lx_t, Lu_t, Lk_t))

    if lip < 0.251:
        print(f'mse:{er},mae:{er2},epoch:{i},lip:{lip}')
        break
    if i % 100 == 0 and Ver:
        print(f'mse:{er},mae:{er2},epoch:{i},lip:{lip}')
    with tf.GradientTape() as id_tape:

        u = model(xxpt[:, 0:4], training=True)
        y_true = xxpt[:, 4:8]
        # Vector field of the target system:
        # x2 = xxpt[:,2].reshape(xxpt.shape[0],1) + tau * (a1 * xxpt[:,0].reshape(xxpt.shape[0],1) * xxpt[:,0].reshape(xxpt.shape[0],1) * xxpt[:,2].reshape(xxpt.shape[0],1) + (u[:]))
        fake_trajectory_1 = xxpt[:, 0] + tau * (xxpt[:, 1] + a1 * 0.005 * u[:, 0])
        fake_trajectory_2 = xxpt[:, 1] + tau * a1 * u[:, 0]
        fake_trajectory_3 = xxpt[:, 2] + tau * (xxpt[:, 3] + a1 * 0.005 * u[:, 1])
        fake_trajectory_4 = xxpt[:, 3] + tau * a1 * u[:, 1]
        fake_trajectory_1 = tf.expand_dims(fake_trajectory_1, axis=1)
        fake_trajectory_2 = tf.expand_dims(fake_trajectory_2, axis=1)
        fake_trajectory_3 = tf.expand_dims(fake_trajectory_3, axis=1)
        fake_trajectory_4 = tf.expand_dims(fake_trajectory_4, axis=1)
        x2 = tf.concat([fake_trajectory_1, fake_trajectory_2, fake_trajectory_3, fake_trajectory_4], axis=1)
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
