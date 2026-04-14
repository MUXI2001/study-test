import time
import random
import torch
import numpy.linalg as LA
import click
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras import layers
from case_6D import prob_6D


@click.command()
@click.argument('num_epoch', type=int)
@click.option('--disc_lr', default=1e-4, help='Discriminator learning rate')
@click.option('--gene_lr', default=1e-4, help='Encoder learning rate')
def main(num_epoch, disc_lr, gene_lr):
    gene_lr = float(gene_lr)
    num_epoch = int(num_epoch)
    disc_lr = float(disc_lr)

    encoder_save_path = "D:\PyCharm_Project\\transfer-barr\case_6D\\result\\target_gene_6D.keras"
    adversary_save_path = "D:\PyCharm_Project\\transfer-barr\case_6D\\result\\target_disc_6D.keras"

    seed = 19
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.random.set_seed(seed + 2)

    barr = torch.load('D:\PyCharm_Project\\transfer-barr\case_6D\source\domain\\barr.pth')
    ctrl_nn = torch.load('D:\PyCharm_Project\\transfer-barr\case_6D\source\domain\ctrl.pth')
    ctrl_nn.eval()
    barr.eval()

    eps = 0.15
    xs = torch.arange(0, 1, eps, dtype=torch.double)
    zs = torch.cartesian_prod(xs, xs, xs, xs, xs, xs)
    xu = zs.detach().numpy()
    xpt = prob_6D.vector_field(torch.tensor(zs), ctrl_nn(torch.tensor(zs)), 0).detach().numpy()
    xxpt = np.hstack((xu, xpt))
    bv = barr(torch.tensor(xxpt[:, 0:6]))
    ind1 = np.where(bv < 0)
    xxpt = xxpt[ind1[0], :]
    print(xxpt.shape)


    gene_model = tf.keras.Sequential()
    gene_model.add(layers.Dense(200, use_bias=True, activation='relu', input_shape=(6,), kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    gene_model.add(layers.Dense(200, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    gene_model.add(layers.Dense(200, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    gene_model.add(layers.Dense(200, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    gene_model.add(layers.Dense(1, use_bias=True, activation='linear', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    gene_model.add(tf.keras.layers.Lambda(lambda x: x * 10))
    gene_model.save_weights('D:\PyCharm_Project\\transfer-barr\\model.weights.h5')

    disc_model = tf.keras.Sequential()
    disc_model.add(layers.Dense(12, use_bias=True, activation='relu', input_shape=(6,), kernel_regularizer=tf.keras.regularizers.L2(0.00001)))
    disc_model.add(layers.Dense(12, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.00001)))
    disc_model.add(layers.Dense(1, use_bias=True, activation='linear', kernel_regularizer=tf.keras.regularizers.L2(0.00001)))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=gene_lr,
        decay_steps=1500,
        decay_rate=0.90,
        staircase=True)

    # 定义优化器
    gene_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr)

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
            # print(test.shape)
            # print(weights[i].shape)
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

    # 梯度惩罚
    def gradient_penalty(critic, real_data, fake_data, lambda_gp=10.0):
        batch_size = tf.shape(real_data)[0]

        # 在真实数据和生成数据之间插值
        alpha = tf.random.uniform([real_data.shape[0], 1], 0, 1)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = tf.Variable(interpolates)  # 使其可求梯度

        # 计算判别器对插值数据的输出
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            d_interpolates = critic(interpolates)

        # 计算插值数据的梯度
        gradients = tape.gradient(d_interpolates, interpolates)

        # 计算梯度的 L2 范数
        gradients = tf.reshape(gradients, [batch_size, -1])
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)

        # 梯度惩罚项
        gp = lambda_gp * tf.reduce_mean((gradient_norm - 1.0) ** 2)
        gp = tf.cast(gp, tf.float64)
        return gp

    # 判别器的损失函数（WGAN-GP）
    def discriminator_loss(real_output, fake_output, real_data, fake_data, critic, lambda_gp=10):
        real_loss = -tf.reduce_mean(real_output)  # 真实样本的损失（最大化）
        fake_loss = tf.reduce_mean(fake_output)  # 伪造样本的损失（最小化）

        # 计算梯度惩罚项
        gp = gradient_penalty(critic, real_data, fake_data, lambda_gp)

        wasserstein_distance = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)

        # 总判别器损失
        total_loss = real_loss + fake_loss + gp
        return total_loss, wasserstein_distance

    # 生成器的损失函数
    def generator_loss(fake_output, pre, lab, beata, alpha):
        wasserstein_loss = -tf.reduce_mean(fake_output)
        mse = tf.reduce_mean(tf.square(pre - lab))
        mae = tf.reduce_max(tf.abs(lab - pre))
        total_loss = beata * wasserstein_loss + alpha * mse + alpha * mae
        return total_loss, mse, mae

    K = 50
    fd = 0.0
    er = 1
    er2 = 1
    Ver = True
    disc_losses = []
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

    wasserstein_distance = 0
    t1 = time.perf_counter()
    # 训练循环
    for i in range(num_epoch):
        Lk_t = calculate_lip_tf(gene_model)
        lip = float(calculate_condition(0.099, er2, Lb, Lx, Lu, Lk, Lx_t, Lu_t, Lk_t))
        beta = 1.0 * (1 - (i + 1500) / num_epoch)

        if lip < 1.915:
            print('结束')
            print(f'mse:{float(er)}, mae:{float(er2)}, epoch:{i}, lip:{lip}, fd:{fd}, lr:{lr_schedule(i).numpy()}')
            break

        if i % 100 == 0 and Ver:
            print(f'mse:{float(er)}, mae:{float(er2)}, epoch:{i}, lip:{lip}, fd:{fd}, lr:{lr_schedule(i).numpy()}')

        # 准备真实轨迹和初始状态
        label = xxpt[:, 6:12]

        # Step 1: 训练判别器
        for _ in range(3):
            with tf.GradientTape() as tape:
                u = gene_model(xxpt[:, 0:6], training=False)

                fake_trajectory = xxpt[:, 0].reshape(xxpt.shape[0], 1) + tau * (a1 * (xxpt[:, 0].reshape(xxpt.shape[0], 1)) ** 3 + a2 * (
                                  xxpt[:, 1].reshape(xxpt.shape[0], 1)) ** 3 + (u[:]))
                fake_trajectory = np.hstack((fake_trajectory, xxpt[:, 7:12]))
                # 判别真实数据和伪造数据
                real_preds = disc_model(label, training=True)
                fake_preds = disc_model(fake_trajectory, training=True)

                # 计算判别器损失
                disc_loss, wasserstein_distance = discriminator_loss(real_preds, fake_preds, label, fake_trajectory,
                                                                     disc_model)

                disc_losses.append(disc_loss.numpy())

                # 如果超过 K，则只保留最近的 K 个损失
                if len(disc_losses) > K:
                    disc_losses.pop(0)
                # 输出 Wasserstein 距离
            if len(disc_losses) == K:
                mu_disc_loss = sum(disc_losses) / K  # 计算损失的均值
                fd = (sum((loss - mu_disc_loss) ** 2 for loss in disc_losses) / K) ** 0.4  # 计算 fd

            # 更新判别器
            gradients = tape.gradient(disc_loss, disc_model.trainable_variables)
            disc_optimizer.apply_gradients(zip(gradients, disc_model.trainable_variables))

        with tf.GradientTape() as tape:
            u = gene_model(xxpt[:, 0:6], training=True)
            # 再次生成伪造轨迹并计算生成器损失
            fake_trajectory = xxpt[:, 0].reshape(xxpt.shape[0], 1) + tau * (a1 * (xxpt[:, 0].reshape(xxpt.shape[0], 1)) ** 3 + a2 * (
                              xxpt[:, 1].reshape(xxpt.shape[0], 1)) ** 3 + (u[:]))
            fake_trajectory = tf.concat([fake_trajectory, xxpt[:, 7:12]], axis=1)
            fake_preds = disc_model(fake_trajectory, training=False)
            # 计算生成器损失

            gene_loss, er, er2 = generator_loss(fake_preds, fake_trajectory[:, 0], label[:, 0], beta, 0.8)

        # 更新生成器
        gradients = tape.gradient(gene_loss, gene_model.trainable_variables)
        gene_optimizer.apply_gradients(zip(gradients, gene_model.trainable_variables))

    t2 = time.perf_counter()
    total_time = t2 - t1
    print("Total time：", total_time)

    gene_model.save(encoder_save_path)
    disc_model.save(adversary_save_path)
    print('save')


if __name__ == '__main__':
    main()
