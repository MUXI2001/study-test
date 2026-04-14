import time
import random
import os
import sys
import torch

# 将 'code_combine' 目录添加到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'code_combine')))
import click
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from case_2D_2 import prob_2D_2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'


@click.command()
@click.argument('num_epoch', type=int)
@click.option('--disc_lr', default=1e-4, help='Discriminator learning rate')
@click.option('--gene_lr', default=1e-4, help='Encoder learning rate')
def main(num_epoch, disc_lr, gene_lr):
    gene_lr = float(gene_lr)
    num_epoch = int(num_epoch)
    disc_lr = float(disc_lr)

    encoder_save_path = "D:\PyCharm_Project\\transfer-barr\case_2D_2\\result\\tf_result\\target_encoder.keras"
    adversary_save_path = "D:\PyCharm_Project\\transfer-barr\case_2D_2\\result\\tf_result\\adversary.keras"

    seed = 19
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.random.set_seed(seed + 2)

    barr = torch.load('D:\PyCharm_Project\\transfer-barr\case_2D_2\source\domain\\2\\barr.pth')
    ctrl_nn = torch.load('D:\PyCharm_Project\\transfer-barr\case_2D_2\source\domain\\2\\ctrl.pth')
    ctrl_nn.eval()
    barr.eval()

    eps = 5.7e-4
    # Constructing the state space
    xs1 = torch.range(-0.7, 0.7, eps, dtype=torch.double)
    xs2 = torch.range(-0.1, 0.1, eps, dtype=torch.double)
    zs = torch.cartesian_prod(xs1, xs2)
    xu = zs.detach().numpy()
    # Vectorfield of the source system
    xpt = prob_2D_2.vector_field(torch.tensor(zs), ctrl_nn(torch.tensor(zs))[:, 0], 0).detach().numpy()
    xxpt = np.hstack((xu, xpt))
    bv = barr(torch.tensor(xxpt[:, 0:2]))
    ind1 = np.where(bv < 0)
    xxpt = xxpt[ind1[0], :]

    gene_model = tf.keras.Sequential()
    gene_model.add(layers.Dense(100, use_bias=True, activation='relu', input_shape=(2,),kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    gene_model.add(layers.Dense(100, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    # gene_model.add(layers.Dense(100, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    gene_model.add(layers.Dense(1, use_bias=True, activation='linear', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    gene_model.add(tf.keras.layers.Lambda(lambda x: x * 10))

    disc_model = tf.keras.Sequential()
    disc_model.add(layers.Dense(10, use_bias=True, activation='relu', input_shape=(2,),kernel_regularizer=tf.keras.regularizers.L1(0.00001)))
    disc_model.add(layers.Dense(10, use_bias=True, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.00001)))
    disc_model.add(layers.Dense(1, use_bias=True, activation='linear', kernel_regularizer=tf.keras.regularizers.L1(0.00001)))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=gene_lr,
        decay_steps=1000,
        decay_rate=0.88,
        staircase=True)
    lr_schedule2 = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=4e-5,
        decay_steps=600,
        decay_rate=0.9,
        staircase=True)
    # 定义优化器
    gene_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    mlp_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule2)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr)

    # 梯度惩罚
    def gradient_penalty(critic, real_data, fake_data, lambda_gp=10):
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
    def generator_loss(fake_output, pre, lab, alpha):
        wasserstein_loss = -tf.reduce_mean(fake_output)
        mse = tf.reduce_mean(tf.square(pre - lab))
        mae = tf.reduce_max(tf.abs(lab - pre))
        total_loss = wasserstein_loss + alpha * mse + alpha * mae
        return total_loss, mse, mae



    K = 50
    fd = 0.0
    er = 1
    er2 = 1
    Ver = True
    flag = False
    disc_losses = []
    k = 0.01
    l = 0.55
    r = 1.2
    wasserstein_distance = 0
    t1 = time.perf_counter()
    # 训练循环
    for i in range(num_epoch):
        if not flag:
            lip = 1
            for j in range(gene_model.weights[0].shape[0]):
                lip *= np.max(np.abs(gene_model.weights[0][j]))

            if i % 50 == 0 and Ver:
                print(f'mse:{float(er)}, mae:{float(er2)}, epoch:{i}, lip:{lip}, fd:{fd}, lr:{lr_schedule(i).numpy()}')

            # 准备真实轨迹和初始状态
            label = xxpt[:, 2:4]

            # Step 1: 训练判别器
            for _ in range(3):
                with tf.GradientTape() as tape:
                    u = gene_model(xxpt[:, 0:2], training=False)

                    fake_trajectory = xxpt[:, 0].reshape(xxpt.shape[0], 1) + 0.01 * ((-r/l) * xxpt[:, 0].reshape(xxpt.shape[0], 1) - (k/l)
                                                                                     * xxpt[:, 1].reshape(xxpt.shape[0], 1) + 1/l * (u[:]))
                    fake_trajectory = np.hstack((fake_trajectory, xxpt[:, 3].reshape(-1, 1)))
                    # print(label[:5, :])
                    # print("--------------------")
                    # print(fake_trajectory[:5, :])
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
                    fd = (sum((loss - mu_disc_loss) ** 2 for loss in disc_losses) / K) ** 0.5  # 计算 fd

                # 更新判别器
                gradients = tape.gradient(disc_loss, disc_model.trainable_variables)
                disc_optimizer.apply_gradients(zip(gradients, disc_model.trainable_variables))

            with tf.GradientTape() as tape:
                u = gene_model(xxpt[:, 0:2], training=True)
                # 再次生成伪造轨迹并计算生成器损失
                fake_trajectory = xxpt[:, 0].reshape(xxpt.shape[0], 1) + 0.01 * ((-r / l) * xxpt[:, 0].reshape(xxpt.shape[0], 1) - (k / l)
                                                                                 * xxpt[:, 1].reshape(xxpt.shape[0], 1) + 1 / l * (u[:]))
                fake_trajectory = tf.concat([fake_trajectory, xxpt[:, 3].reshape(-1, 1)], axis=1)
                fake_preds = disc_model(fake_trajectory, training=False)

                # 计算生成器损失
                gene_loss, er, er2 = generator_loss(fake_preds, fake_trajectory[:, 0], label[:, 0], 0.4)

            # 更新生成器
            gradients = tape.gradient(gene_loss, gene_model.trainable_variables)
            gene_optimizer.apply_gradients(zip(gradients, gene_model.trainable_variables))

            if lip < 0.06 and er2 < 0.004:
                print('对抗训练结束')
                print(f'mse:{float(er)}, mae:{float(er2)}, epoch:{i}, lip:{lip}, fd:{fd}')
                gene_model.save(encoder_save_path)
                disc_model.save(adversary_save_path)
                flag = True
        if flag:
            lip = 1
            for j in range(gene_model.weights[0].shape[0]):
                lip *= np.max(np.abs(gene_model.weights[0][j]))
            # for layer in model.layers:
            #     if hasattr(layer, 'kernel'):  # 检查层是否有权重
            #         weights = layer.kernel.numpy()  # 提取权重矩阵
            #         max_norm = np.max(np.sum(np.abs(weights), axis=0))  # 按列计算最大范数
            #         lip *= max_norm

            if lip < 0.06 and er2 < 0.001:
                break
            if i % 100 == 0 and Ver:
                print(f'mse:{er},mae:{er2},epoch:{i},lip:{lip},lr:{lr_schedule2(i).numpy()}')
            with tf.GradientTape() as id_tape:

                u = gene_model(xxpt[:, 0:2], training=True)
                y_true = xxpt[:, 2].reshape(xxpt.shape[0], 1)
                # Vector field of the target system:
                x2 = xxpt[:, 0].reshape(xxpt.shape[0], 1) + 0.01 * (
                            (-r / l) * xxpt[:, 0].reshape(xxpt.shape[0], 1) - (k / l)
                            * xxpt[:, 1].reshape(xxpt.shape[0], 1) + 1 / l * (u[:]))

                loss = tf.reduce_mean(tf.square(tf.subtract(y_true, x2))) + tf.math.reduce_max(
                    (tf.abs(tf.subtract(y_true, x2))))

                er = np.copy(tf.reduce_mean(tf.square(tf.subtract(y_true, x2))))
                er2 = np.copy(tf.math.reduce_max((tf.abs(tf.subtract(y_true, x2)))))
            grads = id_tape.gradient(loss, gene_model.trainable_variables)
            mlp_optimizer.apply_gradients(zip(grads, gene_model.trainable_variables))

    t2 = time.perf_counter()
    total_time = t2 - t1
    print("Total time：", total_time)

    gene_model.save(encoder_save_path)
    disc_model.save(adversary_save_path)
    print('save')

if __name__ == '__main__':
    main()
