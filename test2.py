import tensorflow as tf
import numpy as np

# 加载 .keras 模型
model1 = tf.keras.models.load_model("D:\PyCharm_Project\\transfer-barr\case_4D_2\\result\\target_gene_4D_2_samll.keras")
model2 = tf.keras.models.load_model("D:\PyCharm_Project\\transfer-barr\case_4D_2\\result\ctrl_4D_2.keras")

# 设置阈值
threshold = 1e-3

# 统计参数
total_params = 0
above_threshold = 0

for layer in model1.layers:
    for weight in layer.get_weights():
        weight = weight.flatten()
        total_params += weight.size
        above_threshold += np.sum(np.abs(weight) > threshold)

ratio = above_threshold / total_params if total_params > 0 else 0

print(f"总参数数量: {total_params}")
print(f"超过阈值 {threshold} 的参数数量: {above_threshold}")
print(f"比例: {ratio:.4f}")


total_params = 0
above_threshold = 0
for layer in model2.layers:
    for weight in layer.get_weights():
        weight = weight.flatten()
        total_params += weight.size
        above_threshold += np.sum(np.abs(weight) > threshold)

ratio = above_threshold / total_params if total_params > 0 else 0

print(f"总参数数量: {total_params}")
print(f"超过阈值 {threshold} 的参数数量: {above_threshold}")
print(f"比例: {ratio:.4f}")

