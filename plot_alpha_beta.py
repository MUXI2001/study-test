import pandas as pd
import matplotlib.pyplot as plt
import glob

# 设置绘图风格
plt.style.use('default')

# ==== 参数设置 ====
eta = 1.96  # 阈值
csv_path = "D:/PyCharm_Project/transfer-barr/case_6D/result/"  # CSV文件所在文件夹

# === 图1: αstart固定, βend变化 ===
alpha_fixed = 1.0
beta_values = [0.1, 0.5, 1.0, 2.0]
colors = ['r', 'g', 'b', 'm']

plt.figure(figsize=(8,5))
for beta, color in zip(beta_values, colors):
    file_pattern = f"{csv_path}/epoch_lip_{alpha_fixed}_{beta}.csv"
    files = glob.glob(file_pattern)
    if not files:
        print(f"文件未找到: {file_pattern}")
        continue
    df = pd.read_csv(files[0])
    plt.plot(df['epoch'], df['lip'], label=f'βend={beta}', color=color)

plt.axhline(y=eta, color='k', linestyle='--', label='η=1.96')
plt.xlabel('Iteration')
plt.ylabel('L_lip')
plt.title(f'Sensitivity analysis of six-dimensional systems (αstart={alpha_fixed})')
plt.legend()
plt.tight_layout()
plt.savefig(f"{csv_path}/6D_beta_sensitivity", dpi=600)
plt.show()


# === 图2: βend固定, αstart变化 ===
beta_fixed = 1.0
alpha_values = [0.1, 0.5, 1.0, 2.0]

plt.figure(figsize=(8,5))
for alpha, color in zip(alpha_values, colors):
    file_pattern = f"{csv_path}/epoch_lip_{alpha}_{beta_fixed}.csv"
    files = glob.glob(file_pattern)
    if not files:
        print(f"文件未找到: {file_pattern}")
        continue
    df = pd.read_csv(files[0])
    plt.plot(df['epoch'], df['lip'], label=f'αstart={alpha}', color=color)

plt.axhline(y=eta, color='k', linestyle='--', label='η=1.96')
plt.xlabel('Iteration')
plt.ylabel('L_lip')
plt.title(f'Sensitivity analysis of six-dimensional systems (βend={beta_fixed})')
plt.legend()
plt.tight_layout()
plt.savefig(f"{csv_path}/6D_alpha_sensitivity", dpi=600)
plt.show()
