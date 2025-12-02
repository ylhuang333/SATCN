import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


CONF_LEVEL = 0.95
INPUT_PATH = "数据.xlsx"
OUTPUT_PATH = "高斯过程局部变异性.xlsx"


CONF_LEVEL = max(0.50, min(0.95, float(CONF_LEVEL)))


df = pd.read_excel(INPUT_PATH)
x = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values
n = len(x)

print(f"数据加载完成，共 {n} 个数据点")
print(f"数据范围: [{x.min():.4f}, {x.max():.4f}]")
print(f"数据均值: {x.mean():.4f}, 标准差: {x.std():.4f}")


X = np.arange(n).reshape(-1, 1)
y = x.copy()


kernel = (ConstantKernel(1.0) * RBF(length_scale=10.0) +  # 长期趋势，ConstantKernel (常数核)控制核函数的幅度（垂直方向的缩放），值越大，函数波动幅度越大
          ConstantKernel(0.5) * RBF(length_scale=2.0) +   # 短期波动，RBF (径向基函数核)衡量数据点之间的相似性，控制函数的平滑度。值大：函数变化缓慢，捕捉长期趋势；值小：函数变化快速，捕捉短期波动
          WhiteKernel(noise_level=0.2, noise_level_bounds='fixed'))                   # 噪声项，值越大，数据中的随机噪声越多

print("核函数配置:", kernel)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-5,           # 添加少量数值稳定性
    n_restarts_optimizer=10,  # 优化重启次数，避免局部最优
    random_state=42
)

print("开始训练高斯过程模型...")
gp.fit(X, y)


print("优化后的核函数:", gp.kernel_)
print("模型训练完成")


print("进行预测...")
y_pred, y_std = gp.predict(X, return_std=True)


alpha = 1.0 - CONF_LEVEL
z_critical = norm.ppf(1 - alpha / 2.0)


gp_lower = y_pred - z_critical * y_std
gp_upper = y_pred + z_critical * y_std
gp_interval_width = gp_upper - gp_lower


mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

print(f"\n模型性能指标:")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")
print(f"平均预测标准差: {y_std.mean():.4f}")
print(f"预测标准差范围: [{y_std.min():.4f}, {y_std.max():.4f}]")


result_df = pd.DataFrame({
    '时间索引': np.arange(n),
    '原始数据': y,
    'GP预测值': y_pred,
    'GP预测标准差': y_std,
    f'GP_{int(CONF_LEVEL*100)}%_下界': gp_lower,
    f'GP_{int(CONF_LEVEL*100)}%_上界': gp_upper,
    'GP区间宽度': gp_interval_width,
    '残差': y - y_pred,
    '标准化残差': (y - y_pred) / y_std
})

result_df.to_excel(OUTPUT_PATH, index=False)
print(f"\n结果已保存到: {OUTPUT_PATH}")


plt.figure(figsize=(15, 12))


plt.subplot(3, 1, 1)
plt.plot(X, y, 'o', markersize=3, alpha=0.7, label='true data', color='blue')
plt.plot(X, y_pred, '-', linewidth=2, label='GP prediction', color='red')
plt.fill_between(X.ravel(), gp_lower, gp_upper,
                 alpha=0.3, label=f'GP {int(CONF_LEVEL*100)}% confidence interval', color='orange')
plt.xlabel('points')
plt.ylabel('salinity')
plt.title('GP confidence interval prediction')
plt.legend()
plt.grid(True, alpha=0.3)


plt.subplot(3, 1, 2)
plt.plot(X, y_std, 'g-', linewidth=2, label='prediction σ', alpha=0.8)
plt.fill_between(X.ravel(), 0, y_std, alpha=0.3, color='green')
plt.xlabel('points')
plt.ylabel('σ')
plt.title('prediction σ')
plt.legend()
plt.grid(True, alpha=0.3)


plt.subplot(3, 1, 3)
residuals = y - y_pred
plt.plot(X, residuals, 'o', markersize=3, alpha=0.7, label='residual', color='purple')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.fill_between(X.ravel(), -2*y_std, 2*y_std, alpha=0.2, label='±2σ range', color='gray')
plt.xlabel('points')
plt.ylabel('residual')
plt.title('residual analysis')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('高斯过程局部变异性分析.png', dpi=300, bbox_inches='tight')
plt.show()

z_scores = np.abs((y - y_pred) / y_std)
outlier_mask = z_scores > 2.0  # 2σ阈值

print(f"\n异常点检测 (|标准化残差| > 2):")
print(f"异常点数量: {np.sum(outlier_mask)}")
print(f"异常点比例: {np.mean(outlier_mask)*100:.2f}%")

if np.any(outlier_mask):
    outlier_indices = np.where(outlier_mask)[0]
    outlier_df = result_df.iloc[outlier_indices].copy()
    outlier_df['Z分数'] = z_scores[outlier_mask]
    print("\n异常点详情:")
    print(outlier_df[['时间索引', '原始数据', 'GP预测值', '标准化残差', 'Z分数']].round(4))


print(f"\n变异性区域分析:")
high_var_mask = y_std > np.percentile(y_std, 75)  # 高变异性区域（上四分位数）
low_var_mask = y_std < np.percentile(y_std, 25)   # 低变异性区域（下四分位数）

print(f"高变异性区域点数: {np.sum(high_var_mask)}")
print(f"低变异性区域点数: {np.sum(low_var_mask)}")
print(f"高变异性区域平均标准差: {y_std[high_var_mask].mean():.4f}")
print(f"低变异性区域平均标准差: {y_std[low_var_mask].mean():.4f}")


report = f"""
高斯过程局部变异性分析报告
========================================
分析时间: {pd.Timestamp.now()}
数据文件: {INPUT_PATH}
数据点数: {n}
置信水平: {CONF_LEVEL*100}%

模型性能:
- RMSE: {rmse:.4f}
- R²: {r2:.4f}
- 平均预测标准差: {y_std.mean():.4f}

不确定性统计:
- 最小标准差: {y_std.min():.4f}
- 最大标准差: {y_std.max():.4f}
- 标准差中位数: {np.median(y_std):.4f}

异常点检测:
- 异常点数量: {np.sum(outlier_mask)}
- 异常点比例: {np.mean(outlier_mask)*100:.2f}%

变异性区域:
- 高变异性区域: {np.sum(high_var_mask)} 点
- 低变异性区域: {np.sum(low_var_mask)} 点
- 高/低变异性比: {y_std[high_var_mask].mean()/y_std[low_var_mask].mean():.2f}

核函数参数:
{gp.kernel_}
"""

with open('高斯过程分析报告.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n详细分析报告已保存到: 高斯过程分析报告.txt")
print("分析完成！")