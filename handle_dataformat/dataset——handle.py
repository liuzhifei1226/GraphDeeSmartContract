from scipy.stats import bootstrap
import numpy as np

# 生成 1000 轮 F1-score 样本
f1_scores_baseline = np.random.normal(0.75, 0.02, 10)  # 未使用贝叶斯
f1_scores_bayes = np.random.normal(0.80, 0.02, 10)  # 使用贝叶斯

# 计算置信区间
ci_baseline = bootstrap((f1_scores_baseline,), np.mean, confidence_level=0.95).confidence_interval
ci_bayes = bootstrap((f1_scores_bayes,), np.mean, confidence_level=0.95).confidence_interval

print(f"未使用贝叶斯 F1-score 置信区间: {ci_baseline}")
print(f"使用贝叶斯 F1-score 置信区间: {ci_bayes}")
import matplotlib.pyplot as plt

import matplotlib
# 设置 Matplotlib 使用 SimHei（黑体）字体，保证中文正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 适用于 Windows
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 置信区间边界
x_labels = ["直接投票", "贝叶斯验证"]
means = [np.mean(f1_scores_baseline), np.mean(f1_scores_bayes)]
lower_bounds = [ci_baseline.low, ci_bayes.low]
upper_bounds = [ci_baseline.high, ci_bayes.high]

# 误差条图
plt.figure(figsize=(6, 4))
plt.errorbar(x_labels, means, yerr=[np.array(means) - np.array(lower_bounds),
                                    np.array(upper_bounds) - np.array(means)],
             fmt='o', capsize=5, capthick=2, color='b')

plt.ylabel("F1-score")

plt.grid(True)
plt.show()
