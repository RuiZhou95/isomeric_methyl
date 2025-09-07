import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from stability_selection import StabilitySelection
import matplotlib.pyplot as plt
import os

# 修改目标变量名称 Glass_T VI
target_var = "Glass_T"

# 动态创建输出路径（和 target_var 关联）
out_dir = f'./figure/ElasticNet_stability/{target_var}/'
os.makedirs(out_dir, exist_ok=True)

path = f"./data/{target_var}_ElasticNet_stability_opt_Des/"  
os.makedirs(path, exist_ok=True)
df = pd.read_csv(f'./data/{target_var}_Statistical_filtering/05Cor_descriptor.csv')

X_all = df.drop(df.columns[0:2], axis=1)
y_all = df[target_var]

# train-test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
train_data.to_pickle(f'./data/{target_var}_train_data.pkl')
test_data.to_pickle(f'./data/{target_var}_test_data.pkl')

X_train = train_data.drop(train_data.columns[0:2], axis=1)
y_train = train_data[target_var]
X_test = test_data.drop(test_data.columns[0:2], axis=1)
y_test = test_data[target_var]

# 标准化：只fit训练集，避免数据泄漏
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X_all)

# ======================================================
# 🔹 Stability Selection + ElasticNet
# ======================================================
print(f"\nRunning Stability Selection with ElasticNet for target variable: {target_var} ...")

# 定义基模型（ElasticNet）
base_estimator = ElasticNet(
    l1_ratio=0.7,       # L1 / L2 权重，0.7 偏向稀疏解
    max_iter=20000,
    random_state=1
)

# 稳定性选择
stability_model = StabilitySelection(
    base_estimator=base_estimator,
    lambda_name="alpha",                # 对应 ElasticNet 的正则化参数
    lambda_grid=np.logspace(-2, 3, 50), # 搜索范围 (-2, 3, 50)
    n_bootstrap_iterations=1000,         # bootstrap 次数
    sample_fraction=0.75,               # 每次子样本比例
    threshold=0.9,                      # 未使用，后续通过取前20
    random_state=1,
    n_jobs=-1
)

stability_model.fit(X, y_all)

# 每个特征的最高选择概率
feature_scores = stability_model.stability_scores_.mean(axis=1)
feature_scores_series = pd.Series(feature_scores, index=X_all.columns)

# 取前20个最稳定特征
top20_features = feature_scores_series.sort_values(ascending=False).head(20)
print("\nTop 20 most stable features:")
print(top20_features)

# 保存结果
df_top20 = df[["mix_ID", target_var]].join(df[top20_features.index])
out_csv = path + f"{target_var}_Top20_StabilitySelection_ElasticNet_descriptor.csv"
df_top20.to_csv(out_csv, index=False)
print(f"The first 20 features have been saved to: {out_csv}")

feature_scores_df = feature_scores_series.reset_index()
feature_scores_df.columns = ["feature", "selection_probability"] # selection_probability==feature importance
feature_scores_df = feature_scores_df.sort_values(by="selection_probability", ascending=False)
out_scores_csv = out_dir + f"{target_var}_AllFeature_StabilitySelection_scores.csv"
feature_scores_df.to_csv(out_scores_csv, index=False)
print(f"All feature selection probabilities have been saved to: {out_scores_csv}")

# 画出前20个特征的选择概率
plt.figure(figsize=(8, 6))
top20_features.plot(kind="bar", color="#01579B")
plt.ylabel(f"{target_var} Selection probability")
plt.tight_layout()
out_fig = out_dir + "Top20_StabilitySelection_scores.png"
plt.savefig(out_fig, dpi=600)
plt.close()
print(f"The first 20 feature selection probability graphs have been saved to {out_fig}")
