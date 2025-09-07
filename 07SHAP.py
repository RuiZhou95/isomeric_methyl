import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score
from joblib import Parallel, delayed
import shap
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib  # 导入 matplotlib

# 修改目标变量名称 Glass_T VI
target_var = "Glass_T"

# 1. 数据加载与预处理
path = "./data/"
train_file = f'{target_var}_train_data.pkl'
train_df = pd.read_pickle(path + train_file)
test_file = f'{target_var}_test_data.pkl'
test_df = pd.read_pickle(path + test_file)

X_train = train_df.drop(train_df.columns[0:2], axis=1)
print(X_train)
y_train = train_df[target_var]
print(y_train)
X_test = test_df.drop(test_df.columns[0:2], axis=1)
y_test = test_df[target_var]

# select_feature = ['MPC10', 'MDEC-22', 'EState_VSA3', 'ATSC8d', 'MDEC-11'] # VI
select_feature = ['MDEC-23', 'MDEC-12', 'AATSC8v', 'AATSC6m'] # Glass_T

X_train = X_train[select_feature]
X_test = X_test[select_feature]

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为 DataFrame，方便后续操作
X_train_df = pd.DataFrame(X_train_scaled, columns=select_feature)
X_test_df = pd.DataFrame(X_test_scaled, columns=select_feature)

# 2. 使用简化参数的 XGBoost 模型
best_xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42,
    n_estimators=20,       # 减少树的数量
    max_depth=2,           # 降低树深度
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=1
)

# 3. KFold 交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_xgb_model, X_train_df, y_train, cv=kf, scoring="r2")
print(f"KFold CV R2 mean: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# 训练 XGBoost 模型
best_xgb_model.fit(X_train_df, y_train)

# 预测并打印分数
train_score = best_xgb_model.score(X_train_df, y_train)
test_score = best_xgb_model.score(X_test_df, y_test)
print(f"Training set score: {train_score}")
print(f"Test set score: {test_score}")

explainer = shap.TreeExplainer(best_xgb_model, approximate=True)

# 并行化计算 SHAP 值
def compute_shap_for_chunk(chunk):
    return explainer.shap_values(chunk)

# 分块并并行计算 SHAP 值
start_time = time.time()
shap_values_list = Parallel(n_jobs=-1)(delayed(compute_shap_for_chunk)(chunk) for chunk in np.array_split(X_train_df, 8))
shap_values = np.concatenate(shap_values_list, axis=0)
end_time = time.time()

plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = 12

inch2cm = 1 / 2.54

# 6. SHAP 可视化和保存
# cmap 颜色选项：coolwarm viridis plasma inferno magma cividis Blues Reds PiYG
plt.figure(figsize=(14 * inch2cm, 50 * inch2cm))
shap.summary_plot(
    shap_values, 
    X_train_df, 
    max_display=40, 
    show=False, 
    cmap=plt.get_cmap("coolwarm"), 
    plot_size=(5, 5), 
    sort=False
)

# 修改散点大小
ax = plt.gca()
for collection in ax.collections:
    collection.set_sizes([40])  # 默认 20 左右，改大就会放大点

plt.tight_layout()
plt.xlabel("SHAP value", fontsize=18, fontname='Times New Roman')  
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=10, fontname='Times New Roman', rotation=30)
plt.xlabel('SHAP value', fontsize=18)

plt.savefig(f'./figure/shap/{target_var}/shap_value.png', dpi=600)
plt.close()

# 生成交互SHAP值图
shap_interaction_values = explainer.shap_interaction_values(X_train_df)

plt.figure(figsize=(100 * inch2cm, 100 * inch2cm))
shap.summary_plot(
    shap_interaction_values,
    X_train_df,
    max_display=40,
    cmap=plt.get_cmap("coolwarm"),
    show=False
)

# 交互 SHAP 散点图
ax = plt.gca()
for collection in ax.collections:
    collection.set_sizes([20])  # 修改交互 SHAP 散点大小

plt.tight_layout()
plt.xticks(fontsize=12, fontname='Times New Roman')
plt.yticks(fontsize=12, fontname='Times New Roman')

plt.savefig(f'./figure/shap/{target_var}/shap_interaction_values.png', bbox_inches='tight', dpi=600)
plt.close()

print("SHAP analysis and visualizations completed and saved.")
