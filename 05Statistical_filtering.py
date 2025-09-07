import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.spatial.distance import pdist, squareform
from minepy import MINE
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 1_remove_low_variance_features
file_path = "./data/VI_Statistical_filtering/"
# file_path = "./data/Glass_T_Statistical_filtering/"

all_data = pd.read_csv('./data/04-A-all_mix_VI_glassT_Des.csv') # viscosity index

id_mix = all_data.iloc[:, 0]

X = all_data.iloc[:, 3:]  # 只选择描述符列，第4列（包含）到最后一列

y = all_data.iloc[:, 1]  # VI
# y = all_data.iloc[:, 2]  # glass tempture

print(f"Number of descriptor columns before any processing: {X.shape}")

# VarianceThreshold 方差过滤
vt = VarianceThreshold(threshold=0.1)
X_selected = vt.fit_transform(X)
lowvariance_data = pd.DataFrame(X_selected)

# 获取选择后的列名
all_name = X.columns.values.tolist()
select_name_index0 = vt.get_support(indices=True)
select_name0 = [all_name[i] for i in select_name_index0]

lowvariance_data.columns = select_name0
lowvariance_data_y = pd.concat((y, lowvariance_data), axis=1)
lowvariance_data_y.to_pickle(file_path + "Variance_descriptor.pkl")

# 打印方差过滤后描述符列的数量
print(f"Number of descriptor columns after variance filtering: {lowvariance_data.shape}") 

# 特征过滤后的数据
all_data = lowvariance_data_y
data = all_data.iloc[:, all_data.columns != "mix_ID"]
descriptor_data = data.iloc[:, (data.columns != "VI") & (data.columns != "Glass_T")]
descriptor_name_list = list(descriptor_data)

# 数据标准化
scaler = StandardScaler()
data_scaler = scaler.fit_transform(descriptor_data)
DataFrame_data_scaler = pd.DataFrame(data_scaler)

# 计算 Pearson 和 Spearman 相关系数
print("Calculating Pearson and Spearman correlations...")
data_pearson = DataFrame_data_scaler.corr(method='pearson').iloc[:, 0]
data_spearman = DataFrame_data_scaler.corr(method='spearman').iloc[:, 0]

# 定义距离相关系数函数
def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    return np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

# 并行计算距离相关系数
def calculate_distcorr(i):
    return distcorr(data_scaler[:, i], data_scaler[:, 0])

print("Calculating distance correlations in parallel...")
distance_correlation_list = Parallel(n_jobs=-1)(delayed(calculate_distcorr)(i) for i in tqdm(range(0, data_scaler.shape[1])))

# MIC 计算部分：不使用并行化，保持串行
print("Calculating MIC (serial, due to parallelization limitations)...")
mic_correlation_list = []
mine = MINE(alpha=0.6, c=15)
for i in tqdm(range(0, data_scaler.shape[1])):
    mine.compute_score(data_scaler[:, i], data_scaler[:, 0])
    mic_correlation_list.append(mine.mic())

# 选择器函数
def selection_by_threshold(correlation_list, threshold):
    return [1 if abs(corr) > threshold else 0 for corr in correlation_list]


# 相关性阈值 
pearson_threshold = 0.88 # VI=0.88 glass_T=0.9 统一用0.88
spearman_threshold = 0.8
distance_threshold = 0.4
mic_threshold = 0.3

# 根据阈值选择特征
pearson_selection = selection_by_threshold(data_pearson, pearson_threshold)
spearman_selection = selection_by_threshold(data_spearman, spearman_threshold)
distance_selection = selection_by_threshold(distance_correlation_list, distance_threshold)
mic_selection = selection_by_threshold(mic_correlation_list, mic_threshold)

# 计算总和并应用不同阈值进行过滤
sum_list = np.array(pearson_selection) + np.array(spearman_selection) + np.array(distance_selection) + np.array(mic_selection)

# 确保sum_list和descriptor_data的长度一致
assert len(sum_list) == descriptor_data.shape[1], "sum_list length does not match descriptor_data columns"

# 对描述符进行布尔过滤
descriptor_filter1 = descriptor_data.loc[:, sum_list >= 1]
descriptor_filter2 = descriptor_data.loc[:, sum_list >= 2]
descriptor_filter3 = descriptor_data.loc[:, sum_list >= 3]
descriptor_filter4 = descriptor_data.loc[:, sum_list >= 4]

filter_data1 = pd.concat([id_mix, y, descriptor_filter1], axis=1)
filter_data2 = pd.concat([id_mix, y, descriptor_filter2], axis=1)
filter_data3 = pd.concat([id_mix, y, descriptor_filter3], axis=1)
filter_data4 = pd.concat([id_mix, y, descriptor_filter4], axis=1)

filter_data1.to_csv(file_path + 'filter_threshold_1.csv', index=False)
filter_data2.to_csv(file_path + 'filter_threshold_2.csv', index=False)
filter_data3.to_csv(file_path + 'filter_threshold_3.csv', index=False)
filter_data4.to_csv(file_path + '05Cor_descriptor.csv', index=False)

# 打印相关系数过滤后描述符列的数量（sum_list >= 4）
print(f"Number of descriptor columns after correlation filtering: {filter_data4.shape}")
print("Filtering complete. Files saved.")
