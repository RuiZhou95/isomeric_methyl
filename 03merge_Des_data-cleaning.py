import pandas as pd
import numpy as np

# 读取两个CSV文件
gaff_df = pd.read_csv('./data/descriptors/02gaff2_MD_Des_qtpie.csv')
mordred_df = pd.read_csv('./data/descriptors/01Des_Mordred.csv')

# 拼接Des_Mordred.csv：排除ID和SMILES列
mordred_exclude_id_smiles = mordred_df.drop(columns=['ID', 'smiles'])

# 将gaff数据按列拼接到mw_vdw数据上
final_combined_df = pd.concat([gaff_df, mordred_exclude_id_smiles], axis=1)

# 打印拼接后的行数和列数
print(f"Initial combined data has {final_combined_df.shape[0]} rows and {final_combined_df.shape[1]} columns.")

# 定义错误值类型
error_values = (np.nan, np.inf, -np.inf)

# 删除所有空缺值和错误值列
final_combined_df = final_combined_df.loc[:, final_combined_df.isin(error_values).sum() <= 0]

print("Initial combined data after removing columns with >0 missing/error values:")
print(f"{final_combined_df.shape[0]} rows and {final_combined_df.shape[1]} columns.")

# 分离出ID、SMILES
protected_columns = ['ID', 'smiles']
protected_df = final_combined_df[protected_columns]

# 过滤掉字符串列以及保护列
numeric_df = final_combined_df.drop(columns=protected_columns).select_dtypes(include=[np.number])

# 3）删除标准差为0的列
non_zero_std = numeric_df.std() != 0
numeric_df = numeric_df.loc[:, non_zero_std]

# 4）保留数值列的参数精度到4位小数
numeric_df = numeric_df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

# 将保护列与处理后的数值列合并
final_combined_df = pd.concat([protected_df, numeric_df], axis=1)

# 保存为all_clean_Des.csv
final_combined_df.to_csv('./data/descriptors/03all_clean_Des.csv', index=False)

# 打印最终清洗后的数据的行数和列数
print(f"Cleaned data has {final_combined_df.shape[0]} rows and {final_combined_df.shape[1]} columns.")
