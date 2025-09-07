import pandas as pd

des_path = './data/descriptors/03all_clean_Des.csv'
df_des = pd.read_csv(des_path)

# 只保留 ID 和 描述符（去掉 smiles）
df_descriptors = df_des.drop(columns=['smiles'])
df_descriptors.set_index('ID', inplace=True)

# 读取混合方案
mix_plan_path = './data/mix_plan.csv'
df_mix = pd.read_csv(mix_plan_path)

# 结果列表
results = []

# 遍历每个方案
for _, row in df_mix.iterrows():
    mix_id = int(row['mix_ID'])  # 确保为正整数
    weights = [row[f'w_mol{i}'] for i in range(5)]
    
    # 加权求和
    weighted_sum = (df_descriptors.T * weights).T.sum(axis=0)
    
    # 保存结果
    result = {'mix_ID': mix_id}
    result.update(weighted_sum.to_dict())
    results.append(result)


out_path = './data/descriptors/04all_mix_Des.csv'
df_out = pd.DataFrame(results)

df_out['mix_ID'] = df_out['mix_ID'].astype(int)

df_out.to_csv(out_path, index=False, float_format="%.6f")

print(f"✅ The descriptor calculation of the mixture solution is completed: {out_path}")
