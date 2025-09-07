#!!!!! env: gaffDescriptor
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from mordred import error as mordred_error

def calculate_descriptors(mol, calc):
    """
    尝试计算给定分子的所有描述符，包括2D和3D。如果3D计算失败，返回仅包含2D描述符的结果。
    """
    try:
        # 计算所有描述符（包括3D）
        descriptors_df = calc.pandas([mol])
        return descriptors_df
    except Exception as e:
        print(f"Error calculating full descriptors for molecule: {e}")
        return None

if __name__ == '__main__':
    # 读取数据
    mols = pd.read_csv("./data/ester_CH3_data.csv")

    # 将SMILES转换为RDKit分子对象
    mols['rdmol'] = mols['smiles'].map(Chem.MolFromSmiles)
    
    # 初始化Mordred描述符计算器
    calc = Calculator(descriptors)
    
    # 用于存储计算结果
    result_list = []
    
    # 遍历所有分子
    for idx, row in mols.iterrows():
        mol = row['rdmol']
        if mol is not None:
            try:
                # 计算当前分子的所有描述符
                descriptors_df = calculate_descriptors(mol, calc)
                
                if descriptors_df is not None:
                    # 如果成功计算所有描述符，将描述符与ID和SMILES合并
                    result_df = pd.concat([pd.DataFrame({'ID': [row['ID']], 'smiles': [row['smiles']]}), descriptors_df], axis=1)
                else:
                    # 如果描述符计算失败，保留ID和SMILES，描述符部分为空
                    result_df = pd.DataFrame({'ID': [row['ID']], 'smiles': [row['smiles']]})
            except Exception as e:
                print(f"Error calculating descriptors for SMILES: {row['smiles']}. Error: {e}")
                result_df = pd.DataFrame({'ID': [row['ID']], 'smiles': [row['smiles']]})
        else:
            # 如果RDKit无法解析SMILES，保留ID和SMILES，描述符部分为空
            print(f"Invalid SMILES: {row['smiles']}")
            result_df = pd.DataFrame({'ID': [row['ID']], 'smiles': [row['smiles']]})
        
        # 将结果保存到结果列表中
        result_list.append(result_df)
    
    # 合并所有结果
    final_df = pd.concat(result_list, ignore_index=True)
    
    # 填充所有无法计算的描述符为空
    final_df = final_df.applymap(lambda x: np.nan if isinstance(x, (mordred_error.Missing, mordred_error.Error)) else x)

    # 保存结果到CSV文件
    final_df.to_csv("./data/descriptors/01Des_Mordred.csv", index=False)

    # 统计包含空缺参数的行数
    missing_values_count = final_df.isnull().any(axis=1).sum()
    print(f"Number of rows with missing descriptor values: {missing_values_count}")

    print("描述符计算完成，结果已保存。")
