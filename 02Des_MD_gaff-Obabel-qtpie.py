# GAFF2 Force field inspired physical descriptors, a total of 130
# Rui Zhou 
# email: zhourui@licp.cas.cn
# Affiliation: LICP

import subprocess
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import shutil
from scipy import stats

def generate_frcmod(smiles, id_value):
    output_dir = './GAFF_Des_temp_file-bcc/'
    os.makedirs(output_dir, exist_ok=True)
    output_temp_dir = './GAFF_Des_temp_file-bcc/temp/'
    os.makedirs(output_temp_dir, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # 使用 RDKit 尝试嵌入分子构象
    # 注意：RDKit分子嵌入过程中未能生成有效的三维构象时会报错：Bad Conformer Id
    # 常见于复杂或具有环结构的分子，这些分子在嵌入过程中可能会遇到几何问题，导致生成无效构象
    # 增加 maxAttempts 可提高嵌入过程的成功率
    result = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=1000)

    if result == -1:  # 如果嵌入失败，使用 OpenBabel 生成构象
        print(f"Failed to generate a valid conformer for {smiles} using RDKit. Trying OpenBabel.")
        
        # 使用 OpenBabel 从 SMILES 生成三维构象
        obabel_output_mol2 = os.path.join(output_temp_dir, 'temp_obabel.mol2')
        try:
            subprocess.run(['obabel', '-:' + smiles, '--gen3d', '-O', obabel_output_mol2,  '--partialcharge qtpie'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"OpenBabel failed to generate conformer for {smiles}: {e}")
            return None  # 返回 None，表示未成功生成构象
        
        # 尝试加载 OpenBabel 生成的 mol2 文件回 RDKit 进行优化
        mol = Chem.MolFromMol2File(obabel_output_mol2, sanitize=True, removeHs=False)
        if mol is None:
            print(f"Failed to load OpenBabel generated mol2 file for {smiles}.")
            return None

    try:
        # 如果构象生成成功，尝试 UFF 优化
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        print(f"UFF optimization failed for {smiles}: {e}")
        print(f"Trying MMFF94 optimization for {smiles}.")
        try:
            # 如果 UFF 优化失败，尝试使用 MMFF94 优化
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
            AllChem.MMFFOptimizeMolecule(mol, mmff_props)
        except Exception as e:
            print(f"MMFF94 optimization failed for {smiles}: {e}")
            return None  # 返回 None，表示优化失败
    
    # 保存优化后的分子结构为 SDF 文件
    sdf_file = os.path.join(output_temp_dir, 'temp.sdf')
    w = Chem.SDWriter(sdf_file)
    w.write(mol)
    w.close()

    mol2_file = os.path.join(output_temp_dir, 'temp.mol2')
    try:
        subprocess.run(['obabel', sdf_file, '-O', mol2_file], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OpenBabel failed: {e}")

    frcmod_file = os.path.join(output_temp_dir, 'ANTECHAMBER.FRCMOD')
    try:
        subprocess.run(['antechamber', '-i', mol2_file, '-fi', 'mol2', '-o', os.path.join(output_dir, f'{id_value}_bcc.mol2'), 
                        '-fo', 'mol2'], check=True)
        subprocess.run(['parmchk2', '-i', os.path.join(output_dir, f'{id_value}_bcc.mol2'), '-f', 'mol2', 
                        '-o', frcmod_file, '-a', 'Y'], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Antechamber failed: {e}")

    new_frcmod_file = os.path.join(output_dir, f"{id_value}.FRCMOD")
    shutil.copy(frcmod_file, new_frcmod_file)

    # 清理临时文件
    for prefix in ['ANTECHAMBER', 'sqm', 'ATOMTYPE']:
        for filename in os.listdir('.'):
            if filename.startswith(prefix):
                os.remove(filename)
                print(f"Deleted: {filename}")

    descriptors = calculate_descriptors(new_frcmod_file)
    charge_df = extract_charges(os.path.join(output_dir, f'{id_value}_bcc.mol2'))
    charge_descriptors = calculate_charge_descriptors(charge_df)

    return descriptors, charge_descriptors

def calculate_descriptors(frcmod_file):
    descriptors = {}
    data = {
        'MASS': [],
        'BOND': [],
        'ANGLE': [],
        'DIHE': [],
        'IMPROPER': [],
        'NONBON': []
    }

    with open(frcmod_file, 'r') as f:
        lines = f.readlines()

        current_section = None

        for line in lines:
            line = line.strip()
            # 检查是否进入新的部分
            if line in data.keys():
                current_section = line
            elif current_section and line:
                # 对于 IMPROPER 部分，不做处理
                if current_section == 'IMPROPER':
                    values = []
                elif current_section == 'NONBON' and '-' in line:
                    # 如果是 NONBON 部分且名称部分包含 '-'，跳过该行
                    continue
                else:
                    # 提取数值部分和注释部分
                    values = split_line_with_annotations(line)

                # 如果结果的长度满足要求，并且 values 不是空列表
                if values and len(values) >= 3:
                    # 尝试将第二列及后面的值转换为浮点数，捕捉转换异常
                    try:
                        # 处理 values，其中数值部分转换为浮点数，注释部分保持不变
                        numeric_values = [float(x) if is_float(x) else x for x in values[1:]]
                        data[current_section].append([values[0]] + numeric_values)
                    except ValueError as e:
                        print(f"Failed to convert values to float in line: {line}, error: {e}")

    # 将数据转换为 DataFrame 并计算描述符
    for section, values in data.items():
        if values:
            df = pd.DataFrame(values)
            # 跳过第一列（原子类型或键类型），只处理数值列
            numeric_df = df.iloc[:, 1:]  # 从第二列开始
            for col_idx in range(numeric_df.shape[1]):  # 从第二列开始
                col_data = numeric_df.iloc[:, col_idx]

                if pd.api.types.is_numeric_dtype(col_data):
                    descriptors[f'{section}_col{col_idx + 1}_max'] = col_data.max()
                    descriptors[f'{section}_col{col_idx + 1}_mean'] = round(col_data.mean(), 4)
                    descriptors[f'{section}_col{col_idx + 1}_min'] = col_data.min()
                    descriptors[f'{section}_col{col_idx + 1}_median'] = col_data.median()                                    # 中位数
                    descriptors[f'{section}_col{col_idx + 1}_mode'] = stats.mode(col_data)[0][0]                            # 众数
                    descriptors[f'{section}_col{col_idx + 1}_variance'] = round(col_data.var(), 5)                          # 方差
                    descriptors[f'{section}_col{col_idx + 1}_std_dev'] = round(col_data.std(), 5)                           # 标准差
                    descriptors[f'{section}_col{col_idx + 1}_range'] = round(col_data.max() - col_data.min(), 5)            # 范围，最大值-最小值
                    descriptors[f'{section}_col{col_idx + 1}_skew'] = round(col_data.skew(), 5)                             # 偏度
                    descriptors[f'{section}_col{col_idx + 1}_kurt'] = round(col_data.kurtosis(), 5)                         # 峰度

    return descriptors

# 辅助函数：对分子BOND、ANGLE 和 DIHE 等部分名称进行合理划分
def split_line_with_annotations(line):
    parts = line.split()  # 先按空格进行初步分割

    # 1. 删除从 'same' 或 'Calculated' 开始的注释部分
    if 'same' in parts or 'Calculated' in parts:
        # 找到 'same' 或 'Calculated' 中第一个出现的索引
        comment_index = min(parts.index('same') if 'same' in parts else len(parts), 
                            parts.index('Calculated') if 'Calculated' in parts else len(parts))
        # 删除从该索引起的所有元素
        parts = parts[:comment_index]

    name_part = []
    values = []

    # 2. 识别第一个为 float 类型的元素，作为数值部分的起始
    for i, part in enumerate(parts):
        if is_float(part):  # 判断元素是否为 float 类型
            name_part = parts[:i]  # 之前的元素为名称部分
            remaining_parts = parts[i:]  # 从当前元素开始为数值部分
            break
    else:
        # 如果没有找到浮点数，说明可能只有名称部分
        name_part = parts
        remaining_parts = []

    # 将名称部分合并为一个字符串
    if name_part:
        values.append(' '.join(name_part))  # 把名称部分合并成一个字符串

    # 剩余的部分视为数值部分
    values.extend(remaining_parts)

    return values

# 辅助函数：检查一个字符串是否可以转换为 float
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def extract_charges(mol2_file):
    with open(mol2_file, 'r') as f:
        lines = f.readlines()
    
    start = lines.index('@<TRIPOS>ATOM\n') + 1
    end = lines.index('@<TRIPOS>BOND\n')
    atom_lines = lines[start:end]
    
    data = []
    for line in atom_lines:
        if line.strip():
            parts = line.split()
            atom_id = parts[0]               # 原子ID
            atom_name = parts[1]             # 原子名称
            x_coord = float(parts[2])        # X 坐标
            y_coord = float(parts[3])        # Y 坐标
            z_coord = float(parts[4])        # Z 坐标
            atom_type = parts[5]             # 原子类型
            charge = float(parts[8])         # 电荷信息
            data.append([atom_id, atom_name, x_coord, y_coord, z_coord, atom_type, charge])
    
    # 转换为DataFrame并命名列
    df = pd.DataFrame(data, columns=['Atom_ID', 'Atom_Name', 'X', 'Y', 'Z', 'Atom_Type', 'Charge'])
    return df

def calculate_charge_descriptors(charge_df):
    if not charge_df.empty:
        return {

            'Charge_max': charge_df['Charge'].max(),
            'Charge_mean': round(charge_df['Charge'].mean(), 5),
            'Charge_min': charge_df['Charge'].min(),
            'Charge_median': charge_df['Charge'].median(),
            'Charge_mode': stats.mode(charge_df['Charge'])[0][0],
            'Charge_variance': round(charge_df['Charge'].var(), 5),
            'Charge_std_dev': round(charge_df['Charge'].std(), 5),
            'Charge_range': charge_df['Charge'].max() - charge_df['Charge'].min(),
            'Charge_skew': round(charge_df['Charge'].skew(), 5),
            'Charge_kurt': round(charge_df['Charge'].kurtosis(), 5)

        }
    return {}

if __name__ == '__main__':
    dataframe = pd.read_csv("./data/ester_CH3_data.csv")
    smiles = dataframe['smiles']
    ids = dataframe['ID']

    all_descriptors = []

    # 遍历每个分子SMILES和对应的ID
    for smi, id_value in zip(smiles, ids):
        try:
            # 生成描述符
            descriptors, charge_descriptors = generate_frcmod(smi, id_value)

            # 组合ID、SMILES
            combined = {
                'ID': id_value,
                'smiles': smi,
            }
            combined.update(descriptors)
            combined.update(charge_descriptors)

            # 添加到结果列表
            all_descriptors.append(combined)
        except Exception as e:
            print(f"Failed to process {smi}: {e}")

    # 转换为DataFrame并保存
    final_df = pd.DataFrame(all_descriptors)
    # final_df.to_csv('test_gaff2_MD_Des.csv', index=False)

    # 修改 MASS_col1_* 和 MASS_col2_* 的列名
    final_df.columns = final_df.columns.str.replace(r'^MASS_col1_(.*)', r'Atomic_Mass_\1', regex=True)
    final_df.columns = final_df.columns.str.replace(r'^MASS_col2_(.*)', r'Atomic_Radius_\1', regex=True)

    # 修改 BOND_col1_* 和 BOND_col2_* 的列名
    final_df.columns = final_df.columns.str.replace(r'^BOND_col1_(.*)', r'Bond_Constant_\1', regex=True)
    final_df.columns = final_df.columns.str.replace(r'^BOND_col2_(.*)', r'Bond_Length_\1', regex=True)

    # 修改 ANGLE_col1_* 和 ANGLE_col2_* 的列名
    final_df.columns = final_df.columns.str.replace(r'^ANGLE_col1_(.*)', r'Angle_Constant_\1', regex=True)
    final_df.columns = final_df.columns.str.replace(r'^ANGLE_col2_(.*)', r'Angle_\1', regex=True)

    # 修改 DIHE_col1_*、DIHE_col2_*、DIHE_col3_* 和 DIHE_col4_* 的列名
    final_df.columns = final_df.columns.str.replace(r'^DIHE_col1_(.*)', r'Dihe_Periodicity_\1', regex=True)
    final_df.columns = final_df.columns.str.replace(r'^DIHE_col2_(.*)', r'Dihe_Constant_\1', regex=True)
    final_df.columns = final_df.columns.str.replace(r'^DIHE_col3_(.*)', r'Dihe_Phase_Shift_\1', regex=True)
    final_df.columns = final_df.columns.str.replace(r'^DIHE_col4_(.*)', r'Dihe_Ter_Phase_\1', regex=True)

    # 修改 NONBON_col1_* 和 NONBON_col2_* 的列名
    final_df.columns = final_df.columns.str.replace(r'^NONBON_col1_(.*)', r'Nb_vdW_Radius_\1', regex=True)
    final_df.columns = final_df.columns.str.replace(r'^NONBON_col2_(.*)', r'Nb_vdW_Potential_\1', regex=True)

    final_df.to_csv('./data/descriptors/02gaff2_MD_Des_qtpie.csv', index=False)
