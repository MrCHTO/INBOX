import pandas as pd
import numpy as np
from numpy.random import binomial, normal, choice

# 设置随机种子（确保数据可复现）
np.random.seed(42)

# 1. 生成基础字段
n = 1000
data = pd.DataFrame()
data['个体ID'] = range(1, n+1)
# 性别：男性512人，女性488人
data['性别'] = choice([0, 1], size=n, p=[0.512, 0.488])
# 年龄：18-85岁，偏态分布（45-65岁占42%）
age = np.random.normal(52, 15, n).astype(int)
age = np.clip(age, 18, 85)
data['年龄'] = age
# 城乡分布：城市658人，农村342人
data['城乡分布'] = choice([0, 1], size=n, p=[0.342, 0.658])

# 2. 生成糖尿病状态（关联年龄、家族史）
family_history = choice([0, 1], size=n, p=[0.7, 0.3])  # 30%有家族史
# 有家族史糖尿病患病率28%，无家族史8%；前期患病率统一35%
diabetes_prob = np.where(family_history == 1, 0.28, 0.08)
diabetes = np.where(np.random.rand(n) < diabetes_prob, 1, 
                   np.where(np.random.rand(n) < 0.35, 2, 0))
data['糖尿病状态'] = diabetes
data['家族糖尿病史'] = family_history

# 3. 生成BMI（关联糖尿病状态）
bmi_mean = np.where(diabetes == 1, 27.8, 23.2)  # 患者均值更高
bmi = normal(bmi_mean, 3.5, n)
bmi = np.clip(bmi, 16.5, 42.8)
data['BMI'] = np.round(bmi, 1)

# 4. 生成体育锻炼时长（关联城乡、糖尿病状态）
exercise_mean = np.where((data['城乡分布'] == 1) & (diabetes == 0), 85,
                        np.where((data['城乡分布'] == 1) & (diabetes == 1), 42,
                                 np.where((data['城乡分布'] == 0) & (diabetes == 0), 56, 38)))
exercise = normal(exercise_mean, 30, n)
exercise = np.clip(exercise, 0, 360).astype(int)
data['每周锻炼时长（分钟）'] = exercise

# 5. 生成蔬果摄入量（关联糖尿病状态）
veg_fruit_mean = np.where(diabetes == 1, 260, 380)  # 患者摄入更少
veg_fruit = normal(veg_fruit_mean, 120, n)
veg_fruit = np.clip(veg_fruit, 100, 1200).astype(int)
data['每日蔬果摄入（克）'] = veg_fruit

# 6. 生成吸烟状态（关联性别、糖尿病状态）
smoke_prob = np.where((data['性别'] == 0) & (diabetes == 1), 0.35,
                     np.where((data['性别'] == 0) & (diabetes == 0), 0.22,
                              np.where((data['性别'] == 1) & (diabetes == 1), 0.05, 0.02)))
smoke = np.where(np.random.rand(n) < smoke_prob, 1,
                np.where(np.random.rand(n) < 0.1, 2, 0))  # 10%曾经吸烟
data['吸烟状态'] = smoke

# 7. 生成收缩压（关联糖尿病状态）
bp_mean = np.where(diabetes == 1, 145, 125)  # 患者血压更高
bp = normal(bp_mean, 15, n)
bp = np.clip(bp, 95, 185).astype(int)
data['收缩压（mmHg）'] = bp

# 保存为Excel文件
data.to_excel('中国人群糖尿病及相关健康指标_1000人数据.xlsx', index=False)
print("文件生成完成！")