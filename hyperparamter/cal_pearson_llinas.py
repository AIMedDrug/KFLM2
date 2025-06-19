import pandas as pd
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error,mean_absolute_error

# 配置参数
data_name = "Llinas_opt"  # 替换为实际数据名称
model_name = "CombinedAtt"  # 替换为实际模型名称
num_iters = 10  # idx从0到9
results_dir = "result"  # 结果文件目录

# 创建结果存储DataFrame
results_df = pd.DataFrame(columns=['Iteration', 'Pearson', 'P-value', 'RMSE', 'MAE'])

for idx in range(num_iters):
    # 构建文件名
    set1_file = os.path.join(results_dir, f"predictions_{data_name}_{model_name}_iter{idx}_set1.csv")
    set2_file = os.path.join(results_dir, f"predictions_{data_name}_{model_name}_iter{idx}_set2.csv")
    
    # 检查文件是否存在并读取
    if os.path.exists(set1_file) and os.path.exists(set2_file):
        df_set1 = pd.read_csv(set1_file)
        df_set2 = pd.read_csv(set2_file)
        # 合并对应idx的set1和set2
        combined = pd.concat([df_set1, df_set2], ignore_index=True)
        
        # 计算指标
        if 'Predictions' in combined.columns and 'True_Values' in combined.columns:
            # Pearson相关系数
            pearson_corr, p_value = pearsonr(combined['Predictions'], combined['True_Values'])
            
            mae = mean_absolute_error(combined['True_Values'], combined['Predictions'])
            # RMSE
            rmse = np.sqrt(mean_squared_error(combined['True_Values'], combined['Predictions']))
            
            # 存储结果
            results_df.loc[idx] = [idx, pearson_corr, p_value, rmse, mae]
        else:
            print(f"Warning: Required columns missing in iteration {idx}")
    else:
        print(f"Warning: Files missing for iteration {idx}")

# 保存结果到新文件
output_file = os.path.join(results_dir, f"performance_metrics_{data_name}_{model_name}.csv")
results_df.to_csv(output_file, index=False)
print(f"Performance metrics saved to: {output_file}")

# 打印统计摘要
print("\nPerformance Metrics Summary:")
print(f"Average Pearson: {results_df['Pearson'].mean():.4f} ± {results_df['Pearson'].std():.4f}")
print(f"Average RMSE: {results_df['RMSE'].mean():.4f} ± {results_df['RMSE'].std():.4f}")
print(f"Average MAE: {results_df['MAE'].mean():.4f} ± {results_df['MAE'].std():.4f}")
