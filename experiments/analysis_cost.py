import os
import matplotlib.pyplot as plt
import pandas as pd
import re

raw_path = 'experiments/'
save_path = 'experiments/analysis_result/'
input_path = 'experiments/output_file'
files = [name for name in os.listdir(input_path)]

data = []
labels = []

for file in files:
    if file.lower().endswith('cost.csv'):
        df = pd.read_csv(os.path.join(input_path, file))
        
        model_name = re.sub(r'cost\.csv$', '', file, flags=re.IGNORECASE).rstrip('_- ')
        
        data.append(df['predict_cost'].dropna().values)
        labels.append(model_name)

# Vẽ boxplot
plt.figure(figsize=(12, 6))
box = plt.boxplot(data, tick_labels=labels, patch_artist=True)  # ✅ dùng tick_labels

# Tô màu cho các box
colors = plt.cm.Set3.colors
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Trang trí
plt.title('Cost Prediction Distribution by Model', fontsize=14, weight='bold')
plt.ylabel('Predicted Cost', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(os.path.join(save_path, 'cost_distribution.png'))
plt.close()
