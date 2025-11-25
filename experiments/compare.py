import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import re
from pathlib import Path
from collections import defaultdict

raw_path = 'experiments/'
save_path = 'experiments/analysis_result/'
os.makedirs(save_path, exist_ok=True)

METHOD_NAMES = {
    'random': 'Random',
    'drl_prediction_with_history_task_observation': 'DRL với History',
    'drl_prediction': 'DRL',
    'fast_detect_outage': 'Fast Detect Outage (FDO)'
}

METHOD_COLORS = {
    'random': '#FF6B6B',
    'drl_prediction_with_history_task_observation': '#4ECDC4',
    'drl_prediction': '#45B7D1',
    'fast_detect_outage': '#96CEB4'
}

def add_server_service(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm thông tin server_id và service vào dataframe"""
    server_ids = []
    service_ids = []
    for res_str in df['results']:
        try:
            res = ast.literal_eval(res_str)
        except:
            res = {}
        backend = res.get('backend', '')
        if backend:
            try:
                port = int(backend.split(":")[-1].split("/")[0])
                last_three = port % 1000
                server_id = (last_three // 100) - 1
            except:
                server_id = None
        else:
            server_id = None
        server_ids.append(server_id)
        model = res.get('payload', {}).get('model', '')
        if model == "ssd":
            service_id = 9
        elif model == "resnet34":
            service_id = 3
        else:
            service_id = 0
        service_ids.append(service_id)
    df['server_id'] = server_ids
    df['service'] = service_ids
    return df

def identify_method(filename):
    """Xác định phương pháp từ tên file"""
    filename_lower = filename.lower()
    
    if 'fast_detect_outage' in filename_lower or 'fdo' in filename_lower:
        return 'fast_detect_outage'
    elif 'drl_prediction_with_history' in filename_lower:
        return 'drl_prediction_with_history_task_observation'
    elif 'drl_prediction' in filename_lower:
        return 'drl_prediction'
    elif 'random' in filename_lower:
        return 'random'
    
    return None

def load_and_analyze_csv(file_path, method_name):
    """Đọc CSV và tính toán thống kê"""
    try:
        df = pd.read_csv(file_path)
        df = add_server_service(df)
        
        if 'total_delay' not in df.columns:
            print(f"Warning: File {file_path} không có cột 'total_delay'")
            return None
        
        delays = df['total_delay'].dropna()
        if len(delays) == 0:
            return None
        
        stats = {
            'method': method_name,
            'mean': delays.mean(),
            'median': delays.median(),
            'std': delays.std(),
            'min': delays.min(),
            'max': delays.max(),
            'q25': delays.quantile(0.25),
            'q75': delays.quantile(0.75),
            'count': len(delays),
            'data': delays.tolist()
        }

        if 'compute_delay' in df.columns:
            compute_delays = df['compute_delay'].dropna()
            if len(compute_delays) > 0:
                stats['compute_delay_mean'] = compute_delays.mean()
                stats['compute_delay_data'] = compute_delays.tolist()
        
        return stats
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {str(e)}")
        return None

def find_result_files(output_dir, n_users=None):
    """Tìm các file CSV kết quả cho 3 phương pháp"""
    script_dir = Path(__file__).parent
    output_path = None

    possible_paths = []
    
    direct_path = Path(output_dir)
    if direct_path.exists() and direct_path.is_dir():
        possible_paths.append(direct_path)
        print(f"Tìm thấy thư mục (đường dẫn trực tiếp): {direct_path.absolute()}")

    project_root = script_dir.parent
    project_path = project_root / output_dir
    if project_path.exists() and project_path.is_dir():
        possible_paths.append(project_path)
        print(f"Tìm thấy thư mục (trong project root): {project_path.absolute()}")
    
    if not possible_paths:
        print(f"Không tìm thấy thư mục .")
        return {}, None
    
   
    if possible_paths:
        output_path = possible_paths[0]
    else:
        print(f"\n Không tìm thấy thư mục: {output_dir}")
        return {}, None
    
    output_path = output_path.resolve() 
    print(f"\n Đang sử dụng thư mục: {output_path}")
    print(f"   Đường dẫn tuyệt đối: {output_path.absolute()}")
    
    csv_files = list(output_path.glob('results_*.csv'))
    print(f"\nTìm thấy {len(csv_files)} file CSV:")
    
    if len(csv_files) == 0:
        print(f"  Không có file results_*.csv nào trong thư mục!")
        print(f"   Các file có trong thư mục:")
        all_files = list(output_path.glob('*'))
        for f in all_files[:10]:  # Hiển thị 10 file đầu tiên
            print(f"      - {f.name}")
        if len(all_files) > 10:
            print(f"      ... và {len(all_files) - 10} file khác")
        return {}, output_path
    
    method_files = defaultdict(list)
    
    for csv_file in csv_files:
        method = identify_method(csv_file.name)
        if method:
            method_files[method].append(csv_file)
            print(f"   {csv_file.name} → {METHOD_NAMES.get(method, method)}")
        else:
            print(f"    {csv_file.name} → Không xác định được phương pháp")
    
    print(f"\n Tổng hợp theo phương pháp:")
    for method, files in method_files.items():
        print(f"   - {METHOD_NAMES.get(method, method)}: {len(files)} file(s)")
    
    return method_files, output_path

def aggregate_stats(method_files, method_name):
    """Tổng hợp thống kê từ nhiều file (nhiều seed) của cùng một phương pháp"""
    all_stats = []
    
    for file_path in method_files:
        stats = load_and_analyze_csv(file_path, method_name)
        if stats:
            all_stats.append(stats)
    
    if not all_stats:
        return None
    
    aggregated = {
        'method': method_name,
        'mean': np.mean([s['mean'] for s in all_stats]),
        'median': np.mean([s['median'] for s in all_stats]),
        'std': np.mean([s['std'] for s in all_stats]),
        'min': np.min([s['min'] for s in all_stats]),
        'max': np.max([s['max'] for s in all_stats]),
        'q25': np.mean([s['q25'] for s in all_stats]),
        'q75': np.mean([s['q75'] for s in all_stats]),
        'count': sum([s['count'] for s in all_stats]),
        'data': [item for s in all_stats for item in s['data']]  
    }
    
    if 'compute_delay_data' in all_stats[0]:
        aggregated['compute_delay_mean'] = np.mean([s.get('compute_delay_mean', 0) for s in all_stats])
        aggregated['compute_delay_data'] = [item for s in all_stats if 'compute_delay_data' in s for item in s['compute_delay_data']]
    
    return aggregated

def plot_comparison(results_data, save_path, n_users=None):
    """Vẽ các biểu đồ so sánh"""
    
    if not results_data:
        print("Không có dữ liệu để vẽ biểu đồ!")
        return
    
    methods = list(results_data.keys())
    method_labels = [METHOD_NAMES.get(m, m) for m in methods]
    colors = [METHOD_COLORS.get(m, '#95A5A6') for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    means = [results_data[m]['mean'] for m in methods]
    medians = [results_data[m]['median'] for m in methods]
    stds = [results_data[m]['std'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, means, width, label='Mean', alpha=0.8, color=colors)
    bars2 = ax1.bar(x + width/2, medians, width, label='Median', alpha=0.8, color=[c.replace('1.0', '0.7') for c in colors])
    
    ax1.set_xlabel('Phương pháp', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Delay (giây)', fontsize=12, fontweight='bold')
    title = 'So sánh Mean và Median Total Delay'
    if n_users:
        title += f' (n_users={n_users})'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (m, med) in enumerate(zip(means, medians)):
        ax1.text(i - width/2, m, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, med, f'{med:.3f}', ha='center', va='bottom', fontsize=9)
    
    bars = ax2.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
    ax2.set_xlabel('Phương pháp', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Total Delay (giây)', fontsize=12, fontweight='bold')
    title2 = 'So sánh Mean Total Delay với Standard Deviation'
    if n_users:
        title2 += f' (n_users={n_users})'
    ax2.set_title(title2, fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (m, s) in enumerate(zip(means, stds)):
        ax2.text(i, m + s, f'{m:.3f}±{s:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    filename = 'comparison_mean_median_std.png'
    if n_users:
        filename = f'comparison_mean_median_std_{n_users}users.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    data_list = [results_data[m]['data'] for m in methods]
    
    bp = ax.boxplot(data_list, tick_labels=method_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Phương pháp', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Delay (giây)', fontsize=12, fontweight='bold')
    title3 = 'Phân bố Total Delay (Boxplot)'
    if n_users:
        title3 += f' (n_users={n_users})'
    ax.set_title(title3, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = 'comparison_boxplot.png'
    if n_users:
        filename = f'comparison_boxplot_{n_users}users.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
    if len(methods) == 1:
        axes = [axes]
    
    for idx, method in enumerate(methods):
        data = results_data[method]['data']
        axes[idx].hist(data, bins=30, alpha=0.7, color=colors[idx], edgecolor='black')
        axes[idx].axvline(results_data[method]['mean'], color='red', linestyle='--', 
                         linewidth=2, label=f"Mean: {results_data[method]['mean']:.3f}")
        axes[idx].axvline(results_data[method]['median'], color='blue', linestyle='--', 
                         linewidth=2, label=f"Median: {results_data[method]['median']:.3f}")
        axes[idx].set_xlabel('Total Delay (giây)', fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(METHOD_NAMES.get(method, method), fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = 'comparison_histograms.png'
    if n_users:
        filename = f'comparison_histograms_{n_users}users.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    SHORT_NAMES = {
        'random': 'Random',
        'drl_prediction_with_history_task_observation': 'DRL History',
        'drl_prediction': 'DRL',
        'fast_detect_outage': 'FDO'
    }

    num_methods = len(methods)
    fig_width = max(16, 14 + (num_methods - 3) * 0.5)  
    fig_height = max(6, 4 + num_methods * 0.5) 
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    
    table_data = []
    headers = ['Phương pháp', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q25', 'Q75', 'Count']
    
    for method in methods:
        stats = results_data[method]
        row = [
            SHORT_NAMES.get(method, METHOD_NAMES.get(method, method)), 
            f"{stats['mean']:.3f}",
            f"{stats['median']:.3f}",
            f"{stats['std']:.3f}",
            f"{stats['min']:.3f}",
            f"{stats['max']:.3f}",
            f"{stats['q25']:.3f}",
            f"{stats['q75']:.3f}",
            f"{stats['count']}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])  # Bảng chiếm toàn bộ figure
    
    table.auto_set_font_size(False)

    base_font_size = 9
    if num_methods > 3:
        base_font_size = 8
    
    table.set_fontsize(base_font_size)
    
    scale_x = min(1.0, 0.95 - (len(headers) - 8) * 0.02)
    scale_y = min(1.0, 0.9 - (num_methods - 3) * 0.05)
    table.scale(scale_x, scale_y)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    for i, method in enumerate(methods):
        color_hex = colors[i]
        if isinstance(color_hex, str) and color_hex.startswith('#'):
            r = int(color_hex[1:3], 16) / 255.0
            g = int(color_hex[3:5], 16) / 255.0
            b = int(color_hex[5:7], 16) / 255.0
            row_color = (r, g, b, 0.25)  
        else:
            row_color = color_hex
        
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(row_color)
            table[(i+1, j)].set_height(0.06)
    
    for i in range(len(headers)):
        if i == 0:  # Cột phương pháp
            table.auto_set_column_width([i])
        else:
            table.auto_set_column_width([i])
    
    title_table = 'Bảng Thống Kê So Sánh Total Delay'
    if n_users:
        title_table += f' (n_users={n_users})'

    fig.suptitle(title_table, fontsize=14, fontweight='bold', y=0.98)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    filename = 'comparison_statistics_table.png'
    if n_users:
        filename = f'comparison_statistics_table_{n_users}users.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"\nĐã tạo các biểu đồ so sánh trong thư mục: {save_path}")

def print_statistics(results_data):
    """In thống kê ra console"""
    print("\n" + "="*80)
    print("THỐNG KÊ SO SÁNH 3 PHƯƠNG PHÁP")
    print("="*80)
    
    for method, stats in results_data.items():
        method_name = METHOD_NAMES.get(method, method)
        print(f"\n{method_name}:")
        print(f"  Mean:   {stats['mean']:.4f} giây")
        print(f"  Median: {stats['median']:.4f} giây")
        print(f"  Std:    {stats['std']:.4f} giây")
        print(f"  Min:    {stats['min']:.4f} giây")
        print(f"  Max:    {stats['max']:.4f} giây")
        print(f"  Q25:    {stats['q25']:.4f} giây")
        print(f"  Q75:    {stats['q75']:.4f} giây")
        print(f"  Count:  {stats['count']} tasks")
    
    print("\n" + "="*80)
    print("SO SÁNH HIỆU QUẢ (Mean Total Delay - Càng thấp càng tốt):")
    print("="*80)
    
    sorted_methods = sorted(results_data.items(), key=lambda x: x[1]['mean'])
    for i, (method, stats) in enumerate(sorted_methods, 1):
        method_name = METHOD_NAMES.get(method, method)
        print(f"{i}. {method_name:30s}: {stats['mean']:.4f} giây")
    
    if len(sorted_methods) >= 2:
        best = sorted_methods[0][1]['mean']
        print("\nCải thiện so với Random:")
        for method, stats in sorted_methods[1:]:
            method_name = METHOD_NAMES.get(method, method)
            improvement = ((results_data['random']['mean'] - stats['mean']) / results_data['random']['mean']) * 100
            print(f"  {method_name:30s}: {improvement:+.2f}%")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='So sánh 3 phương pháp: Random, DRL, FDO')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Thư mục chứa các file CSV kết quả (ví dụ: output_file_3_service_8_users)')
    parser.add_argument('--n_users', type=int, default=None,
                       help='Số lượng users (optional, để tự động tìm thư mục)')
    parser.add_argument('--save_path', type=str, default='experiments/analysis_result',
                       help='Thư mục để lưu kết quả')
    
    args = parser.parse_args()
    
    method_files_dict, output_path = find_result_files(args.output_dir, args.n_users)
    
    if output_path is None:
        print("\n Không thể tìm thấy thư mục output!")
        print("Hãy kiểm tra lại đường dẫn hoặc chạy từ thư mục gốc của project.")
        return
    
    if not method_files_dict:
        print("\n Không tìm thấy file kết quả nào!")
        print(f"\nThư mục đang kiểm tra: {output_path}")
        print("\nCác file CSV có trong thư mục:")
        csv_files = list(output_path.glob('*.csv'))
        for f in csv_files[:20]:
            print(f"  - {f.name}")
        if len(csv_files) > 20:
            print(f"  ... và {len(csv_files) - 20} file khác")
        print("\nHãy đảm bảo có các file:")
        print("  - results_*random*.csv")
        print("  - results_*drl_prediction*.csv")
        print("  - results_*fast_detect_outage*.csv")
        return
    
    print(f"\nĐã tìm thấy các file:")
    for method, files in method_files_dict.items():
        print(f"  {METHOD_NAMES.get(method, method)}: {len(files)} file(s)")
        for f in files:
            print(f"    - {f.name}")
    
    results_data = {}
    
    target_methods = ['random', 'fast_detect_outage']
    
    if 'drl_prediction_with_history_task_observation' in method_files_dict:
        target_methods.append('drl_prediction_with_history_task_observation')
    elif 'drl_prediction' in method_files_dict:
        target_methods.append('drl_prediction')
    
    for method in target_methods:
        if method in method_files_dict:
            stats = aggregate_stats(method_files_dict[method], method)
            if stats:
                results_data[method] = stats
    
    if not results_data:
        print("Không có dữ liệu hợp lệ để so sánh!")
        return
    
    print_statistics(results_data)

    n_users = args.n_users
    if n_users is None:
        match = re.search(r'_(\d+)_users', str(output_path))
        if match:
            n_users = int(match.group(1))
    
    plot_comparison(results_data, args.save_path, n_users)
    
    print("\n" + "="*80)
    print("Hoàn thành!")
    print("="*80)

if __name__ == '__main__':
    main()

