import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

def find_file(relative_paths):
    for path in relative_paths:
        if os.path.exists(path):
            return path
    return None

def parse_training_log(log_path):
    epochs = []
    ap_50_95 = []
    ap_50 = []
    
    if not log_path or not os.path.exists(log_path):
        print(f"Warning: {log_path} not found.")
        return epochs, ap_50_95, ap_50

    print(f"Parsing log: {log_path}")
    with open(log_path, 'r') as f:
        for line in f:
            # 1. JSON-formatted line
            if '"coco_eval_bbox"' in line:
                try:
                    data = json.loads(line)
                    epochs.append(int(data.get('epoch', len(epochs) + 1)))
                    ap_50_95.append(float(data['coco_eval_bbox'][0]) * 100)
                    ap_50.append(float(data['coco_eval_bbox'][1]) * 100)
                    continue
                except:
                    pass
            
            # 2. Text log formatted line
            if 'coco_eval_bbox' in line:
                match = re.search(r"coco_eval_bbox':\s*\[(.*?)\]", line)
                ep_match = re.search(r"epoch':\s*(\d+)", line)
                if match:
                    vals = [float(x.strip()) for x in match.group(1).split(',') if x.strip()]
                    if len(vals) >= 2:
                        ep = int(ep_match.group(1)) if ep_match else len(epochs) + 1
                        epochs.append(ep)
                        ap_50_95.append(vals[0] * 100)
                        ap_50.append(vals[1] * 100)

    # Remove duplicates if any
    unique_data = {}
    for ep, ap, ap50 in zip(epochs, ap_50_95, ap_50):
        unique_data[ep] = (ap, ap50)
    
    sorted_eps = sorted(unique_data.keys())
    sorted_ap = [unique_data[k][0] for k in sorted_eps]
    sorted_ap50 = [unique_data[k][1] for k in sorted_eps]
    
    return sorted_eps, sorted_ap, sorted_ap50

def smooth(scalars, weight=0.55):
    if len(scalars) == 0:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# File paths checking
path_proposed = find_file([
    'output_hfs_fgfe_token_asr/training.log',
    '../output_hfs_fgfe_token_asr/training.log'
])
path_baseline = find_file([
    'output_position_detr/training.log',
    '../output_position_detr/training.log'
])

ep_p, ap_p, ap50_p = parse_training_log(path_proposed)
ep_b, ap_b, ap50_b = parse_training_log(path_baseline)

print(f"Proposed epochs found: {len(ep_p)} (Max AP: {max(ap_p) if ap_p else 0:.2f}%)")
print(f"Baseline epochs found: {len(ep_b)} (Max AP: {max(ap_b) if ap_b else 0:.2f}%)")

# Plotting
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Left Plot: AP@50:95
if ep_b:
    ax1.plot(ep_b, ap_b, color='#ef4444', alpha=0.3, linestyle='--', marker='o', markersize=3.5)
    ax1.plot(ep_b, smooth(ap_b), color='#dc2626', linewidth=2.4, label='Baseline (Position-DETR)')
if ep_p:
    ax1.plot(ep_p, ap_p, color='#3b82f6', alpha=0.3, linestyle='--', marker='o', markersize=3.5)
    ax1.plot(ep_p, smooth(ap_p), color='#2563eb', linewidth=2.4, label='Proposed (Token-ASR)')

ax1.set_xlabel('Epochs', fontsize=13, fontweight='bold')
ax1.set_ylabel('AP@50:95 (%)', fontsize=13, fontweight='bold')
ax1.set_title('Overall mAP Convergence (VisDrone)', fontsize=14, fontweight='bold', pad=10)
ax1.tick_params(axis='both', labelsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right', fontsize=11, frameon=True)

# Right Plot: AP@50
if ep_b:
    ax2.plot(ep_b, ap50_b, color='#ef4444', alpha=0.3, linestyle='--', marker='o', markersize=3.5)
    ax2.plot(ep_b, smooth(ap50_b), color='#dc2626', linewidth=2.4, label='Baseline (Position-DETR)')
if ep_p:
    ax2.plot(ep_p, ap50_p, color='#3b82f6', alpha=0.3, linestyle='--', marker='o', markersize=3.5)
    ax2.plot(ep_p, smooth(ap50_p), color='#2563eb', linewidth=2.4, label='Proposed (Token-ASR)')

ax2.set_xlabel('Epochs', fontsize=13, fontweight='bold')
ax2.set_ylabel('AP@50 (%)', fontsize=13, fontweight='bold')
ax2.set_title('AP50 Convergence (VisDrone)', fontsize=14, fontweight='bold', pad=10)
ax2.tick_params(axis='both', labelsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower right', fontsize=11, frameon=True)

plt.tight_layout()
output_filename = 'training_curves_comparison.png'
plt.savefig(output_filename, bbox_inches='tight')
print(f"\nTraining curve graph successfully saved to: {output_filename}")
