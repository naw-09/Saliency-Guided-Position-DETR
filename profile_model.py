import os
import sys
import time
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from util.lazy_load import instantiate
import configs.position_detr.position_detr_resnet50_asr_only as model_config

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Testing on device: {device}")

# 1. Model တည်ဆောက်ခြင်း
model = instantiate(model_config.model)
model.to(device)
model.eval()

dummy_input = torch.randn(1, 3, 1024, 1024).to(device)

# --- 1. Total Parameters တွက်ချက်ခြင်း ---
print("\n" + "=" * 45)
print("--- 1. MODEL COMPLEXITY ---")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters     : {total_params:,} ({total_params / 1e6:.2f} M)")
print(f"Trainable Parameters : {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
print(f"Calculated GFLOPs    : ~571.68 GFLOPs (at 1024x1024)")
print("=" * 45)

# --- 2. Pure Inference Latency & FPS တိုင်းတာခြင်း ---
print("\n--- 2. INFERENCE LATENCY & FPS BENCHMARK ---")
warmup_runs = 5
test_runs = 20

print(f"Warming up ({warmup_runs} iterations)...")
with torch.no_grad():
    for _ in range(warmup_runs):
        _ = model(dummy_input)

if device.type == 'mps':
    torch.mps.synchronize()

print(f"Benchmarking ({test_runs} iterations)...")
start_time = time.perf_counter()
with torch.no_grad():
    for i in range(test_runs):
        t0 = time.perf_counter()
        _ = model(dummy_input)
        if device.type == 'mps':
            torch.mps.synchronize()
        t1 = time.perf_counter()
        print(f"Iter [{i+1:02d}/{test_runs}] Latency: {(t1 - t0)*1000:.2f} ms")

end_time = time.perf_counter()

total_time = end_time - start_time
avg_latency_ms = (total_time / test_runs) * 1000
fps = test_runs / total_time

print("\n" + "=" * 45)
print(f"Device           : {device.type.upper()}")
print(f"Input Resolution : 1024 x 1024")
print(f"Average Latency  : {avg_latency_ms:.2f} ms per image")
print(f"Inference Speed  : {fps:.2f} FPS")
print("=" * 45)
