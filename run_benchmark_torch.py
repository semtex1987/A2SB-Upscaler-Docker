import torch
import time

def run_benchmark(n_steps=1000, batch_size=1):
    shape = (batch_size, 3, 256, 256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Warmup
    for _ in range(10):
        t = torch.randn(*shape, device=device)
        t.cpu()

    start_time = time.time()
    all_pred_x0s = []
    for _ in range(n_steps):
        pred_x0 = torch.randn(*shape, device=device)
        all_pred_x0s.append(pred_x0.cpu())
    if device == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    append_time = end_time - start_time
    print(f"List append: {append_time:.6f}s")

    start_time = time.time()
    all_pred_x0s_pre = [None] * n_steps
    for i in range(n_steps):
        pred_x0 = torch.randn(*shape, device=device)
        all_pred_x0s_pre[i] = pred_x0.cpu()
    if device == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    prelist_time = end_time - start_time
    print(f"Pre-allocated list: {prelist_time:.6f}s")

    start_time = time.time()
    all_pred_x0s_tensor = torch.empty(n_steps, *shape, device='cpu')
    for i in range(n_steps):
        pred_x0 = torch.randn(*shape, device=device)
        all_pred_x0s_tensor[i] = pred_x0.cpu()
    if device == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    pretensor_time = end_time - start_time
    print(f"Pre-allocated tensor (assignment): {pretensor_time:.6f}s")

    return append_time, prelist_time

print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
for steps in [50, 100, 200, 500, 1000]:
    print(f"--- Steps: {steps} ---")
    run_benchmark(steps)
