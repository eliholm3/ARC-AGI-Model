import torch

def report_param_stats(model, name="", max_layers=10):
    print(f"\n========= PARAM REPORT: {name} =========")
    for i, (n, p) in enumerate(model.named_parameters()):
        if i >= max_layers:
            print("... (truncated)")
            break
        if p.grad is None:
            print(f"{n}: grad=None | weight mean={p.data.mean().item():.4f} std={p.data.std().item():.4f}")
        else:
            print(f"{n}: grad mean={p.grad.mean().item():.4f} | grad std={p.grad.std().item():.4f} | "
                  f"weight mean={p.data.mean().item():.4f} std={p.data.std().item():.4f}")