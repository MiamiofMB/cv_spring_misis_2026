import statistics
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    return TensorDataset(X, y)

def train():
    dataloader = DataLoader(
        prepare_data(), 
        batch_size=256, 
        shuffle=True,  
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2
    )

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []
    
    warmup_sets = 5
    time_start_fwd = torch.cuda.Event(enable_timing=True)
    time_end_fwd = torch.cuda.Event(enable_timing=True)
    time_start_bwd = torch.cuda.Event(enable_timing=True)
    time_end_bwd = torch.cuda.Event(enable_timing=True)

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        noise = torch.randn_like(data, device='cuda')
        data = data + noise
        
        optimizer.zero_grad(set_to_none=True)  

        torch.cuda.synchronize()
        time_start_fwd.record()
        output = model(data)
        loss = criterion(output, target)
        time_end_fwd.record()
        torch.cuda.synchronize()
        fwd_time = time_start_fwd.elapsed_time(time_end_fwd) / 1000.0

        torch.cuda.synchronize()
        time_start_bwd.record()
        loss.backward()
        time_end_bwd.record()
        torch.cuda.synchronize()
        bwd_time = time_start_bwd.elapsed_time(time_end_bwd) / 1000.0

        if batch_idx > warmup_sets:
            backward_times.append(bwd_time) 
            forward_times.append(fwd_time) 
        
        optimizer.step()
        losses_history.append(loss.item())

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times):.6f}, "
          f"avg backward time is {statistics.mean(backward_times):.6f}")

if __name__ == '__main__':
    train()
