import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


'''
Что в сущности я сделал

1. Утечка в основном была из-за того, что в питонячьем массиве хранились не значения лоса, а полный loss 
объект, что приводило к тому, что весь вычислительный граф каждой эпохи оставался в памяти, что включается в себя все 
активации, а это очень много 
2. Не знаю что такое асинхронный конвейер, но в изначальном коде данные копировались на видеокарту не асинхронно - поправил
заранее выделив pinned секцию в памяти. Вычисления асинхронные тут, не знаю, что еще могло имется ввиду
3. Вместо таймера который замеряет тайминги ЦПУ, а в следствии просто постановку задач в очередь гпу вместо реальных таймингов forward 
и backward  поставил куда ивенты, синхронизировал до и после (синхронизация до нужна по причинам, описанным в коде).добавио burn in хотя наверное в такой
простой задаче он и не нужен, я эмпирически не выявил большой разницы, но по приколу все равно можно оставить. Вот наверное и все
'''


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    dataloader = DataLoader(prepare_data(), batch_size=256, shuffle=True,  pin_memory=True)

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
    time_start = torch.cuda.Event(enable_timing=True)
    time_end = torch.cuda.Event(enable_timing=True)
    time_start_bwd = torch.cuda.Event(enable_timing=True)
    time_end_bwd = torch.cuda.Event(enable_timing=True)

    for batch_idx, (data, target) in enumerate(dataloader):
        noise = torch.randn(data.shape).to('cuda', non_blocking=True)
        data = data.to('cuda', non_blocking=True) + noise
        target = target.to('cuda', non_blocking=True)
        
        
        optimizer.zero_grad()  

        # Синхронизация до нужна для того, чтобы состояние девайса было детерменировано. Я сам в шоке
        # но оказывается для честного замера метрики времени выполнения куда должна быть не разогретой
        # она может быть разогнонаной если до этого считала тяжелый forward и по сути замер времени без синхрона 
        # до стартовой метки может гулять из за этого 
        torch.cuda.synchronize()
        time_start.record()
        output = model(data)
        loss = criterion(output, target)
        time_end.record()
        torch.cuda.synchronize()
        fwd_time = time_start.elapsed_time(time_end) / 1000
        

        torch.cuda.synchronize()
        time_start_bwd.record()
        loss.backward()
        time_end_bwd.record()
        torch.cuda.synchronize()
        bwd_time = time_start_bwd.elapsed_time(time_end_bwd) / 1000

        if batch_idx > warmup_sets:
          backward_times.append(bwd_time) 
          forward_times.append(fwd_time) 
        
        
        optimizer.step()

        losses_history.append(loss.item())
        print(f"Batch {batch_idx} loss: {loss.item():.4f}")
        # torch.cuda.empty_cache() <- вонючка 

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times)}, "
          f"avg backward time is {statistics.mean(backward_times)}")

if __name__ == '__main__':
    train()
