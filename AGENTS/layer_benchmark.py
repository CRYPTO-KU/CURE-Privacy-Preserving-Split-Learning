import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import sys

class LayerTimer:
    def __init__(self):
        self.forward_times = []
        self.backward_times = []
        self.start_time = 0

    def forward_pre_hook(self, module, input):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def forward_hook(self, module, input, output):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        self.forward_times.append(end_time - self.start_time)

    def backward_pre_hook(self, module, grad_output):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def backward_hook(self, module, grad_input, grad_output):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        self.backward_times.append(end_time - self.start_time)

class DynamicModel(nn.Module):
    def __init__(self, layers_config):
        super(DynamicModel, self).__init__()
        self.layers = nn.ModuleList()
        self.timers = []
        
        for layer_def in layers_config:
            layer_type = layer_def['type']
            module = None
            
            if layer_type == 'Linear':
                module = nn.Linear(layer_def['in'], layer_def['out'])
            elif layer_type == 'Conv2D':
                module = nn.Conv2d(layer_def['in'], layer_def['out'], kernel_size=layer_def['k'])
            elif layer_type == 'Conv1D':
                module = nn.Conv1d(layer_def['in'], layer_def['out'], kernel_size=layer_def['k'])
            elif layer_type == 'MaxPool2D':
                module = nn.MaxPool2d(kernel_size=layer_def['k'])
            elif layer_type == 'MaxPool1D':
                module = nn.MaxPool1d(kernel_size=layer_def['k'])
            elif layer_type == 'Flatten':
                module = nn.Flatten()
            elif layer_type == 'ReLU':
                module = nn.ReLU()
            
            if module:
                self.layers.append(module)
                
                # Register timer for "weight" layers or layers of interest
                # User asked for "layer 1 input X time spent..."
                # Usually we care about Conv and Linear. 
                # Let's time Conv, Linear, and MaxPool (as it does computation).
                # Flatten and ReLU are usually negligible or fused, but we can time them if needed.
                # Based on "784 128 32 10" example, user cares about the weight layers.
                # But for LeNet, pooling is significant.
                # Let's time everything that is not ReLU/Flatten for now, or maybe everything?
                # User example: "layer 1 input 784... layer 2 input 128..."
                # This suggests counting "layers" as the major blocks.
                # I will time Conv, Linear, MaxPool.
                
                if layer_type in ['Linear', 'Conv2D', 'Conv1D', 'MaxPool2D', 'MaxPool1D']:
                    timer = LayerTimer()
                    module.register_forward_pre_hook(timer.forward_pre_hook)
                    module.register_forward_hook(timer.forward_hook)
                    module.register_full_backward_pre_hook(timer.backward_pre_hook)
                    module.register_full_backward_hook(timer.backward_hook)
                    self.timers.append((layer_type, timer))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def get_model_config(model_name):
    if model_name == 'mnistfc':
        # 784 -> 128 -> 32 -> 10
        return [
            {'type': 'Linear', 'in': 784, 'out': 128},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in': 128, 'out': 32},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in': 32, 'out': 10},
        ], (1, 784)
        
    elif model_name == 'bcwfc':
        # 64 -> 32 -> 16 -> 10
        return [
            {'type': 'Linear', 'in': 64, 'out': 32},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in': 32, 'out': 16},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in': 16, 'out': 10},
        ], (1, 64)
        
    elif model_name == 'lenet':
        # Input: (1, 1, 28, 28)
        # Conv2D(1, 6, 5) -> (1, 6, 24, 24)
        # MaxPool(2) -> (1, 6, 12, 12)
        # Conv2D(6, 16, 5) -> (1, 16, 8, 8)
        # MaxPool(2) -> (1, 16, 4, 4)
        # Flatten -> 256
        # Linear(256, 120)
        # Linear(120, 84)
        # Linear(84, 10)
        return [
            {'type': 'Conv2D', 'in': 1, 'out': 6, 'k': 5},
            {'type': 'ReLU'},
            {'type': 'MaxPool2D', 'k': 2},
            {'type': 'Conv2D', 'in': 6, 'out': 16, 'k': 5},
            {'type': 'ReLU'},
            {'type': 'MaxPool2D', 'k': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'in': 16*4*4, 'out': 120},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in': 120, 'out': 84},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in': 84, 'out': 10},
        ], (1, 1, 28, 28)
        
    elif model_name == 'audio1d':
        # Input: (1, 12, 1006)
        # Conv1D(12, 16, 3) -> (1, 16, 1004)
        # MaxPool(2) -> (1, 16, 502)
        # Conv1D(16, 8, 3) -> (1, 8, 500)
        # MaxPool(2) -> (1, 8, 250)
        # Flatten -> 8*250 = 2000
        # Linear(2000, 5)
        return [
            {'type': 'Conv1D', 'in': 12, 'out': 16, 'k': 3},
            {'type': 'ReLU'},
            {'type': 'MaxPool1D', 'k': 2},
            {'type': 'Conv1D', 'in': 16, 'out': 8, 'k': 3},
            {'type': 'ReLU'},
            {'type': 'MaxPool1D', 'k': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'in': 2000, 'out': 5},
        ], (1, 12, 1006)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")

def benchmark_model(model_name, dataset_size=60000, num_iterations=100, warmup=10):
    print(f"\nBenchmarking Model: {model_name}")
    
    config, input_shape = get_model_config(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    model = DynamicModel(config).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Determine output dim for loss
    last_layer = config[-1]
    output_dim = last_layer['out']
    
    criterion = nn.CrossEntropyLoss()
    
    x = torch.randn(*input_shape).to(device)
    target = torch.randint(0, output_dim, (1,)).to(device)
    
    # Warmup
    model.train()
    for _ in range(warmup):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    # Clear timers
    for _, timer in model.timers:
        timer.forward_times = []
        timer.backward_times = []
        
    # Benchmark
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    # Process Results
    print(f"{'Layer Type':<15} | {'Input Shape':<20} | {'Time/Epoch (ms)':<20} | {'Forward (avg ms)':<20} | {'Backward (avg ms)':<20}")
    print("-" * 100)
    
    total_epoch_time = 0
    
    # We need to track input shapes. We can do a dummy forward pass with hooks to capture shapes?
    # Or just calculate manually?
    # Let's do a dummy pass to capture input shapes for printing.
    
    layer_input_shapes = []
    def shape_hook(module, input):
        layer_input_shapes.append(list(input[0].shape))
        
    # Register shape hooks on timed layers
    hooks = []
    for _, timer in model.timers:
        # We need to find the module associated with this timer. 
        # But we didn't store the module in the timer list, just the type and timer.
        # We can re-iterate model.layers and match? No, that's messy.
        pass
        
    # Simpler: Just run one pass and print shapes? 
    # Or just print the "Fan In" as requested.
    # User asked: "layer 1 input 784... layer 2 input 128..."
    # For Conv, Fan In is usually Channels * Kernel? Or just Input Volume?
    # "fan in value of that layer (input dim)"
    # For Linear: Input Features.
    # For Conv: Input Channels * Height * Width? Or just Input Channels?
    # Let's print the full input shape.
    
    # Re-run one pass to capture shapes
    layer_shapes = {}
    
    def get_shape_hook(idx):
        def hook(module, input):
            layer_shapes[idx] = list(input[0].shape)
        return hook

    hook_handles = []
    timer_idx = 0
    for layer in model.layers:
        if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.MaxPool2d, nn.MaxPool1d)):
            h = layer.register_forward_pre_hook(get_shape_hook(timer_idx))
            hook_handles.append(h)
            timer_idx += 1
            
    with torch.no_grad():
        model(x)
        
    for h in hook_handles:
        h.remove()

    for i, (l_type, timer) in enumerate(model.timers):
        avg_fwd = (sum(timer.forward_times) / len(timer.forward_times)) * 1000
        avg_bwd = (sum(timer.backward_times) / len(timer.backward_times)) * 1000
        
        total_sample_ms = avg_fwd + avg_bwd
        total_epoch_ms = total_sample_ms * dataset_size
        
        in_shape = str(layer_shapes.get(i, "?"))
        
        print(f"{l_type:<15} | {in_shape:<20} | {total_epoch_ms:<20.2f} | {avg_fwd:<20.4f} | {avg_bwd:<20.4f}")
        
        total_epoch_time += total_epoch_ms

    print("-" * 100)
    print(f"Total Extrapolated Time per Epoch: {total_epoch_time/1000:.2f} s")

if __name__ == "__main__":
    models_to_run = ['mnistfc', 'bcwfc', 'lenet', 'audio1d']
    
    for m in models_to_run:
        try:
            benchmark_model(m)
        except Exception as e:
            print(f"Failed to run {m}: {e}")
