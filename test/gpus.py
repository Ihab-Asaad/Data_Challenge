import torch
print("Cuda Available: ", torch.cuda.is_available())
print("Num of GPUs: ", torch.cuda.device_count())
print("Current device: " , torch.cuda.current_device())
print("Cuda name: ", torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device )
torch.cuda.init()
print("Initialize cuda: ", torch.cuda.is_initialized())
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print("Max memory reserved: ", torch.cuda.max_memory_reserved(device=device))
    print(torch.cuda.memory_summary(device=None, abbreviated=False)) # allows you to figure the reason of CUDA running out of memory and restart the kernel to avoid the error from happening again
    
# get real-time insight on used resources: will loop and call the view at every second.
# nvidia-smi -l 1 

# watch : 1 is the time interval
# watch -n1 nvidia-smi 

# gpustat -cp # from lib gpustat