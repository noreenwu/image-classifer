import torch
    
def check_device(gpu):
    
    if gpu:
        specified_device = "cuda"
    else:
        specified_device = "cpu"


    if specified_device == "cuda":
        if not torch.cuda.is_available():
            print("GPU specified but not available. Sorry")
            exit(1)
            
            
    return specified_device