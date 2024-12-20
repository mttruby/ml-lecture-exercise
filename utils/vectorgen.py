import torch

def synthetic_data(w, b, num_examples, noise_std=0.01, noise_type='normal'):  
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    
    if noise_type == 'normal':
        y += torch.normal(0, noise_std, y.shape)
    elif noise_type == 'uniform':
        y += torch.empty(y.shape).uniform_(-noise_std, noise_std)
    else:
        raise ValueError("Invalid noise_type. Supported types are 'normal' and 'uniform'.")
    
    return X, y.reshape((-1, 1))