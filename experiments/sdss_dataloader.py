import torch
import torch.nn as nn   
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, random_split

def prepare_dataloader(flux, y_reg, y_cls, 
                       val_test_split=0.4, batch_size=256, seed=42, val_fraction = 0.25):

    ''' 
    Preparing the tensor sets for flux and labels
    '''

    flux_tensor  = torch.FloatTensor(flux)    # rank 1 tensors
    y_reg_tensor = torch.FloatTensor(y_reg)
    y_cls_tensor = torch.LongTensor(y_cls)     # tensor with int values

    # Shuffle before splitting
    n    = len(flux_tensor)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    flux_tensor  = flux_tensor[perm]
    y_reg_tensor = y_reg_tensor[perm]
    y_cls_tensor = y_cls_tensor[perm]

    dataset = TensorDataset(flux_tensor, y_reg_tensor, y_cls_tensor)

    # Compute integer sizes
    n_val  = int(n * val_fraction)
    n_test = int(n * val_test_split) - n_val  # remainder goes to test
    n_train = n - n_val - n_test

    # splitting the datasets into train, val and test datasets

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, drop_last=False)

    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
    return train_loader, val_loader, test_loader

