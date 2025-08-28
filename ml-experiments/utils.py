# data_utils.py
import time, json
import random
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Experiments data loaders
def get_loaders(exp, batch_size=128, num_workers=4, pin_memory=False, dataset_for_vit="cifar10"):
    if exp == "mnist_cnn":
        tr = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST("./data", train=True, download=True, transform=tr)
        test  = datasets.MNIST("./data", train=False, transform=te)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader
    elif exp == "cifar10_resnet32":
        mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
        tr = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train = datasets.CIFAR10("./data", train=True, download=True, transform=tr)
        test  = datasets.CIFAR10("./data", train=False, transform=te)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader
    elif exp == "cifar100_resnet56":
        mean = (0.5071, 0.4867, 0.4408); std = (0.2675, 0.2565, 0.2761)
        tr = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train = datasets.CIFAR100("./data", train=True, download=True, transform=tr)
        test  = datasets.CIFAR100("./data", train=False, transform=te)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader
    else:
        raise ValueError("Unknown experiment key")


def train_epoch(model, loader, optimizer, criterion, device, show_progress=False, desc=""):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    batch_losses, batch_evals = [], []

    iterator = tqdm(loader, desc=desc) if show_progress else loader

    for data, target in iterator:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss_tensor = criterion(outputs, target)
        loss_tensor.backward()
        optimizer.step()
        batch_loss = float(loss_tensor.item())
        preds = outputs.argmax(dim=1)
        weighted_evals = 1.0  # 1 eval per batch w/ our convention

        batch_size = data.size(0)
        total_loss += batch_loss * batch_size
        correct += preds.eq(target).sum().item()
        n += batch_size
        batch_losses.append(batch_loss)
        batch_evals.append(weighted_evals)

    return total_loss / n, correct / n, batch_losses, batch_evals


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        total_loss += loss.item() * data.size(0)
        correct += outputs.argmax(dim=1).eq(target).sum().item()
        n += data.size(0)
    return total_loss / n, correct / n



