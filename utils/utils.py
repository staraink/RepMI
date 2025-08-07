import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.linalg import fractional_matrix_power
from scipy.spatial.distance import cdist
from utils.channel_list import *
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch.nn as nn
from dataset import EEGDataset
from model.mlm import mlm_mask
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_missing_channels_diff(x, target_channels, actual_channels):
    B, C, T = x.shape
    num_target = len(target_channels)
    
    existing_pos = np.array([channel_positions[ch] for ch in actual_channels])

    target_pos = np.array([channel_positions[ch] for ch in target_channels])
    
    W = np.zeros((num_target, C))
    for i, (target_ch, pos) in enumerate(zip(target_channels, target_pos)):
        if target_ch in actual_channels:
            src_idx = actual_channels.index(target_ch)
            W[i, src_idx] = 1.0
        else:
            dist = cdist([pos], existing_pos)[0]
            weights = 1 / (dist + 1e-6)  
            weights /= weights.sum()     
            W[i] = weights
    
    padded = np.zeros((B, num_target, T))
    for b in range(B):
        padded[b] = W @ x[b]  
    
    return padded


def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5) 
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def process_and_replace_loader(loader,ischangechn,dataset):
    all_data = []
    all_labels = []
    for i in range(len(loader.dataset)):
        data, label = loader.dataset[i]
        all_data.append(data.numpy())
        all_labels.append(label)
    
    data_np = np.stack(all_data, axis=0)
    labels_tensor = torch.stack(all_labels)
    
    processed_data = EA(data_np).astype(np.float32)  

    if ischangechn:
        print("before processed：", processed_data.shape)
        if dataset == 'BNCI2014001':
            channels_names = BNCI2014001_chn_names
        elif dataset == 'BNCI2014004':
            channels_names = BNCI2014004_chn_names
        elif dataset == 'BNCI2014001-4':
            channels_names = BNCI2014001_chn_names
        elif dataset == 'AlexMI':
            channels_names = AlexMI_chn_names
        elif dataset =='BNCI2015001':
            channels_names = BNCI2015001_chn_names
        processed_data = pad_missing_channels_diff(processed_data,use_channels_names,channels_names)
        print("after processed：", processed_data.shape)
    new_dataset = TensorDataset(
        torch.from_numpy(processed_data).float(),  
        labels_tensor
    )
    
    loader_args = {
        'batch_size': loader.batch_size,
        'num_workers': loader.num_workers,
        'pin_memory': loader.pin_memory,
        'drop_last': loader.drop_last,
        'shuffle': isinstance(loader.sampler, torch.utils.data.RandomSampler)
    }
    
    return torch.utils.data.DataLoader(new_dataset, **loader_args)


def train(model, train_loader, criterion, optimizer, device, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        _, outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']

    epoch_loss = running_loss / len(train_loader)
    accuracy = correct / total * 100
    
    return epoch_loss, accuracy, current_lr

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            _, outputs = model(data)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    accuracy = correct / total * 100
    return epoch_loss, accuracy

def run_experiment(args, log_file):
    """Run complete experiment pipeline with configurable hyperparameters"""
    # Set up experiment tracking
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"./result/log/{args.dataset_name}_{args.model_name}_{now}_log.txt"
    csv_filename = f"./result/acc/{args.dataset_name}_{args.model_name}_{now}_results.csv"
    
    # Create log file handler
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    file_handler = open(log_filename, 'w')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    # Dataset configuration
    dataset_subjects = {
        'BNCI2014001': 9,
        'BNCI2015001': 12,
        'BNCI2014004': 9,
        'BNCI2014001-4': 9,
        'AlexMI': 8
    }
    args.sub_num = dataset_subjects.get(args.dataset_name, 0)
    
    # Seed iteration
    for seed_offset in range(args.num_exp):
        seed = seed_offset + 666
        set_seed(seed)
        subject_results = {}

        # Subject iteration
        for subject in range(args.sub_num):
            log_message = f"Starting Subject {subject}: Seed {seed}\n"
            log_file.write(log_message)
            file_handler.write(log_message)
            val_acc = train_subject(args, subject, seed, device, log_file)
            subject_results[subject] = val_acc

        results.append([seed] + list(subject_results.values()))
    
    save_results(results, args.sub_num, csv_filename)
    file_handler.close()

def train_subject(args, subject, seed, device, log_file):
    """Train and validate model for single subject with configurable hyperparams"""
    # Prepare dataset
    args.sub = [subject]
    dataset = EEGDataset(args=args)
    train_data, val_data = train_test_split(dataset, test_size=args.val_split, random_state=seed)
    
    # Configure data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Preprocess data
    train_loader = process_and_replace_loader(
        train_loader, 
        ischangechn=True, 
        dataset=args.dataset_name
    )
    val_loader = process_and_replace_loader(
        val_loader, 
        ischangechn=True, 
        dataset=args.dataset_name
    )
    
    # Initialize model
    model = mlm_mask(
        emb_size=args.emb_size,
        depth=args.depth,
        n_classes=args.num_classes,
        pretrainmode=False,
        pretrain=args.pretrain_path
    ).to(device)
    
    # Set up training components
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    else:
        scheduler = None
    
    final_val_acc = 0.0
    print(f"Seed: {seed}, Subject: {subject}\n")
    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        train_loss, train_acc, curr_lr = train(
            model, train_loader, criterion, 
            optimizer, device, scheduler
        )
        
        # Validation phase
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        final_val_acc = val_acc
        
        # Log epoch results
        log_file.write(
            f"Seed: {seed}, Subject: {subject}, Epoch: {epoch+1}\n"
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}, "
            f"LR: {curr_lr:.6f}\n"
        )
    
    
    return final_val_acc

def save_results(results, subject_count, filename):
    """Save experiment results to CSV file"""
    columns = ["Seed"] + [f"Subject_{i}_Acc" for i in range(subject_count)]
    results_df = pd.DataFrame(results, columns=columns)
    
    # Calculate summary statistics
    results_df['Seed_Avg_Acc'] = results_df.iloc[:, 1:].mean(axis=1)
    subject_avg = results_df.iloc[:, 1:-1].mean(axis=0)
    seed_avg = results_df['Seed_Avg_Acc'].mean()
    
    # Add summary row
    summary_row = ['Average'] + subject_avg.tolist() + [seed_avg]
    results_df.loc[len(results_df)] = summary_row
    
    # Save to CSV
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
