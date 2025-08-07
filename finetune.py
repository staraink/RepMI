from datetime import datetime
import argparse
from utils.utils import *

def parse_args():
    """Parse command line arguments with all hyperparameters"""
    parser = argparse.ArgumentParser(description='EEG Classification with Configurable Hyperparameters')
    
    # Experiment configuration
    parser.add_argument('--dataset_name', default='BNCI2014004', help='EEG dataset name')
    parser.add_argument('--model_name', default='MIRepNet', help='Model identifier')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    
    # Optimizer selection
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam', 
                       help='Optimizer to use (adam or sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, 
                       help='Momentum for SGD optimizer')
    
    # Scheduler configuration
    parser.add_argument('--scheduler', choices=['cosine', 'step', 'none'], 
                       default='cosine', help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30, 
                       help='Step size for step LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, 
                       help='Gamma for step LR scheduler')
    
    # Model architecture
    parser.add_argument('--emb_size', type=int, default=256, 
                       help='Embedding size for the model')
    parser.add_argument('--depth', type=int, default=6, 
                       help='Number of transformer layers')
    parser.add_argument('--num_classes', type=int, default=2, 
                       help='Number of output classes')
    
    # Data loading
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of workers for data loading')
    parser.add_argument('--val_split', type=float, default=0.7, 
                       help='Validation set split ratio')
    
    # Experiment repetition
    parser.add_argument('--num_exp', type=int, default=1, 
                       help='Number of experiment repetitions')
    
    # Pretrained weights
    parser.add_argument('--pretrain_path', default='./weight/MIRepNet.pth', 
                       help='Path to pretrained model weights')
    
    return parser.parse_args()

if __name__ == '__main__':
    print("Starting EEG Classification with Configurable Hyperparameters\n")
    args = parse_args()
    
    # Initialize empty channel names list
    args.datasetchnname = []  
    args.sub_num = 0  # Will be set in run_experiment
    
    # Initialize logging
    experiment_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = open(
        f"./result/log/{args.dataset_name}_{args.model_name}_{experiment_time}_log.txt", 
        'w'
    )
    
    try:
        run_experiment(args, log_file)
    finally:
        log_file.close()
    
    print("Experiment completed successfully")
