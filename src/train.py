"""
Training script for disease classification model.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import argparse

from data import get_dataloaders
from model import get_model
from evaluate import evaluate_model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100.*correct/total:.2f}%")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train(
    data_dir="data/raw",
    output_dir="models",
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-3,
    architecture="resnet50",
    device_name="auto",
    seed=42
):
    """
    Train disease classification model.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to save model
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        architecture: 'resnet50' or 'mobilenet_v2'
        device_name: 'cpu', 'cuda', or 'auto'
        seed: Random seed
    """
    # Set seed
    torch.manual_seed(seed)
    
    # Device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    
    print(f"Using device: {device}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, disease_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        image_size=224,
        augment=True
    )
    
    num_classes = len(disease_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {disease_names}\n")
    
    # Model
    print(f"Creating {architecture} model...")
    model = get_model(num_classes, architecture=architecture, pretrained=True)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_loss': None, 'test_acc': None
    }
    
    print("Starting training...\n")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n")
        
        # Validate
        val_loss, val_acc, _, _ = evaluate_model(
            model, val_loader, criterion, device
        )
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
        
        # Log history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, f"{architecture}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Best model saved to {model_path}\n")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_precision, test_recall = evaluate_model(
        model, test_loader, criterion, device
    )
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Save history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = {
        'architecture': architecture,
        'num_classes': num_classes,
        'class_names': disease_names,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'device': str(device),
        'seed': seed
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Training complete! Model saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train disease classification model")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Path to save model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--architecture", type=str, default="resnet50",
                        help="Model architecture (resnet50 or mobilenet_v2)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu, cuda, or auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        architecture=args.architecture,
        device_name=args.device,
        seed=args.seed
    )
