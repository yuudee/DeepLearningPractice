# -*- coding: utf-8 -*-
"""
Model save/load utility functions
"""
import os
import torch
from datetime import datetime


def create_save_directory(base_dir='./models'):
    """
    Create a directory for saving models with timestamp

    Args:
        base_dir (str): Base directory for models (default: './models')

    Returns:
        str: Path to the created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_model(model, save_path, epoch=None, optimizer=None, loss=None, accuracy=None):
    """
    Save model checkpoint

    Args:
        model (nn.Module): Model to save
        save_path (str): Path to save the model
        epoch (int): Current epoch number (optional)
        optimizer: Optimizer state (optional)
        loss (float): Current loss value (optional)
        accuracy (float): Current accuracy (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if loss is not None:
        checkpoint['loss'] = loss
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy

    torch.save(checkpoint, save_path)
    print(f"Model saved: {save_path}")


def save_epoch_model(model, save_dir, epoch, optimizer=None, train_loss=None, test_loss=None, accuracy=None):
    """
    Save model for a specific epoch

    Args:
        model (nn.Module): Model to save
        save_dir (str): Directory to save the model
        epoch (int): Epoch number
        optimizer: Optimizer state (optional)
        train_loss (float): Training loss (optional)
        test_loss (float): Test loss (optional)
        accuracy (float): Test accuracy (optional)

    Returns:
        str: Path to the saved model
    """
    filename = f"model_epoch_{epoch:03d}.pth"
    save_path = os.path.join(save_dir, filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'epoch': epoch,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if train_loss is not None:
        checkpoint['train_loss'] = train_loss
    if test_loss is not None:
        checkpoint['test_loss'] = test_loss
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy

    torch.save(checkpoint, save_path)
    return save_path


def save_best_model(model, save_dir, epoch, accuracy, optimizer=None):
    """
    Save the best model

    Args:
        model (nn.Module): Model to save
        save_dir (str): Directory to save the model
        epoch (int): Epoch number
        accuracy (float): Test accuracy
        optimizer: Optimizer state (optional)

    Returns:
        str: Path to the saved model
    """
    filename = "best_model.pth"
    save_path = os.path.join(save_dir, filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'epoch': epoch,
        'accuracy': accuracy,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, save_path)
    print(f"Best model saved: {save_path} (accuracy: {accuracy:.2f}%)")
    return save_path


def load_model(model, load_path, device=None):
    """
    Load model from checkpoint

    Args:
        model (nn.Module): Model instance to load weights into
        load_path (str): Path to the checkpoint file
        device (torch.device): Device to load the model on (optional)

    Returns:
        dict: Checkpoint dictionary containing model info
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {load_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'accuracy' in checkpoint:
        print(f"  Accuracy: {checkpoint['accuracy']:.2f}%")
    if 'train_loss' in checkpoint:
        print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
    if 'test_loss' in checkpoint:
        print(f"  Test Loss: {checkpoint['test_loss']:.4f}")

    return checkpoint


def load_model_for_inference(model_class, load_path, device=None, **model_kwargs):
    """
    Load model for inference (creates new model instance)

    Args:
        model_class: Model class (e.g., SimpleCNN)
        load_path (str): Path to the checkpoint file
        device (torch.device): Device to load the model on (optional)
        **model_kwargs: Arguments for model initialization

    Returns:
        tuple: (model, checkpoint) - Loaded model and checkpoint info
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_class(**model_kwargs)
    checkpoint = load_model(model, load_path, device)

    return model, checkpoint


def list_saved_models(save_dir):
    """
    List all saved models in a directory

    Args:
        save_dir (str): Directory containing saved models

    Returns:
        list: List of model file paths
    """
    if not os.path.exists(save_dir):
        print(f"Directory not found: {save_dir}")
        return []

    model_files = []
    for filename in sorted(os.listdir(save_dir)):
        if filename.endswith('.pth'):
            model_files.append(os.path.join(save_dir, filename))

    return model_files


def get_latest_model_dir(base_dir='./models'):
    """
    Get the latest model directory

    Args:
        base_dir (str): Base directory for models

    Returns:
        str: Path to the latest model directory, or None if not found
    """
    if not os.path.exists(base_dir):
        return None

    dirs = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            dirs.append(path)

    if not dirs:
        return None

    return max(dirs, key=os.path.getmtime)

