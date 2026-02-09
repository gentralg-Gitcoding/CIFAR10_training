'''Training functions for models.'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch(model, data_loader, criterion, optimizer, device, lazy_loading=False, cyclic_scheduler=None, history=None):
    '''Run one training epoch, tracking metrics per batch.
    
    Args:
        model: PyTorch model to train
        data_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device for training (e.g., 'cuda' or 'cpu')
        lazy_loading: If True, move batches to device during training. If False,
                     assumes data is already on device (default: False)
        cyclic_scheduler: Optional CyclicLR scheduler to step after each batch
        history: Optional history dictionary to record batch-level metrics
    
    Returns:
        Tuple of (average_loss, accuracy_percentage) for the epoch
    '''

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        if lazy_loading:
            images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Step cyclic scheduler after each batch
        if cyclic_scheduler is not None:
            cyclic_scheduler.step()
        
        # Track batch-level metrics
        batch_loss = loss.item()
        _, predicted = torch.max(outputs, 1)
        batch_correct = (predicted == labels).sum().item()
        batch_total = labels.size(0)
        batch_acc = 100 * batch_correct / batch_total
        
        # Record batch metrics if history is provided
        if history is not None:
            history['batch_train_loss'].append(batch_loss)
            history['batch_train_accuracy'].append(batch_acc)
            history['batch_learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            if cyclic_scheduler is not None:
                history['batch_base_lrs'].append(cyclic_scheduler.base_lrs[0])
                history['batch_max_lrs'].append(cyclic_scheduler.max_lrs[0])
        
        running_loss += batch_loss
        correct += batch_correct
        total += batch_total
    
    return running_loss / len(data_loader), 100 * correct / total


def evaluate(model, data_loader, criterion, device, lazy_loading=False):
    '''Evaluate model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: Data loader (validation or test set)
        criterion: Loss function
        device: Device for evaluation (e.g., 'cuda' or 'cpu')
        lazy_loading: If True, move batches to device during evaluation. If False,
                     assumes data is already on device (default: False)
    
    Returns:
        Tuple of (average_loss, accuracy_percentage) for the dataset
    '''

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            if lazy_loading:
                images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(data_loader), 100 * correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    criterion: nn.Module = None,
    optimizer: optim.Optimizer = None,
    device: torch.device = None,
    lazy_loading: bool = False,
    cyclic_scheduler = None,
    epoch_scheduler = None,
    lr_schedule: dict = None,
    epochs: int = 10,
    early_stopping_patience: int = 10,
    print_every: int = 1
) -> dict[str, list[float]]:
    '''Training loop with optional validation and early stopping.
    
    Tracks metrics at both batch and epoch levels.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader (None for training without validation)
        criterion: Loss function
        optimizer: Optimizer
        device: Device for training (e.g., 'cuda' or 'cpu')
        lazy_loading: If True, move batches to device during training. If False,
                     assumes data is already on device (default: False)
        cyclic_scheduler: CyclicLR scheduler (steps per batch)
        epoch_scheduler: Epoch-based scheduler like ReduceLROnPlateau (steps per epoch)
        lr_schedule: Optional dict with scheduled LR bounds reduction:
                    {'initial_base_lr', 'initial_max_lr', 'final_base_lr', 
                     'final_max_lr', 'schedule_epochs'}
        epochs: Maximum number of epochs
        early_stopping_patience: Stop if val_loss doesn't improve for this many epochs
                                (ignored if val_loader is None)
        print_every: Print progress every N epochs
    
    Returns:
        Dictionary containing training history with epoch and batch-level metrics
    '''

    history = {
        # Epoch-level metrics
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'learning_rates': [],
        'base_lrs': [],
        'max_lrs': [],
        
        # Batch-level metrics
        'batch_train_loss': [],
        'batch_train_accuracy': [],
        'batch_learning_rates': [],
        'batch_base_lrs': [],
        'batch_max_lrs': []
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):

        # Train and validate (now passing history to track batch metrics)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, lazy_loading, cyclic_scheduler, history)
        
        # Only evaluate on validation set if provided
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, lazy_loading)
        else:
            val_loss, val_acc = None, None
        
        # Record epoch-level metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss if val_loss is not None else float('nan'))
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc if val_acc is not None else float('nan'))
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Record base and max LR if using cyclic scheduler
        if cyclic_scheduler is not None:
            history['base_lrs'].append(cyclic_scheduler.base_lrs[0])
            history['max_lrs'].append(cyclic_scheduler.max_lrs[0])
        
        # Early stopping (only if validation data is provided)
        if val_loader is not None and val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1
        
        # Update LR bounds based on schedule
        if lr_schedule is not None and cyclic_scheduler is not None and epoch < lr_schedule['schedule_epochs']:
            # Linear interpolation of base and max LR
            progress = (epoch + 1) / lr_schedule['schedule_epochs']
            new_base_lr = lr_schedule['initial_base_lr'] * (1 - progress) + lr_schedule['final_base_lr'] * progress
            new_max_lr = lr_schedule['initial_max_lr'] * (1 - progress) + lr_schedule['final_max_lr'] * progress
            
            # Update the cyclic scheduler's base and max LRs
            cyclic_scheduler.base_lrs = [new_base_lr]
            cyclic_scheduler.max_lrs = [new_max_lr]
        
        # Step epoch-based scheduler (e.g., ReduceLROnPlateau)
        if epoch_scheduler is not None:
            if val_loader is not None and val_loss is not None:
                # ReduceLROnPlateau needs a metric
                if hasattr(epoch_scheduler, 'step') and 'metrics' in epoch_scheduler.step.__code__.co_varnames:
                    epoch_scheduler.step(val_loss)
                else:
                    epoch_scheduler.step()
            else:
                epoch_scheduler.step()
        
        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            base_lr = cyclic_scheduler.base_lrs[0] if cyclic_scheduler else lr
            max_lr = cyclic_scheduler.max_lrs[0] if cyclic_scheduler else lr
            
            if val_loader is not None:
                print(
                    f'Epoch {epoch+1:3d}/{epochs} - '
                    f'loss: {train_loss:.4f} - acc: {train_acc:5.2f}% - '
                    f'val_loss: {val_loss:.4f} - val_acc: {val_acc:5.2f}% - '
                    f'lr: {lr:.2e} (base: {base_lr:.2e}, max: {max_lr:.2e})'
                )
            else:
                print(
                    f'Epoch {epoch+1:3d}/{epochs} - '
                    f'loss: {train_loss:.4f} - acc: {train_acc:5.2f}% - '
                    f'lr: {lr:.2e} (base: {base_lr:.2e}, max: {max_lr:.2e})'
                )
        
        # Check early stopping (only if validation data is provided)
        if val_loader is not None and epochs_without_improvement >= early_stopping_patience:
            print(f'\nEarly stopping at epoch {epoch + 1}')
            print(f'Best val_loss: {best_val_loss:.4f} at epoch {epoch + 1 - epochs_without_improvement}')
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Restored best model weights')
    
    return history