import numpy as np, pandas as pd
import torch, copy
from torch import nn
from sklearn.metrics import roc_auc_score
from HIGGS_utils import find_optimal_asimov_threshold
from typing import Any
import logging

def train_loop(logger: logging.Logger, data: Any, model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, steps_per_epoch: int, device: torch.device, s_tot: float, b_tot: float, sys_uncertainty: float) -> tuple[list[float], list[float], list[float]]:
    losses = []
    aucs = []
    adss = []
    
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for train_step, (features, labels) in enumerate(data):
        if train_step == steps_per_epoch:
            break
        
        features = torch.from_numpy(copy.copy(features)).to(device)
        labels = torch.from_numpy(copy.copy(labels)).to(device)
        
        # Compute prediction and loss
        optimizer.zero_grad()
        outputs = model(features)
        
        outputs = torch.squeeze(outputs)
        loss = loss_fn(outputs, labels.float())
        losses.append(loss.detach().cpu().numpy())

        labels_numpy = labels.detach().cpu().numpy()
        outputs_numpy = outputs.detach().cpu().numpy()

        aucs.append(roc_auc_score(labels_numpy, outputs_numpy))
        adss.append(find_optimal_asimov_threshold(
            labels_numpy, outputs_numpy, 
            weight_s=s_tot, weight_b=b_tot, sys_uncertainty=sys_uncertainty, 
            b_min=1, ret_full=False))

        # Backpropagation
        loss.backward()
        optimizer.step()

        if train_step % 10000 == -1:
            loss = loss.item()
            print(f"loss: {loss:.5f}")
            print(f'Auc: {aucs[-1]:.5f}')
    
    ads_sigs = [val["max_significance"] for val in adss]
    
    print(f'Training average loss: {sum(losses)/len(losses):.5f}')
    print(f'Training average auc: {sum(aucs)/len(aucs):.5f}')
    print(f'Training average ADS: {sum(ads_sigs)/len(ads_sigs):.5f}')

    logger.info('Training average loss: %f', round(sum(losses)/len(losses), 5))
    logger.info('Training average auc: %f', round(sum(aucs)/len(aucs), 5))
    logger.info('Training average ADS: %f', round(sum(ads_sigs)/len(ads_sigs), 5))
        
    return losses, aucs, ads_sigs
            
def valid_loop(logger: logging.Logger, data: Any, model: torch.nn.Module, loss_fn: torch.nn.Module, validation_steps: int, device: torch.device, s_tot: float, b_tot: float, sys_uncertainty: float) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any]]:
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    sum_loss = 0
    sum_count = 0
    val_labels = []
    val_preds = []
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for valid_step, (features, labels) in enumerate(data):
            if valid_step == validation_steps:
                break
            
            features = torch.from_numpy(copy.copy(features)).to(device)
            labels = torch.from_numpy(copy.copy(labels)).to(device)
            
            outputs = model(features)
        
            outputs = torch.squeeze(outputs)
            loss = loss_fn(outputs, labels.float()).detach().cpu().numpy()
            sum_loss += loss
            sum_count += 1

            val_labels.extend(labels.detach().cpu().numpy())
            val_preds.extend(outputs.detach().cpu().numpy())
            
        avg_loss = sum_loss / max(sum_count, 1)
        auc = roc_auc_score(val_labels, val_preds)

        ads = find_optimal_asimov_threshold(
            val_labels, val_preds, 
            weight_s=s_tot, weight_b=b_tot, sys_uncertainty=sys_uncertainty, 
            b_min=1e-8, ret_full=True)
        
    print(f"Validation average loss: {avg_loss:.5f}")
    print(f'Validation auc: {auc:.5f}')
    print(f"Validation ADS: {ads["max_significance"]:.5f}")
    print(f"Validation ADS best threshold: {ads["optimal_threshold"]:.5f}")
    print(f"Validation ADS Signal Events: {ads["signal_events"]:.4f}")
    print(f"Validation ADS Background Events: {ads["background_events"]:.2f}")

    logger.info("Validation average loss: %f", round(avg_loss, 5))
    logger.info('Validation auc: %f', round(auc, 5))
    logger.info("Validation ADS: %f", round(ads["max_significance"], 5))
    logger.info("Validation ADS best threshold: %f", round(ads["optimal_threshold"], 5))
    logger.info("Validation ADS Signal Events: %f", round(ads["signal_events"], 4))
    logger.info("Validation ADS Background Events: %f", round(ads["background_events"], 2))
    
    return np.array(val_labels), np.array(val_preds), avg_loss, auc, ads

def get_prediction_train(data: Any, model: nn.Module, loss_fn: nn.modules.loss._Loss, device: torch.device, training_size: int) -> tuple[list[float], list[float]]:
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    ret_labels = []
    ret_preds = []
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for train_step, (features, labels) in enumerate(data):
            if train_step == training_size/5e4:
                break
            
            features = torch.from_numpy(copy.copy(features)).to(device)
            labels = torch.from_numpy(copy.copy(labels)).to(device)
            
            outputs = model(features)
        
            outputs = torch.squeeze(outputs)
            
            ret_labels.extend(labels.detach().cpu().numpy())
            ret_preds.extend(outputs.detach().cpu().numpy())
            
    loss = loss_fn(torch.from_numpy(copy.copy(np.array(ret_preds))).float(), torch.from_numpy(copy.copy(np.array(ret_labels))).float())
    auc = roc_auc_score(ret_labels, ret_preds)
        
    print(f'Train loss: {loss:.5f}')
    print(f'Train auc: {auc:.5f}')
    
    #ret_preds = nn.Sigmoid()(torch.from_numpy(copy.copy(np.array(ret_preds)))).detach().cpu().numpy()
    
    return ret_labels, ret_preds

def get_prediction(data: Any, model: nn.Module, loss_fn: nn.modules.loss._Loss, device: torch.device, s_tot: float, b_tot: float, sys_uncertainty: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    sum_loss = 0
    sum_count = 0
    val_labels = []
    val_preds = []
    val_inputs = []
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        features, labels = next(iter(data))
        
        features = torch.from_numpy(copy.copy(features)).to(device)
        labels = torch.from_numpy(copy.copy(labels)).to(device)
        
        outputs = model(features)

        outputs = torch.squeeze(outputs)
        loss = loss_fn(outputs, labels.float()).item()
        sum_loss += loss
        sum_count += 1

        val_labels.extend(labels.detach().cpu().numpy())
        val_preds.extend(outputs.detach().cpu().numpy())
        val_inputs.extend(features.detach().cpu().numpy())
            
        avg_loss = sum_loss / max(sum_count, 1)
        auc = roc_auc_score(val_labels, val_preds)
        ads = find_optimal_asimov_threshold(
            val_labels, val_preds, 
            weight_s=s_tot, weight_b=b_tot, sys_uncertainty=sys_uncertainty, 
            b_min=1e-8, num_thresholds=1001, ret_full=True)
        
    print(f"Validation loss: {avg_loss:.6f}")
    print(f'Validation auc: {auc:.5f}')
    print(f'Validation ads: {ads["max_significance"]:.5f}')
    
    #val_preds = nn.Sigmoid()(torch.from_numpy(copy.copy(np.array(val_preds)))).detach().cpu().numpy()
    
    return np.array(val_labels), np.array(val_inputs), np.array(val_preds), ads