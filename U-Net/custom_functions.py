import torch
import torch.nn.utils.prune as prune
import copy
import time
import numpy as np

########################################### FUNCTIONS RELATED TO PRUNING ###########################################
# 1.1) Structured - Local - L1
def structured_local(model, amount):
    """Applies structured (filter-wise) pruning locally (layer-by-layer)."""
    model_copy = copy.deepcopy(model)
    for module in model_copy.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0) 
            prune.remove(module, "weight") # apply mask to model permanently (zero weights are still kept in model)
    return model_copy

# 1.2) Structured - Local - Random
def structured_local_random(model, amount):
    """Applies random structured (filter-wise) pruning locally (layer-by-layer)."""
    model_copy = copy.deepcopy(model)
    for module in model_copy.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.random_structured(module, name="weight", amount=amount, dim=0)
            prune.remove(module, "weight") 
    return model_copy

# 2.1) Unstructured - Local - L1
def unstructured_local(model, amount):
    """Applies unstructured pruning locally (layer-by-layer)."""
    model_copy = copy.deepcopy(model)
    for module in model_copy.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model_copy

# 2.2) Unstructured - Local - Random
def unstructured_local_random(model, amount):
    """Applies random unstructured pruning locally (layer-by-layer)."""
    model_copy = copy.deepcopy(model)
    for module in model_copy.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.random_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model_copy

# 2.3) Unstructured - Global
def unstructured_global(model, amount):
    """Applies unstructured pruning globally across layers."""
    model_copy = copy.deepcopy(model)
    parameters_to_prune = [(m, "weight") for m in model_copy.modules() if isinstance(m, torch.nn.Conv2d)]
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for m, _ in parameters_to_prune:
        prune.remove(m, "weight")
    return model_copy

# 2.4) Unstructured - Global - Random
def unstructured_global_random(model, amount):
    """Applies random unstructured pruning globally across layers."""
    model_copy = copy.deepcopy(model)
    parameters_to_prune = [(m, "weight") for m in model_copy.modules() if isinstance(m, torch.nn.Conv2d)]
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.RandomUnstructured, amount=amount)
    for m, _ in parameters_to_prune:
        prune.remove(m, "weight")
    return model_copy

####################################### FUNCTIONS FOR EVALUATION OF PRUNING AND PERFORMANCE ###################################################

# Function for model evaluation
def evaluate_model(model, test_loader, criterion, device):
    # set model to evaluation mode
    model.eval()

    # intialize performance vars and lists
    validation_loss = 0.0
    TP, FP, FN, TN = 0, 0, 0, 0
    inference_times = []
    
    with torch.no_grad(): # disable gradiant computation (increases efficiency and not needed for eval)
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            validation_loss += loss.item()
            
            # Model predictions
            predictions = torch.sigmoid(outputs) # squeeze predictions between [0,1]
            thresholded_preds = (predictions > 0.5).float() # pred > 0.5 => 1 & pred <= 0.5 => 0

            # Compute metrics of confusion matrix
            TP += (thresholded_preds * masks).sum().item()
            FP += (thresholded_preds * (1 - masks)).sum().item()
            FN += ((1 - thresholded_preds) * masks).sum().item()
            TN += ((1 - thresholded_preds) * (1 - masks)).sum().item()
    
    avg_loss = validation_loss / len(test_loader)
    avg_accuracy = (TP + TN) / (TP + TN + FP + FN)
    avg_iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    avg_dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    inference_time = sum(inference_times)
    
    return avg_loss, avg_accuracy, avg_iou, avg_dice, inference_time

# count weights and filters
def countWeightsAndFilters(model):
    """
    Counts the total and zero weights across all parameters in the model,
    and counts the total number of filters (output channels) and the number
    of filters consisting entirely of zeros across all Conv2d layers.
    """
    total_weights = 0
    zero_weights = 0
    total_filters = 0
    zero_filters = 0

    for module in model.modules():
        # Count weights and zero weights
        for param in module.parameters(recurse=False):
            if param is not None:
                total_weights += param.numel()  # Total number of elements in the parameter tensor
                zero_weights += torch.sum((param == 0).int()).item()  # Count of zero weights

        # Count filters and zero filters (for Conv2d layers)
        if isinstance(module, torch.nn.Conv2d):
            total_filters += module.out_channels  # Count all filters (output channels)
            weight_tensor = module.weight.data  # Shape: [out_channels, in_channels, kernel_height, kernel_width]
            for filter_weights in weight_tensor:  # Iterate through filters (output channels)
                if torch.all(filter_weights == 0):  # Check if the entire filter is zeros
                    zero_filters += 1

    return total_weights, zero_weights, total_filters, zero_filters



