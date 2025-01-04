import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

def calculate_metrics(model, data_loader, device):
    """
    Calculate detailed metrics for model evaluation
    """
    model.eval()
    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predictions = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Calculate classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=classes, 
                                 output_dict=True)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Convert to pandas DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    conf_matrix_df = pd.DataFrame(conf_matrix, 
                                 index=classes,
                                 columns=classes)
    
    return {
        'classification_report': report_df,
        'confusion_matrix': conf_matrix_df
    }