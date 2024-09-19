import typing as t
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from torch.nn import Module, Linear


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layer model.
    No non-linear activation allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        self.in_features = input_dim  # Store input dimension
        self.layer1 = Linear(input_dim, 1)
    def forward(self, x):
        x = self.layer1(x)
        return x

def custom_binary_cross_entropy_with_logits(outputs, targets):
    # Calculate the sigmoid of the outputs (logits)    
    # Calculate the binary cross-entropy loss
    # max(x, 0) - x * z + log(1 + exp(-abs(x)))
    term1 = torch.clamp(outputs, min=0)
    term2 = outputs * targets
    term3 = torch.log(1 + torch.exp(-torch.abs(outputs)))
    
    loss = term1 - term2 + term3
    
    # Since we are manually implementing this, ensure to use mean reduction manually
    return torch.mean(loss)

def entropy_loss(outputs, targets, weights):
    weights = torch.tensor(weights, dtype=torch.float32)
    return torch.mean(weights * custom_binary_cross_entropy_with_logits(outputs.squeeze(), targets))

def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    plt.figure()
    for idx, y_pred in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_trues, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Learner {idx + 1} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve per Learner')
    plt.legend(loc='lower right')
    plt.savefig(fpath)
    plt.close()

def compute_accuracy(y_true, y_pred):
    correct_count = np.sum(y_true == y_pred)
    accuracy = correct_count / len(y_true)
    return accuracy

def plot_feature_importance(feature_importance, feature_names, fpath='./feature_importance.png'):
    # Sort feature importances and feature names
    sorted_idx = np.argsort(feature_importance)
    sorted_importance = np.array(feature_importance)[sorted_idx]
    sorted_feature_names = np.array(feature_names)[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_feature_names, sorted_importance, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()