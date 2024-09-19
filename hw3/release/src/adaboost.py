import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        """Implement your code here"""
        n_samples, n_features = X_train.shape
        self.sample_weights = np.full(n_samples, 1 / n_samples)

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        learn_num = 0
        for learner in self.learners:
            learner.train()
            learn_num +=1
            print(learn_num,"/",10)
            optimizer = optim.AdamW(learner.parameters(), lr=learning_rate)
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = learner(X_tensor).squeeze()
                loss = entropy_loss(outputs, y_tensor, self.sample_weights)
                loss.backward()
                optimizer.step()
                if epoch%1000 == 0:
                    print(epoch, loss)
                

            learner.eval()
            predictions = learner(X_tensor).squeeze().detach().numpy()
            prediction_signs = np.sign(predictions) 
            error_rate = np.mean((prediction_signs != y_train) * self.sample_weights)

            # Avoid division by zero and ensure error_rate is in (0, 1)
            error_rate = min(max(error_rate, 1e-10), 1 - 1e-10)
            alpha = 0.5 * np.log((1 - error_rate) / error_rate)
            self.alphas.append(alpha)

            # Update sample weights
            self.sample_weights *= np.exp(-alpha * y_train * prediction_signs)
            self.sample_weights /= np.sum(self.sample_weights)

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        final_output = np.zeros(X.shape[0])  
        probas = []

        for alpha, learner in zip(self.alphas, self.learners):
            output = learner(X_tensor).squeeze().detach().numpy()  
            #print(output)
            final_output += alpha * output
            probas.append(output)
        final_class = np.sign(final_output)  
        final_class[final_class<0]=0
        return final_class.tolist(), probas

    def compute_feature_importance(self) -> t.Sequence[float]:
        importances = np.zeros(self.learners[0].in_features)
        
        for alpha, learner in zip(self.alphas, self.learners):
            weights_first_layer = learner.layer1.weight.data.numpy()
            importances += np.abs(weights_first_layer.sum(axis=0)) * alpha
        
        normalized_importance = importances / np.sum(importances)
        return normalized_importance.tolist()