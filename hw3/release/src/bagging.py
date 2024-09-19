import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import resample
from .utils import WeakClassifier, custom_binary_cross_entropy_with_logits


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """Implement your code here"""

        losses_of_models = []
        for model in self.learners:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            # indices = np.random.randint(len(X_train), size=len(X_train))
            # X_ = X_train[indices]
            # y_ = y_train[indices]
            X_, y_ = resample(X_train, y_train)
            model.train()
            for epoch in range(num_epochs):
                inputs = torch.tensor(X_, dtype=torch.float32)
                labels = torch.tensor(y_, dtype=torch.float32).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = custom_binary_cross_entropy_with_logits(outputs, labels)
                loss.backward()
                optimizer.step()
                if epoch%1000 == 0:
                    print(epoch, loss)
            losses_of_models.append(loss.item())
        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        class_outputs = []
        prob_outputs = []
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            for model in self.learners:
                model.eval()
                probs = torch.sigmoid(model(X_tensor).squeeze())
                classes = (probs > 0.5).int()
                prob_outputs.append(probs.cpu().numpy())
                class_outputs.append(classes)

        # 將所有分類器的結果堆疊起來
        prob_outputs_array = np.array(prob_outputs)
        avg_probs = np.mean(prob_outputs_array, axis=0)
        
        # 根據平均機率進行分類
        final_classes = (avg_probs > 0.5).astype(int)
        return final_classes, prob_outputs

    def compute_feature_importance(self) -> t.Sequence[float]:
        importance = np.zeros(self.learners[0].layer1.weight.shape[1])
        for model in self.learners:
            importance += model.layer1.weight.detach().abs().cpu().numpy().flatten()
        importance /= len(self.learners)
        return importance.tolist()
