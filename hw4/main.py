import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from dataloader import CustomDataset
from model import BagClassifier

class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_acc = 0

        train_dataset = CustomDataset('released/train')
        val_dataset = CustomDataset('released/val')
        test_dataset = CustomDataset('released/test')
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,num_workers=8)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=8)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=8)

        self.model = BagClassifier()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'best_net.pth')))

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.args.epochs):
            self.model.train()

            # training
            self.train_one_epoch(epoch)
            
            # testing 
            acc = self.val(epoch)

            # save model
            if(acc >= self.best_acc):
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, "best_net.pth"))
                print("------ save best net ------")
            if (epoch+1) % 10 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, f"epoch_{epoch+1}_.pth"))

    def train_one_epoch(self, epoch):
        total_loss = 0
        num_samples = 0
        pbar = tqdm(total=len(self.train_loader), ncols=80)
        for idx, (img, label) in enumerate(self.train_loader):
            img, label = img.to(self.device), label.to(self.device).float()
            
            # Get prediction
            outputs = self.model(img)
            loss = self.criterion(outputs.view(-1), label)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.detach().item()
            num_samples += 1
            avg_loss = total_loss / num_samples

            pbar.set_description(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
            pbar.update(1)

        pbar.close()    
        print(f'Train Epoch_{epoch} loss: {total_loss / len(self.train_loader) :.5f}\n')

    def val(self, epoch = None):
        self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        pbar = tqdm(total=len(self.val_loader), ncols=80)
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1), labels)
                
                preds = torch.sigmoid(outputs).view(-1) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.update(1)
        val_acc = correct / total
        pbar.close()    
        print(f'Val Epoch_{epoch} accuracy: {val_acc}\n')
        return val_acc
    
    def test(self):
        self.model.to(self.device)
        self.model.eval()
        results = []
        pbar = tqdm(total=len(self.test_loader), ncols=80)
        with torch.no_grad():
            for inputs, file_name in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (preds > 0.5).astype(int) 
                for pred, name in zip(preds, file_name):
                    results.append((name, pred))
                pbar.update(1)
        pbar.close()    
        df = pd.DataFrame(results, columns=['image_id', 'y_pred'])
        df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', default=1, help='Size of batches')
    parser.add_argument('--lr', default=1e-4, help='Learning rate')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--model_path', default='weight/v1', help='Path to save model checkpoint')
    parser.add_argument('--resume', default=False, help='Continue for training')
    parser.add_argument('--ckpt', default='best_net.pth', help='Checkpoint for network')
    args = parser.parse_args()

    trainer = Trainer(args=args)
    os.makedirs(args.model_path, exist_ok=True)
    
    if args.test_only:
        trainer.load_model()
        acc = trainer.test()
        print(f'test acc: {acc}')
    
    else:
        trainer.train()