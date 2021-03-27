""" 
  In voting-based ensemble methods, each base estimator is trained independently,
  ad the final prediction takes the average over predictions from all base 
  estimators.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

from ._base import BaseModule
from ._base import save_path, accuracy, show_result

# TODO: Memory optimization by registering read-only objects into shared memory.
"""def _parallel_fit(epoch, estimator_idx, 
                  estimator, data_loader, criterion, lr, weight_decay, 
                  device, log_interval):    
    optimizer = torch.optim.Adam(estimator.parameters(), 
                                 lr=lr, weight_decay=weight_decay)
    
    for batch_idx, (X_train, y_train) in enumerate(data_loader):
        
        batch_size = X_train.size()[0]
        X_train, y_train = (X_train.to(device), 
                            y_train.to(device))
        
        output = estimator(X_train)
        loss = criterion(output, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print training status
        if batch_idx % log_interval == 0:
            y_pred = F.softmax(output, dim=1).data.max(1)[1]
            correct = y_pred.eq(y_train.view(-1).data).sum()
            
            msg = ('Estimator: {:03d} | Epoch: {:03d} |' 
                   ' Batch: {:03d} | Loss: {:.5f} | Correct:'
                   ' {:d}/{:d}')
            print(msg.format(estimator_idx, epoch, batch_idx, loss, 
                             correct, batch_size))
    
    return estimator
"""

import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
def show_result(y_pred, y_val):
    num_acc_data=0
    for i in range(len(y_pred)):
        if y_pred[i] == 2 or y_pred[i] == 3:
            y_pred[i] = 0
        if y_val[i] == 2 or y_val[i] == 3:
            y_val[i] = 0

    incorrect_index = []
    for i in range(len(y_pred)):
        if y_pred[i] != y_val[i]:
            incorrect_index.append(y_pred[i])

    y_pred=torch.tensor(y_pred)
    y_val = torch.tensor(y_val)
    num_acc_data += y_pred.eq(y_val.view(-1).data).sum()
    #accuracy = float(num_acc_data)/float(len(y_val))
    accuracy = 100 * accuracy_score(y_val, y_pred)
    precision = 100 * precision_score(y_val, y_pred)
    recall = 100 * recall_score(y_val, y_pred)
    f1 = 100 * f1_score(y_val, y_pred)
    print("accuracy : {:06f}% precision : {:06f}%, recall : {:06f}% f1 score : {:06f}%".format(accuracy, precision, recall, f1))
    confusion_mtx = confusion_matrix(y_val, y_pred)
    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
    """    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()"""
    return incorrect_index, accuracy, precision, recall, f1

def _parallel_fit(epoch, estimator_idx,
                  estimator, data_loader, val_loader,test_loader,criterion, lr, weight_decay,
                  device, log_interval):
    """ Private function used to fit base estimators in parallel.
    """
    #estimator.train()
    optimizer = torch.optim.Adam(estimator.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    num_correct =0
    train_data_num=0
    for batch_idx, (X_train, y_train) in enumerate(data_loader):

        batch_size = X_train.size()[0]
        X_train, y_train = (X_train.to(device),
                            y_train.to(device))

        output = estimator(X_train)
        loss = criterion(output, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Print training status
        y_pred = F.softmax(output, dim=1).data.max(1)[1]
        num_correct += y_pred.eq(y_train.view(-1).data).sum()
        train_data_num += len(y_pred)
    msg = ('Estimator: {:03d} | Epoch: {:03d} |'
           ' Batch: {:03d} | Loss: {:.5f} | Correct:'
           ' {}')
    print(msg.format(estimator_idx, epoch, batch_idx, loss,
                     (float(num_correct)/float(train_data_num))*100, batch_size))

    return estimator


class VotingClassifier(BaseModule):
    def forward(self, X):
        batch_size = X.size()[0]
        y_pred_proba = torch.zeros(batch_size, self.output_dim).to(self.device)
        
        # Average over the class distributions from all base estimators.
        for estimator in self.estimators_:
            #print(estimator(X))
            y_pred_proba += F.softmax(estimator(X), dim=1)
        y_pred_proba /= self.n_estimators
        
        return y_pred_proba
    
    def fit(self, train_loader,val_loader,test_loader):
        
        self.train()
        self._validate_parameters()
        criterion = nn.CrossEntropyLoss()
        
        # Create a pool of workers for repeated calls to the joblib.Parallel
        with Parallel(n_jobs=self.n_jobs) as parallel:
            for epoch in range(self.epochs):

                rets = parallel(delayed(_parallel_fit)(
                    epoch, idx, estimator, train_loader,val_loader,test_loader, criterion,
                    self.lr, self.weight_decay, self.device, self.log_interval)
                    for idx, estimator in enumerate(self.estimators_))

                # Update the base estimator container
                for i in range(self.n_estimators):
                    self.estimators_[i] = copy.deepcopy(rets[i])
                self.eval()
                print("val -------------------")
                print(self.predict(val_loader))
                print("test ------------------")
                print(self.predict(test_loader,True))

    def predict(self, test_loader, is_test=False):
        y_pred = []
        y_val = []
        num_acc_data = 0
        with torch.no_grad():
            for i, (image, label) in enumerate(test_loader):
                cpu_image = image.detach().numpy()
                image = image.to('cuda')
                label = label.to('cuda')
                outputs = self.forward(image)
                _, output_idx = torch.max(outputs, 1)
                preds = output_idx.cpu().numpy()
                val = label.cpu().numpy()
                y_pred += list(preds)
                y_val += list(val)
                for j in range(len(preds)):
                    if preds[j] == 2 or preds[j] == 3:
                        preds[j] = 0
                    if val[j] == 2 or val[j] == 3:
                        val[j] = 0
        incorrect_index, accuracy, precision, recall, f1 = show_result(y_pred, y_val)
        if (self.best_acc < accuracy and not is_test):
            self.best_acc = accuracy
            #save_path = '/home/mhkim/AGC/no_pre_weight_pt/voting_not_pretrain_3_efficientnetb0_ns_fit_nopad_zip_stage1stage2_yuhyun_15_96.38157894736842'
            if save_path :
                torch.save(self.state_dict(), save_path)

    def loadModels(self,path):
        for i in range(len(path)):
            #print(self.estimators_[i])
            #("test: {}".format(self.estimators_[i].load_state_dict(torch.load(path[i]))))
            self.estimators_[i].load_state_dict(torch.load(path[i]))