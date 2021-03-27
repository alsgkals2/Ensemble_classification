import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import BaseModule
name_model=[]
class BaggingClassifier(BaseModule):
    def forward(self, X):
        batch_size = X.size()[0]
        y_pred_proba = torch.zeros(batch_size, self.output_dim).to(self.device)

        # Average over the class distributions predicted from all base estimators
        for estimator in self.estimators_:
            y_pred_proba += F.softmax(estimator(X), dim=1)
        y_pred_proba /= self.n_estimators

        return y_pred_proba


    def fit(self, train_loader, val_loader,test_loader=None):
        pass


    def predict(self, test_loader, is_test=False):
        pass