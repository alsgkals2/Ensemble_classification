""" Gradient boosting is a classic sequential ensemble method. At each iteration,
    the learning target of a newly-added base estimator is the pseudo residual
    computed based on the ground truth and the output from base estimators 
    fitted before. After then, the current estimator is fitted using ordinary 
    least square.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from ._base import BaseModule
from ._base import save_path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *

from ._base import BaseModule
from ._base import save_path, accuracy, show_result

shrinkage_rate=1
class GradientBoostingClassifier(BaseModule):
    global shirnkage_rate
    # def __init__(self, estimator, n_estimators, output_dim,
    #              lr, weight_decay, epochs,
    #              shrinkage_rate=1., cuda=True, log_interval=100):
    #     super(BaseModule, self).__init__()
    #
    #     self.estimator = estimator
    #     self.n_estimators = n_estimators
    #     self.output_dim = output_dim
    #
    #     self.lr = lr
    #     self.weight_decay = weight_decay
    #     self.epochs = epochs
    #     self.shrinkage_rate = shrinkage_rate
    #
    #     self.log_interval = log_interval
    #     self.device = torch.device('cuda' if cuda else 'cpu')
    #
    #     # Base estimators
    #     self.estimators_ = nn.ModuleList()
    #     for _ in range(self.n_estimators):
    #         self.estimators_.append(estimator().to(self.device))

    def forward(self, X):
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)

        # The output of `GradientBoostingClassifier` is the summation of output
        # from all base estimators, with each of them multiplied by the
        # shrinkage rate.
        for estimator in self.estimators_:
            y_pred += shrinkage_rate * estimator(X) #shirnk rate

        return y_pred

    def _validate_parameters(self):

        if not self.n_estimators > 0:
            msg = ('The number of base estimators = {} should be strictly'
                   ' positive.')
            raise ValueError(msg.format(self.n_estimators))

        if not self.output_dim > 0:
            msg = 'The output dimension = {} should not be strictly positive.'
            raise ValueError(msg.format(self.output_dim))

        if not self.lr > 0:
            msg = ('The learning rate of optimizer = {} should be strictly'
                   ' positive.')
            raise ValueError(msg.format(self.lr))

        if not self.weight_decay >= 0:
            msg = 'The weight decay of parameters = {} should not be negative.'
            raise ValueError(msg.format(self.weight_decay))

        if not self.epochs > 0:
            msg = ('The number of training epochs = {} should be strictly'
                   ' positive.')
            raise ValueError(msg.format(self.epochs))

        if not 0 < shrinkage_rate <= 1:
            msg = ('The shrinkage rate should be in the range (0, 1], but got'
                   ' {} instead.')
            raise ValueError(msg.format(shrinkage_rate))

    def _onehot_coding(self, y):
        """ Convert the class label to a one-hot encoded vector. """

        y = y.view(-1)
        y_onehot = torch.FloatTensor(
            y.size()[0], self.output_dim).to(self.device)
        y_onehot.data.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)

        return y_onehot

    # TODO: Store the output of *fitted* base estimators to avoid repeated data
    # forwarding. Since samples in the data loader can be shuffled, it requires
    # the index of each sample in the original dataset to be kept in memory.

    # TODO: Implement second order learning target, which is used in existing
    # decision tree based GBDT systems like XGBoost and LightGBM. However, this
    # can be problematic using estimators like neural network, as the second
    # order target can be orders of magnitude larger than the first order
    # target, making it hard to conduct ERM on the squared loss using gradient
    # descent based optimization strategies.
    def _pseudo_residual(self, X, y, est_idx):
        y_onehot = self._onehot_coding(y)
        output = torch.zeros_like(y_onehot).to(self.device)

        if est_idx == 0:
            # Before training the first estimator, we assume that the GBM model
            # always returns `0` for any input (i.e., null output).
            return y_onehot - F.softmax(output, dim=1)
        else:
            for idx in range(est_idx):
                output += shrinkage_rate * self.estimators_[idx](X)

            return y_onehot - F.softmax(output, dim=1)

    def fit(self, train_loader,val_loader,test_loader=None):

        self.train()
        self._validate_parameters()
        criterion = nn.MSELoss(reduction='sum')
        #criterion = nn.CrossEntropyLoss(reduction='sum')
        # Base estimators are fitted sequentially in gradient boosting
        for est_idx, estimator in enumerate(self.estimators_):

            # Initialize an independent optimizer for each base estimator to
            # avoid unexpected dependencies.
            learner_optimizer = torch.optim.Adam(
                estimator.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)

            for epoch in range(self.epochs):
                self.train()
                num_correct = 0
                train_data_num = 0
                for batch_idx, (X_train, y_train) in enumerate(train_loader):

                    X_train, y_train = (X_train.to(self.device),
                                        y_train.to(self.device))

                    # Learning target of the estimator with index `est_idx`
                    y_residual = self._pseudo_residual(X_train, y_train,
                                                       est_idx)

                    output = estimator(X_train)
                    loss = criterion(output, y_residual)

                    learner_optimizer.zero_grad()
                    loss.backward()
                    learner_optimizer.step()

                    y_pred = F.softmax(output, dim=1).data.max(1)[1]
                    num_correct += y_pred.eq(y_train.view(-1).data).sum()
                    train_data_num += len(y_pred)
                msg = ('Train Estimator: {:03d} | Epoch: {:03d} |'
                       ' Batch: {:03d} | Loss: {:.5f} | Correct:'
                       ' {}')
                print(msg.format(est_idx, epoch, batch_idx, loss,
                                 (float(num_correct) / float(train_data_num)) * 100, X_train.size()[0]))
                # Update the base estimator container
                self.estimators_[est_idx] = copy.deepcopy(estimator)

                self.eval()
                print("val ---------------------------")
                print(self.predict(val_loader))
                if test_loader is not None:
                    print("test --------------------------")
                    print(self.predict(test_loader, True))
                        # Print training status
                if batch_idx % self.log_interval == 0:
                    msg = ('Estimator: {:03d} | Epoch: {:03d} | Batch:'
                           ' {:03d} | RegLoss: {:.5f}')
                    print(msg.format(est_idx, epoch, batch_idx, loss))

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
            if save_path :
                torch.save(self.state_dict(), save_path)

    def loadModels(self,path):
        for i in range(len(path)):
            #print(self.estimators_[i])
            #("test: {}".format(self.estimators_[i].load_state_dict(torch.load(path[i]))))
            self.estimators_[i].load_state_dict(torch.load(path[i]))