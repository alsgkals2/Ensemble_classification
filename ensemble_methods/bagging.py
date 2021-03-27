import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from ._base import BaseModule
from ._base import save_path, accuracy, show_result


def _parallel_fit(epoch, estimator_idx,
                  estimator, data_loader, criterion ,lr, weight_decay,
                  device, log_interval, ixgb_classification=True):
    """ Private function used to fit base estimators in parallel.
    """
    #estimator.train()
    num_correct =0
    train_data_num=0
    #optimizer = torch.optim.Adam(estimator.parameters(),
    #                             lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(estimator.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    for batch_idx, (X_train, y_train) in enumerate(data_loader):

        batch_size = X_train.size()[0]
        X_train, y_train = (X_train.to(device),
                            y_train.to(device))
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        # In `BaggingClassifier`, each base estimator is fitted on a batch of
        # data after sampling with replacement.
        sampling_mask = torch.randint(high=batch_size,
                                      size=(int(batch_size),),
                                      dtype=torch.int64)
        sampling_mask = torch.unique(sampling_mask)  # remove duplicates
        sampling_X_train = X_train[sampling_mask]
        sampling_y_train = y_train[sampling_mask]
        output = estimator(sampling_X_train.cuda())
        loss = criterion(output, sampling_y_train.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print training status
        y_pred = F.softmax(output, dim=1).data.max(1)[1]
        num_correct += y_pred.eq(sampling_y_train.view(-1).data).sum()
        train_data_num += len(y_pred)
    msg = ('Estimator: {:03d} | Epoch: {:03d} |'
           ' Batch: {:03d} | Loss: {:.5f} | Correct:'
           ' {}')
    print(msg.format(estimator_idx, epoch, batch_idx, loss,
                     (float(num_correct) / float(train_data_num)) * 100, batch_size))

    return estimator

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
        self.train()
        self._validate_parameters()
        criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.Adam(estimator.parameters(),
        #                             lr=0.01, weight_decay=weight_decay)
        with Parallel(n_jobs=self.n_jobs) as parallel:
            for epoch in range(self.epochs):
            # TODO: Parallelization
                self.scheduler.step()
                print(self.scheduler.get_lr()[0])
                # rets = parallel(delayed(_parallel_fit)(
                #     epoch, idx, estimator, train_loader,criterion,
                #     self.lr, self.weight_decay, self.device, self.log_interval)
                #                 for idx, estimator in enumerate(self.estimators_))
                rets = parallel(delayed(_parallel_fit)(
                    epoch, idx, estimator, train_loader, criterion,
                    float(self.scheduler.get_last_lr()[0]), self.weight_decay, self.device, self.log_interval)
                                for idx, estimator in enumerate(self.estimators_))
                # Update the base estimator container
                for i in range(self.n_estimators):
                    self.estimators_[i] = copy.deepcopy(rets[i])
                self.eval()
                print("val -------------------")
                print(self.predict(val_loader))
                if test_loader is not None:
                    print("test ------------------")
                    print(self.predict(test_loader,True))

    def loadModels(self,path):
        for i in range(len(path)):
            self.estimators_[i].load_state_dict(torch.load(path[i]))

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
                """for j in range(len(preds)):
                    if preds[j] == 2 or preds[j] == 3:
                        preds[j] = 0
                    if val[j] == 2 or val[j] == 3:
                        val[j] = 0"""
        incorrect_index, accuracy, precision, recall, f1 = show_result(y_pred, y_val)
        if (self.best_acc < accuracy and not is_test):
            self.best_acc = accuracy
            if save_path :
                from os.path import join
                torch.save(self.state_dict(), join(save_path,"bestweight"))

