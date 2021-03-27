import timm
import abc
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from os.path import join,exists
from os import makedirs
from torch.optim import lr_scheduler
import net_twoclass as twoNet
save_path = '/home/mhkim/AGC/no_pre_weight_pt'
name_model = ['tf_efficientnet_b0_ns','tf_efficientnet_b0_ns','tf_efficientnet_b1_ns']
file_name='ef0ef0ef1_only_two_pretrain_adamw_toy_n_half_iitp_scheduler_5'
save_path = join(save_path,file_name)
if not exists(save_path):
    print("making the path : {}".format(save_path)+'')
    makedirs(save_path)
num_class = 2
pretrain_imagenet = True

def accuracy(output, target, topk=[1]):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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
    return incorrect_index, accuracy, precision, recall, f1

class BaseModule(abc.ABC, nn.Module):
    """
      Base class for ensemble methods.
      
      WARNING: This class cannot be used directly. 
      Use the derived classes instead.
    """
    
    def __init__(self, 
                 estimator, 
                 n_estimators, 
                 output_dim, 
                 lr,
                 weight_decay, 
                 epochs, 
                 cuda=True, 
                 log_interval=100,
                 n_jobs=1):
        """
        Parameters
        ----------
        estimator : torch.nn.Module
            The base estimator class inherited from `torch.nn.Module`. Examples
            are available in the folder named `model`.
        n_estimators : int
            The number of base estimators in the ensemble model.
        output_dim : int
            The output dimension of the model. For instance, for multi-class
            classification problem with K classes, it is set to `K`. For 
            univariate regression problem, it is set to `1`.
        lr : float
            The learning rate of the parameter optimizer.
        weight_decay : float
            The weight decay of the parameter optimizer.
        epochs : int
            The number of training epochs.
        cuda : bool, default=True
            When set to `True`, use GPU to train and evaluate the model. When 
            set to `False`, the model is trained and evaluated using CPU.
        log_interval : int, default=100
            The number of batches to wait before printing the training status,
            including information on the current batch, current epoch, current 
            training loss, and many more.
        n_jobs : int, default=1
            The number of workers for training the ensemble model. This argument
            is used for parallel ensemble methods such as voting and bagging.
            Setting it to an integer larger than `1` enables many base 
            estimators to be jointly trained. However, training many base 
            estimators at the same time may run out of the memory.
        
        Attributes
        ----------
        estimators_ : nn.ModuleList
            A container that stores all base estimators.
        
        """
        super(BaseModule, self).__init__()
        self.estimator = estimator
        #self.estimator = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=3)
        self.n_estimators = n_estimators
        self.output_dim = output_dim

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.log_interval = log_interval
        self.n_jobs = n_jobs
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.best_acc = .0
        # Initialize base estimators
        self.estimators_ = nn.ModuleList()

        self.estimators_.append(timm.create_model('tf_efficientnet_b0_ns', pretrained=pretrain_imagenet, num_classes=num_class).to('cuda'))
        self.estimators_.append(twoNet.Net())
        self.estimators_.append(twoNet.Net())
        #for i in range(self.n_estimators):
            #self.estimators_.append(timm.create_model(name_model[i], pretrained=pretrain_imagenet, num_classes=num_class).to('cuda'))
            #print("appended " + name_model[i])
            #self.estimators_.append(twoNet.Net())


        # A global optimizer
        #self.optimizer = torch.optim.Adam(self.parameters(),
        #                                  lr=lr, weight_decay=weight_decay)
        self.optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=lr, weight_decay=weight_decay)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        print("optimizer : {}".format(self.optimizer))
    def __str__(self):
        msg = '==============================\n'
        msg += '{:<20}: {}\n'.format('Base Estimator', 
                                     self.estimator.__name__)
        msg += '{:<20}: {}\n'.format('n_estimators', self.n_estimators)
        msg += '{:<20}: {:.5f}\n'.format('Learning Rate', self.lr)
        msg += '{:<20}: {:.5f}\n'.format('Weight Decay', self.weight_decay)
        msg += '{:<20}: {}\n'.format('n_epochs', self.epochs)
        msg += '==============================\n'
        
        return msg
    
    def __repr__(self):
        return self.__str__()
    
    def _validate_parameters(self):
        
        if not self.n_estimators > 0:
            msg = ('The number of base estimators = {} should be strictly'
                   ' positive.')
            raise ValueError(msg.format(self.n_estimators))
        
        if not self.output_dim > 0:
            msg = 'The output dimension = {} should be strictly positive.'
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
    
    @abc.abstractmethod
    def forward(self, X):
        """ Implementation on the data forwarding in the ensemble model. Notice
            that the input `X` should be a data batch instead of a standalone
            data loader that contains all data batches.
        """
    
    @abc.abstractmethod
    def fit(self, train_loader):
        """ Implementation on the training stage of the ensemble model.
        """
    
    @abc.abstractmethod
    def predict(self, test_loader):
        """ Implementation on the evaluating stage of the ensemble model.
        """
