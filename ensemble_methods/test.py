import timm
import torch
import torch.nn as nn
from math import ceil
import torchvision
import numpy as np
import pandas as pd
from sklearn.metrics import *
import seaborn as sns
import os
import time
from sys import exit
import matplotlib.pyplot as plt
import math
from torch.optim.optimizer import Optimizer
os.environ["CUDA_VISIBLE_DEVICES"] ="2"
import glob
from PIL import Image
import cv2
import sys



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0

            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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
    print("accuracy는??{}".format(accuracy))
    precision = 100 * precision_score(y_val, y_pred)
    recall = 100 * recall_score(y_val, y_pred)
    f1 = 100 * f1_score(y_val, y_pred)
    print("accuracy : {:06f}% precision : {:06f}%, recall : {:06f}% f1 score : {:06f}%".format(accuracy, precision, recall, f1))
    confusion_mtx = confusion_matrix(y_val, y_pred)
    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    return incorrect_index, accuracy, precision, recall, f1

def save_info(data, file_name):
    n=1
    while(os.path.isfile("./result_info/{}_{}.txt".format(file_name, n))): n = n+1
    with open("./result_info/{}_{}.txt".format(file_name, n), "w") as f:
        f.write(data)

class dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, input_size, train = True, padding = True, inbalance = False, aug_falldown = False,
                 bright_ness = 0.2, hue = 0.15, contrast = 0.15, random_Hflip = 0.3, rotate_deg = 20):
        self.orig_back_paths = glob.glob(os.path.join(root_dir, "background") + "/*.jpg") # 0
        if not aug_falldown:
            self.orig_fall_paths = glob.glob(os.path.join(root_dir, "falldown") + "/*.jpg")  # 1
        else:
            self.orig_fall_paths = glob.glob(os.path.join(root_dir, "jeongho_falldown_augimg") + "/*.jpg") # 1
        self.orig_normal_paths = glob.glob(os.path.join(root_dir, "normal") + "/*.jpg") # 2
        self.back_paths = []
        self.fall_paths = []
        self.normal_paths = []
        for path in self.orig_back_paths:
            img = cv2.imread(path)
            if min(img.shape[0], img.shape[1]) < 32:
                pass
            else:
                self.back_paths.append(path)
        for path in self.orig_fall_paths:
            img = cv2.imread(path)
            if min(img.shape[0], img.shape[1]) < 32:
                pass
            else:
                self.fall_paths.append(path)
        for path in self.orig_normal_paths:
            img = cv2.imread(path)
            if min(img.shape[0], img.shape[1]) < 32:
                pass
            else:
                self.normal_paths.append(path)
        if not inbalance:
            self.normal_paths = np.random.choice(self.normal_paths, size= len(self.fall_paths)).tolist()
        print("normal : {}, fall : {}, back : {}".format(len(self.normal_paths), len(self.fall_paths), len(self.back_paths)))
        self.labels = [0] * len(self.back_paths) + [1] * len(self.fall_paths) + [2] * len(self.normal_paths)
        self.total_paths = self.back_paths + self.fall_paths + self.normal_paths
        transform = []
        if train:
            transform.append(torchvision.transforms.ColorJitter(brightness=bright_ness, hue=hue, contrast=contrast))
            transform.append(torchvision.transforms.RandomHorizontalFlip(p=random_Hflip))
            #transform.append(torchvision.transforms.RandomCrop(224))
            transform.append(torchvision.transforms.RandomRotation(degrees=rotate_deg))
        transform.append(torchvision.transforms.ToTensor())
        transform.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        if padding:
            transform.append(lambda x: torchvision.transforms.Pad(((128 - x.shape[2]) // 2, (128 - x.shape[1]) // 2), fill=0,
                                                     padding_mode="constant")(x))
        transform.append(torchvision.transforms.Resize((input_size, input_size)))
        self.transform = torchvision.transforms.Compose(transform)
        print(self.transform)

    def __len__(self):
        return len(self.total_paths)

    def __getitem__(self, index):
        img = Image.open(self.total_paths[index])
        img = self.transform(img)
        return img, self.labels[index]

class Crossentropy(nn.Module):
    def __init__(self, classes, LS = True, smoothing=0.0, dim=-1):
        super(Crossentropy, self).__init__()
        if LS:
            smoothing = smoothing
        else:
            smoothing = 0.0
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



# 하이퍼 파라미터 * 훈련
DATA = "stage1stage2_yuhyun"
INPUT_SIZE = 128
BATCH_SIZE = 128
NUM_CLASSES = 3
NUM_EPOCH = 50
PATIENCE = 10
MOMENTUM = 0.9
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-03
BRIGHTNESS = 0
HUE = 0.15
CONTRAST = 0.15
RANDOM_HFLIP = 0
ROTATE_DEG = 0
PADDING = False
INBALANCE = False
AUG_FALLDOWN = False
LS = False
LS_alpha = 0.1
FILE_NAME = os.path.relpath(__file__[:-3])
CP_PATH = os.path.join("./checkpoint", FILE_NAME + ".pt")

print("{} start...".format(FILE_NAME))
if DATA == "stage1stage2":
    train_data = dataset(root_dir="/media/data2/AGC/patch_image/labeled/no_padding_stage1stage2_yuhyun/train", input_size=INPUT_SIZE,
                         train=True, padding=PADDING, inbalance=INBALANCE, aug_falldown=AUG_FALLDOWN,
                         bright_ness=BRIGHTNESS, hue=HUE, contrast=CONTRAST, random_Hflip=RANDOM_HFLIP,
                         rotate_deg=ROTATE_DEG)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                               drop_last=True)

    val_data = dataset(root_dir="/media/data2/AGC/patch_image/labeled/no_padding_stage1stage2_yuhyun/valid", input_size=INPUT_SIZE,
                       train=False, padding=PADDING, inbalance=True, aug_falldown=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                             drop_last=True)
elif DATA == "stage1":
    train_data = dataset(root_dir="/media/data2/AGC/patch_image/labeled/no_padding_stage1/train", input_size = INPUT_SIZE,
                         train = True, padding=PADDING, inbalance=INBALANCE, aug_falldown=AUG_FALLDOWN,
                         bright_ness=BRIGHTNESS, hue=HUE, contrast=CONTRAST, random_Hflip=RANDOM_HFLIP, rotate_deg=ROTATE_DEG)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                               drop_last=True)

    val_data = dataset(root_dir="/media/data2/AGC/patch_image/labeled/no_padding_stage1/valid", input_size = INPUT_SIZE,
                        train = False, padding=PADDING, inbalance=True, aug_falldown=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                              drop_last=True)
elif DATA=="stage1stage2_yuhyun":
    train_data = dataset(root_dir="/home/mhkim/AGC2/patch_image/labeled/no_padding_stage1stage2_yuhyun/train",
                         input_size=INPUT_SIZE,
                         train=True, padding=PADDING, inbalance=INBALANCE, aug_falldown=AUG_FALLDOWN,
                         bright_ness=BRIGHTNESS, hue=HUE, contrast=CONTRAST, random_Hflip=RANDOM_HFLIP,
                         rotate_deg=ROTATE_DEG)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                               drop_last=True)

    val_data = dataset(root_dir="/home/mhkim/AGC2/patch_image/labeled/no_padding_stage1stage2_yuhyun/valid",
                       input_size=INPUT_SIZE,
                       train=False, padding=PADDING, inbalance=True, aug_falldown=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                             drop_last=True)
else:
    train_data = dataset(root_dir="/media/data2/AGC/patch_image/labeled/no_padding_stage2/train", input_size=INPUT_SIZE,
                         train=True, padding=PADDING, inbalance=INBALANCE, aug_falldown=AUG_FALLDOWN,
                         bright_ness=BRIGHTNESS, hue=HUE, contrast=CONTRAST, random_Hflip=RANDOM_HFLIP,
                         rotate_deg=ROTATE_DEG)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                               drop_last=True)

    val_data = dataset(root_dir="/media/data2/AGC/patch_image/labeled/no_padding_stage2/valid", input_size=INPUT_SIZE,
                       train=False, padding=PADDING, inbalance=True, aug_falldown=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                             drop_last=True)

test_data = dataset(root_dir="/home/mhkim/AGC2/val",input_size=INPUT_SIZE, train = False, padding = PADDING, inbalance=True,
                    aug_falldown=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# test model
LOAD_FILE_NAME = "efficientnetb0_ns_fit_nopad_zip_stage1stage2_yuhyun_15_96.38157894736842.pt"
LOAD_FILE_PATH = os.path.join("/home/mhkim/AGC/torchensemble/efficientnet/save_model", LOAD_FILE_NAME)
model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=3)
model.load_state_dict(torch.load(LOAD_FILE_PATH))
model.to(device)
print("load success!!!")
print("model evaluation...")
y_pred = []
y_val = []
model.eval()
incorrect_img = []
start_time = time.time()
with torch.no_grad():
    for i, (image, label) in enumerate(test_loader):
        print("[{}/{}]".format(i, len(test_loader)))
        cpu_image = image.detach().numpy()

        image = image.to(device)
        label = label.to(device)

        outputs = model.forward(image)
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
        isincorrect = np.where(preds != val, 1, 0)
        for index, j in enumerate(isincorrect):
            if j:
                incorrect_img.append(cpu_image[index])

inference_time = time.time() - start_time
print("inference time : {}".format(time.time() - start_time))
incorrect_index, accuracy, precision, recall, f1 = show_result(y_pred, y_val)
if not os.path.isdir("./incorrect_imgs"):
    os.makedirs("./incorrect_imgs")
count = [0,0]

# 파일이름 앞자리 0 : 정상, 1: 쓰러짐으로 예측한 것.
for idx, img in enumerate(incorrect_img):
    img = (np.transpose(img, (1,2,0)) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if incorrect_index[idx] ==0:
        cv2.imwrite("./incorrect_imgs/0_{0:06d}.png".format(count[0]), img)
        count[0] +=1
    else:
        cv2.imwrite("./incorrect_imgs/1_{0:06d}.png".format(count[1]), img)
        count[1] +=1