import net
import torch
import os
# from torchsummary import summary
# model = net.Net().cuda()
# summary(model.cuda(), (3, 128, 128))
path = '/home/mhkim/AGC/kd_test_model_json'
model = net.Net().cuda()
model.load_state_dict(torch.load(os.path.join(path,'metrics_val_best_weights_acc97.0203488372093')))

