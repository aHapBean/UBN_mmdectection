import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
# import faiss
import time
    
def get_cos(x):
    # print("bs: ", x.shape[0])       # bs 2
    with torch.no_grad():
        if x.shape[0] > 16: # NOTE here
            x = x[:16]
        n = x.shape[0]
        if n == 1:
            return 1
        x = x.view(n, -1)
        # print(x)
        cos = torch.sum(F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2))
        cos = (cos - n) / (n * (n - 1))
        # print(cos)
        return cos.item()  # 返回标量值

def get_sub_cos(x1,x2):
    return torch.sum(F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=2))

class UnifiedBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, bound=0.15, k=3):  # NOTE 0.15
        super(UnifiedBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.num_features = num_features
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        ### weights for affine transformation in BatchNorm ###
        self.bound = bound
        self.moving_cos = torch.ones(1)
        self.momentum_cos = 0.1


        ### weights for centering calibration ###        
        self.center_weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.center_weight.data.fill_(0)
        ### weights for scaling calibration ###            
        self.scale_weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.scale_bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.scale_weight.data.fill_(0)
        self.scale_bias.data.fill_(1)
        ### calculate statistics ###
        self.stas = nn.AdaptiveAvgPool2d((1,1))
        
        ## BNET
        self.bnconv = nn.Conv2d(num_features,
                              num_features,
                              k,
                              padding=(k - 1) // 2,
                              groups=num_features,
                              bias=True)


    def forward(self, input):
        # self._check_input_dim(input)
        cos = get_cos(input.detach())
        self.moving_cos = ((1.0 - self.momentum_cos) * self.moving_cos + self.momentum_cos * cos).detach()
        ok_to_running = bool(self.moving_cos.item() > self.bound)  # if ok and training then moving average
        ####### centering calibration begin ####### 
        input = input + self.center_weight.view(1,self.num_features,1,1)*self.stas(input)# +=
        ####### centering calibration end ####### 
        ####### BatchNorm begin #######
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        output = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            (self.training or not self.track_running_stats) and (ok_to_running),
            exponential_average_factor, self.eps)
        ####### BatchNorm end #######
        
        ####### scaling calibration begin ####### 
        scale_factor = torch.sigmoid(self.scale_weight*self.stas(output)+self.scale_bias)
        ####### scaling calibration end ####### 
        if self.affine:
            return scale_factor*self.bnconv(output)
        else:
            return scale_factor*output