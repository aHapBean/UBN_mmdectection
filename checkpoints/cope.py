# 我们的pth里面有 model, optimizer, lr_scheduler, epoch, args的 .pth 文件中提取model

import torch
checkpoint = torch.load('r101_ubn_all.pth')
torch.save(checkpoint['model'], 'r101_ubn.pth')