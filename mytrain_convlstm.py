#System
import numpy as np
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import random
#Torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as standard_transforms

#from densenet import DenseNet
#from unet import UNet
#from linknet import LinkNet

from myconvlstm import MyConvLSTM as UConvLstm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#torch.cuda.set_device(1)
ckpt_path = 'ckpt_unet_dataset_v6' #ckpt
exp_name = 'USNeedle-U-ConvLSTM'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(os.path.join(ckpt_path, exp_name)):
    os.makedirs(os.path.join(ckpt_path, exp_name))
args = {
    'num_class': 2,
    'ignore_label': 255,
    'num_gpus': 1,
    'start_epoch': 1,
    'num_epoch': 200,
    'batch_size': 2, #1
    'lr': 0.0001,
    'lr_decay': 0.9,
    'dice': 0,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'opt': 'adam',
}

#         mdice
class TSHRDataset(Dataset):
    def __init__(self, img_dir):
        img_list = glob(img_dir)
        self.img_anno_pairs = []
        for i in range(len(img_list)):
            self.img_anno_pairs.append(img_list[i])
    
    def __len__(self):
        return len(self.img_anno_pairs)

    def __getitem__(self, index):
        img_index = int(self.img_anno_pairs[index][-12:-9])
        _img_sequence = []
        _target_sequence = []
        hflip = random.random() < 0.5
        for i in range (1, 11):
            _img_path = self.img_anno_pairs[index][:-12] + "%03d" % (i) + ".png"
            _target_path = self.img_anno_pairs[index]
            _img = Image.open(_img_path).convert('RGB')
            _target = Image.open(self.img_anno_pairs[index]).convert('L')
            if hflip:
                _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
                _target = _target.transpose(Image.FLIP_LEFT_RIGHT)
            _target = np.array(_target)
            _target[_target == 255] = 1
            _img = torch.from_numpy(np.array(_img).transpose(2,0,1)).float()
            _target = torch.from_numpy(np.array(_target)).long()
            
            _img_sequence.append(_img)
            _target_sequence.append(_target)
               
        _img_sequence = torch.stack(_img_sequence)
        _target_sequence = torch.stack(_target_sequence)
        
        return _img_sequence, _target_sequence


class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = torch.nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


if __name__ == '__main__':

    #img_dir = #removed due to security reasoning
    dataset = TSHRDataset(img_dir=img_dir)
    print(dataset.__len__())
    train_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=True, num_workers=1, drop_last=True)
    model = UConvLstm()
    gpu_ids = range(args['num_gpus'])
    model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()
    if args['opt'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=0.0002,
                              momentum=0.99, weight_decay=0.0001)
    elif args['opt'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args['lr'], weight_decay=0.0001)

    criterion = CrossEntropyLoss2d(size_average=True).cuda()
    model.train()
    epoch_iters = dataset.__len__() / args['batch_size']
    max_epoch = args['num_epoch']
    for epoch in range(max_epoch):
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data
            print(inputs.shape)
            print(labels.shape)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 20 == 0:
                print('[epoch %d], [iter %d / %d], [train main loss %.5f], [lr %.10f]' % (
                    epoch, batch_idx + 1, epoch_iters, loss.item(),
                    optimizer.param_groups[0]['lr']))

        snapshot_name = 'epoch_' + str(epoch)
        torch.save(model.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth.tar'))

