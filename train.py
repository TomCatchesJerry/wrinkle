import math
import os
import pdb
import cv2
import time
import glob
import random
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.autograd import Variable
import gc
import numpy as np
import torch.nn.functional as F
import torch # PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp # https://pytorch.org/docs/stable/notes/amp_examples.html
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, KFold # Sklearn
import albumentations as A # Augmentations
import timm
# import segmentation_models_pytorch as smp # smp
from timm.data.mixup import Mixup
from sklearn.metrics import confusion_matrix
from adamp import AdamP
# import torch_optimizer as optim
import logging
# import common
from albumentations import (Blur,Flip,ShiftScaleRotate,GridDistortion,ElasticTransform,HorizontalFlip,CenterCrop,
                            HueSaturationValue,Transpose,RandomBrightnessContrast,CLAHE,RandomCrop,Cutout,CoarseDropout,
                            CoarseDropout,Normalize,ToFloat,OneOf,Compose,Resize,RandomRain,RandomFog,Lambda
                            ,ChannelDropout,ISONoise,VerticalFlip,RandomGamma,RandomRotate90)


"""
torch 官网各种技巧实现
https://github.com/pytorch/vision/tree/main/references/classification
timm 各种使用方法
https://www.aiuai.cn/aifarm1967.html
albumentations 图像增强库
https://albumentations.ai/docs/
timm 库实现了最新的几乎所有的具有影响力的视觉模型，它不仅提供了模型的权重，还提供了一个很棒的分布式训练和评估的代码框架，方便后人开发
"""




unique_labels = ["0","1","2","3"]
mixup_args = {'mixup_alpha': 1.,
             'cutmix_alpha': 1.,
             'prob': 0.2,
             'switch_prob': 0.2,
             'mode': 'batch',
             'label_smoothing': 0.05,  # 0.05 改为0.1
             'num_classes': 9}
mixup_fn = Mixup(**mixup_args)

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def get_logger(filename, verbosity=1, name=__name__):
    '''
    写log日志的
    :param filename: log的文件名
    :param verbosity:
    :param name:
    :return:
    '''
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def set_seed(seed=42):
    """
    初始化随机数生成器，为了保证能够复现同样的训练结果
    :param seed:随机数种子
    :return:
    """
    ##### why 42? The Answer to the Ultimate Question of Life, the Universe, and Everything is 42.
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True# if True, causes cuDNN to only use deterministic convolution algorithms
    torch.backends.cudnn.benchmark = False# if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest




class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
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

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(c1 * 4, c2, k, 1)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))







train_aug = A.Compose([
    # A.Resize(224,224 ,interpolation=cv2.INTER_NEAREST, p=1.0),
    # A.CenterCrop(448, 448, always_apply=False, p=1.0),
    A.HorizontalFlip(p=0.3),#水平翻转，关于垂直方向对称
    # A.VerticalFlip(p=0.5),#垂直翻转，关于水平向对称
    # A.Transpose(p=0.2),#转置
    # A.RandomRotate90(always_apply=False, p=0.1),#旋转90度
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.3),  #0.5#平移旋转缩放
    A.RandomBrightnessContrast(brightness_limit =  0.1,contrast_limit = 0.1,p=0.2),#随机亮度增强随机对比度
    # A.RandomContrast(limit = 0.2,p=0.2),#

    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, always_apply=False, p=0.2),#随机更改输入图像的色相，饱和度和值
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.4),#网格畸变
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.4),#光学畸变
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.4)#产生类似水的效果
    ], p=0.2),
    A.CoarseDropout(max_holes=8, max_height=224 // 20, max_width=224 // 20,
                    min_holes=5, fill_value=0, mask_fill_value=0, p=0.2),#在图像上生成矩形区域

    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),   #mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)
    ToTensorV2(),
])
# val_aug = timm.data.create_transform(
#         input_size=(384, 384), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
#     )

val_aug = A.Compose([
    # A.Resize(224, 224,interpolation=cv2.INTER_NEAREST, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
    ToTensorV2(),
])



class lwhdataset(Dataset):
    def __init__(self,data_dir,train_transform,size,pad):
        self.pad = pad
        self.size = size
        self.label_dict = {"0":0,"1":1,"2":2,"3":3}
        self.c_paths = sorted(data_dir)
        self.transforms = train_transform

    def __getitem__(self, index):
        # print(self.c_paths[index].split('\\')[-2])

        label = self.label_dict[self.c_paths[index].split('\\')[-2]]
        # print(label)
        # image_name = self.c_paths[index].split('\\')[-1]

        # image = Image.open(self.c_paths[index]).convert('LA')
        image = Image.open(self.c_paths[index]).convert("RGB")
        # image = image.resize((self.size,self.size))
        # image = image.convert("RGB")
        if self.pad:
            image = self.pading(self.size,image)
            image = np.array(image)
        else:
            image = image.resize((self.size, self.size))
            image = np.array(image)
        image = self.transforms(image=image)['image']
        # image = self.transforms(image)

        return image, label

    def __len__(self):
        if len(self.c_paths) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.c_paths)

    @staticmethod
    def get_id_labels(data_dir):
        image_fns = glob.glob(os.path.join(data_dir, '*'))
        label_names = [os.path.split(s)[-1] for s in image_fns]
        unique_labels = list(set(label_names))
        unique_labels.sort()
        id_labels = {_id: name for name, _id in enumerate(unique_labels)}

        return id_labels


    @staticmethod
    def pading(size,img):
        padding_v = tuple([125, 125, 125])

        w, h = img.size

        target_size = size

        interpolation = Image.BILINEAR

        if w > h:
            img = img.resize((int(target_size), int(h * target_size * 1.0 / w)), interpolation)
        else:
            img = img.resize((int(w * target_size * 1.0 / h), int(target_size)), interpolation)

        ret_img = Image.new("RGB", (target_size, target_size), padding_v)
        w, h = img.size
        st_w = int((ret_img.size[0] - w) / 2.0)
        st_h = int((ret_img.size[1] - h) / 2.0)
        ret_img.paste(img, (st_w, st_h))
        # ret_img = np.array(ret_img)
        # ret_img.show()
        return ret_img


class lwhdataset_swa(Dataset):
    def __init__(self,data_dir,train_transform,size,pad):
        self.pad = pad
        self.size = size
        self.label_dict = {"0":0,"1":1,"2":2,"3":3}
        # self.label_dict = {"d1":0,"d2":0,"d3":1,'d4':0,'d5':0,'d6':0,'d7':2,'d8':3,'d9':4}
        self.c_paths = sorted(data_dir)
        self.transforms = train_transform

    def __getitem__(self, index):


        label = self.label_dict[self.c_paths[index].split('\\')[-2]]
        # image_name = self.c_paths[index].split('\\')[-1]
        # image = Image.open(self.c_paths[index]).convert('LA')
        image = Image.open(self.c_paths[index]).convert("RGB")
        # image = image.resize((self.size,self.size))
        # image = image.convert("RGB")
        if self.pad:
            image = self.pading(self.size,image)
            image = np.array(image)

        else:
            image = image.resize((self.size, self.size))
            image = np.array(image)
        #
        image = self.transforms(image=image)['image']
        # image = self.transforms(image)

        return image.cuda()

    def __len__(self):
        if len(self.c_paths) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.c_paths)

    @staticmethod
    def get_id_labels(data_dir):
        image_fns = glob.glob(os.path.join(data_dir, '*'))
        label_names = [os.path.split(s)[-1] for s in image_fns]
        unique_labels = list(set(label_names))
        unique_labels.sort()
        id_labels = {_id: name for name, _id in enumerate(unique_labels)}

        return id_labels


    @staticmethod
    def pading(size,img):
        padding_v = tuple([125, 125, 125])

        w, h = img.size

        target_size = size

        interpolation = Image.BILINEAR

        if w > h:
            img = img.resize((int(target_size), int(h * target_size * 1.0 / w)), interpolation)
        else:
            img = img.resize((int(w * target_size * 1.0 / h), int(target_size)), interpolation)

        ret_img = Image.new("RGB", (target_size, target_size), padding_v)
        w, h = img.size
        st_w = int((ret_img.size[0] - w) / 2.0)
        st_h = int((ret_img.size[1] - h) / 2.0)
        ret_img.paste(img, (st_w, st_h))
        # ret_img = np.array(ret_img)
        # cv2.imshow("show",ret_img)
        # cv2.waitKey(0)
        return ret_img

# if __name__=='__main__':
#     print(timm.list_models())




if __name__ == '__main__':

    print(torch.cuda.is_available())
    #是使用GPU还是CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class CFG:
        # step1: hyper-parameter超参数
        seed = 42
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #选择使用设备
        ckpt_fold = "model"   #保存模型路径
        ckpt_name = "mixnet_l_no"  # for submit.
        img_class_dir = r"E:\train_data\wrinkle_data"   #篡改图片路径

        # step2: data
        n_fold = 5    #交叉验证得次数
        # img_size = [224, 224]   # 图片尺寸
        img_size = [180, 180]   # 图片尺寸
        train_bs = 32 # bachsize
        valid_bs = train_bs
        log_interval = 10
        # step3: model
        backbone = 'mixnet_xl'
        num_classes = 4
        # step4: optimizer
        epoch = 15
        lr = 1e-3
        wd = 5e-2
        # lr_drop = 8
        # step5: infer
        thr = 0.5

    set_seed(CFG.seed)
    # 创建保存模型得路径
    ckpt_path = f"./{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # 加载未篡改得图片路径和篡改图片得路径并合并
    #匹配所有符合条件的文件，并将其以list的形式返回
    train_all_path = sorted(glob.glob(CFG.img_class_dir +"/*/*.jpg"))
    # print(type(train_all_path))
    random.seed(CFG.seed)
    random.shuffle(train_all_path)
    # print(train_all_path)

    #分割训练集和测试集，把原始数据分割为K个子集，每次会将其中一个子集作为测试集，其余K-1个子集作为训练集。
    kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    logger = get_logger(os.path.join(CFG.ckpt_fold, CFG.ckpt_name + '.log'))
    logger.info('Using: {}'.format(CFG.ckpt_name))

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_all_path)):
        print(f'###### Fold: {fold}', flush=True)
        # if fold != 0 :
        #     continue
        trian_path = [train_all_path[i] for i in train_idx]
        val_path = [train_all_path[i] for i in val_idx]


        #创建一个有预训练权重的模型，具有num_classes个分类
        model = timm.create_model(CFG.backbone,
                                  pretrained=True,
                                  num_classes=CFG.num_classes )


        model.to(CFG.device)
        # 随机权重平均SWA,实现更好的泛化
        swa_model = AveragedModel(model)

        train_data = lwhdataset(data_dir=trian_path, train_transform=train_aug, size=CFG.img_size[0], pad=True)
        valid_data = lwhdataset(data_dir=val_path ,train_transform=val_aug, size=CFG.img_size[0], pad=True)
        valid_data_swa = lwhdataset_swa(data_dir=val_path ,train_transform=val_aug, size=CFG.img_size[0], pad=True)

        train_loader = DataLoader(dataset=train_data, batch_size=CFG.train_bs, shuffle=True, num_workers=10,drop_last = True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=CFG.valid_bs, shuffle=False, num_workers=8)
        valid_swa_loader = DataLoader(dataset=valid_data_swa, batch_size=CFG.train_bs, shuffle=False, num_workers=8)
        # x = [3.5, 1]
        # weight = torch.Tensor(x).to("cuda:0")
        # criterion = torch.nn.CrossEntropyLoss(weight=weight)
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.CrossEntropyLoss()
        # criterion_cent = common.CenterLoss_new(9, 384).to(device)
        # criterion = LabelSmoothingLoss(classes = 9, smoothing=0.1)
        # criterion = focal_loss()

#0.0008,0.96
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=CFG.wd)
        # optimizer =  AdamP(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=CFG.wd)
        # optimizer = optim.Adahessian(model.parameters(), lr=0.00005, weight_decay=CFG.wd)
        # optimizer = optim.Adahessian(model.parameters(),lr=0.01,betas=(0.9, 0.999),eps = 1e-4, weight_decay = 0.0,hessian_power = 1.0,)
        # optimizer_centerloss = AdamP(criterion_cent.parameters(), lr=0.00005, weight_decay=CFG.wd)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)#0.95
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0, last_epoch=-1)
        swa_start = 8
        swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=3, swa_lr=0.00005)

        best_val_acc = 0
        best_trp_score = 0
        step = 0

        # for epoch in range(1+6, CFG.epoch + 1):
        for epoch in range(1, CFG.epoch + 1):
            model.train()
            loss_mean = 0.
            correct = 0.
            total = 0.
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(CFG.device, dtype=torch.float)  # [b, c, w, h]
                labels = labels.to(CFG.device)


                optimizer.zero_grad()


                y_preds = model(images)

                loss = criterion(y_preds, labels)

                loss.backward()
                optimizer.step()


                _, predicted = torch.max(y_preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).squeeze().cpu().sum().numpy()

                loss_mean += loss.item()
                current_lr = optimizer.param_groups[0]['lr']

                if (i + 1) % CFG.log_interval == 0:
                    loss_mean = loss_mean / CFG.log_interval
                    logger.info("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} current_lr:{:.5f}".format(
                        epoch, CFG.epoch + 1, i + 1, len(train_loader), loss_mean, correct / total,current_lr))

                    step += 1
                    loss_mean = 0.
            if epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
            gc.collect()
            pres_list = []
            labels_list = []
            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()
                    pres_list += predicted.cpu().numpy().tolist()
                    labels_list += labels.data.cpu().numpy().tolist()

                loss_val_mean = loss_val / len(valid_loader)

                _, _, f_class, _ = precision_recall_fscore_support(y_true=labels_list, y_pred=pres_list,
                                                                   labels=[id for id, name in enumerate(unique_labels)],
                                                                   average=None)
                confusion = confusion_matrix(labels_list, pres_list, labels=None, sample_weight=None)
                # logger.info("confusion_matrix: ",confusion)
                fper_class = {name: "{:.2%}".format(f_class[_id]) for _id, name in enumerate(unique_labels)}
                logger.info('clssse_F1:{}  class_F1_average:{:.2%}'.format(fper_class, f_class.mean()))
                logger.info("Valid_acc:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} ".format(
                    epoch, CFG.epoch + 1, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val))

                if correct_val / total_val > best_val_acc:
                    best_val_acc = correct_val / total_val
                    save_path = f"{ckpt_path}/best_fold{fold}.pth"
                    torch.save(model, save_path)


                logger.info("best_acc_score:\t best_acc:{:.2%}".format(best_val_acc))
            gc.collect()
        torch.optim.swa_utils.update_bn(valid_swa_loader, swa_model)
        torch.save(swa_model.state_dict(), f"{ckpt_path}/best_fold{fold}_swa.pth")
        gc.collect()

        logger.info('stop training...')






