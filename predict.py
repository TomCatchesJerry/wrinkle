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
print(224//20)