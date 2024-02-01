import cv2
import torch
import numpy as np
import os
from PIL import Image


def softmax14(arr):
    arr=np.power(1.41,arr)
    arr_s=np.sum(arr)
    arr=arr/arr_s
    return arr
class WrinkleClassifation(object):
    '''
    额头皱纹打分，最底下有使用示例
    '''
    def __init__(self,video=True,size = 180):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.model = torch.load(os.path.join(os.getcwd(),"model/mixnet_l_no","best_fold0.pth")).to(self.device)
        # torch.save(self.model,"model/eyesglasses.pt")
        # self.model = torch.load(r"D:\lwh\python_project\eyeglasses\model\mixnet_s_2\best_fold4.pth").to(self.device)
        self.model.eval()
        self.video = video
        self.size = size
        self.score_list = np.array([100, 80, 50, 15])

    def __call__(self, img):

        input_tensor = self.transform(img)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # print(outputs)
            wrinkle_type = outputs[0]
            # print(pre_cls_cou)
            # pre_cls_cou=torch.nn.functional.softmax(pre_cls_cou,dim = 0)
            # print(pre_cls_cou)
            wrinkle_type = wrinkle_type.to("cpu").numpy()
            # print(pre_cls_cou)
            wrinkle_type=softmax14(wrinkle_type)
            # print(pre_cls_cou)
            # score = np.array([element * self.score_list[index] for index, element in enumerate(pre_cls_cou)])

            score=wrinkle_type*self.score_list
            score = np.sum(score)
            # print(pre_cls_cou)

        return int(score), np.argmax(wrinkle_type)



    def transform(self,img):

        if self.video:
            image = img
        else:
            image = cv2.imread(img)

        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        img_pad = self.padding(self.size,image_pil)
        img_nor = self.np_nor(img_pad)
        torch_input = torch.from_numpy(img_nor).to(self.device)
        return torch_input

    def padding(self,size,img):
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

        return ret_img

    def np_nor(self,image):
        # image.show()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        std = np.array(std, dtype=np.float32)
        std *= 255.0
        mean *= 255.0
        denominator = np.reciprocal(std, dtype=np.float32)
        image = np.array(image).astype(np.float32)
        image -= mean
        image *= denominator
        # im = image.astype(np.uint8)
        # cv2.imshow("show", im)
        # cv2.waitKey(0)
        image = np.rollaxis(image, 2, 0)

        image = image[np.newaxis, :, :, :]


        return image


import cv2
import time
import glob
if __name__=="__main__":

    # WrinkleClassifation = WrinkleClassifation(video=True)
    # train_all_path = glob.glob(r"E:\train_data\wrinkle_data\*" +"/*.jpg")
    # for path in train_all_path:
    # # path=r"E:\train_data\wrinkle_data\0\86.jpg"
    #     image_org = cv2.imread(path)
    #     wrinkle_score = WrinkleClassifation(image_org)
    #     # print(path)
    #     print(wrinkle_score)

    #使用示例，输入一张额头图片，图片更换为待检测的额头图片路径
    path = r"E:\train_data\wrinkle_data\0\86.jpg"
    image_org = cv2.imread(path)
    #初始化皱纹打分类
    WrinkleClassifation = WrinkleClassifation(video = True)
    #计算皱纹得分，分数0-100，分数越高，皱纹越轻
    wrinkle_score = WrinkleClassifation(image_org)
    print(wrinkle_score)