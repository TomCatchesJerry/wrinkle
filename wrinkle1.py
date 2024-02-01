import cv2 as cv
import cv2

import time
import numpy as np
from skimage.filters import frangi, gabor,hessian,meijering,sato
from skimage import measure, morphology,filters
import detect_face as dface

#本文件与后端皱纹检测算法同步

def testGabor(image):
    '''
    gabor滤波器对横向皱纹使用的参数frequency=0.085，theta=1.57
    theta与皱纹的方向有关，需要与皱纹方向一致
    @param image : 额头的图片
    @return: 皱纹列表,皱纹严重分数（0-100),越高越严重
    '''

    # cv2.namedWindow("result", cv2.WINDOW_KEEPRATIO)
    i_shape = image.shape
    image_o = image.copy()
    # image_w=1000
    image_w = i_shape[1]
    # image_h=300
    image_h = i_shape[0]
    # g=cv2.split(image)[0]
    # image = cv.resize(image, (image_w, image_h))
    # filter_w = image.shape[1] / 200
    filter_w = image_w / 12.5
    # print(image.shape)
    image1 = image.copy()
    g = cv2.split(image1)[0]
    # frequency谐波函数的空间频率，以像素来表示,越小越能展示细节，theta滤波器方向
    # cv2.imshow("result", g)
    # cv2.waitKey(0)
    g = cv.blur(g, (3, 3))
    # cv2.imshow("result", g)
    # cv2.waitKey(0)

    g = g.astype(float)
    # max_v = np.max(g)
    # min_v = np.min(g)
    # g[:, :] = (g[:, :] - min_v) * 255.0 / (max_v - min_v)
    mean_v = np.mean(g)
    factor = 0.4
    g = g.astype(float)
    g[:, :] = (g[:, :] - mean_v) * factor + g[:, :]
    g[g > 255] = 255
    g[g < 0] = 0
    g = g.astype(np.uint8)
    # cv2.imshow("result", g)
    # cv2.waitKey(0)
    mask_g = g.copy()
    mask_g[:, :] = 0
    # g = cv.resize(g, (image_w, image_h))
    # 比较有用的参数修改sigmas，gama
    # sk_frangi_img2 = frangi(g, sigmas = (3, 10, 2.5), alpha = 0.5, beta = 0.95, gamma = 1.2)
    sk_frangi_img = g.copy()
    sk_frangi_img[:, :] = 0
    for the in (1.27, 1.57, 1.87):

        sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.084, theta = the, sigma_x = 6.2)  # 1.77,1.37
        # sk_gabor_img_2, sk_gabor_2 = gabor(g, frequency = 0.084, theta = 1.27, sigma_x = 6.2)#1.77,1.37
        # sk_gabor_img_3, sk_gabor_3 = gabor(g, frequency = 0.084, theta = 1.87, sigma_x = 6.2)#1.77,1.37
        sk_frangi_img2 = sk_gabor_img_1
        # cv2.imshow("result1", sk_frangi_img2)
        # cv2.waitKey(0)
        # cv2.imshow("result2", sk_gabor_img_2)
        # cv2.waitKey(0)
        # cv2.imshow("result3", sk_gabor_img_3)
        # cv2.waitKey(0)
        thres = 30
        sk_frangi_img2[sk_frangi_img2 > thres] = 255
        # sk_gabor_img_2[sk_gabor_img_2 > 30] = 255
        # sk_gabor_img_3[sk_gabor_img_3 > 30] = 255
        sk_frangi_img2[sk_frangi_img2 <= thres] = 0
        # sk_frangi_img2[sk_gabor_img_2==255]=255
        # sk_frangi_img2[sk_gabor_img_3==255]=255

        # cv2.imshow("result", sk_frangi_img2)
        # cv2.waitKey(0)
        image1[sk_frangi_img2 == 255] = 255
        # cv2.imshow("result1", image1)
        # cv2.waitKey(0)
        # sk_gabor_img_1=sk_frangi_img2
        # sk_gabor_img_1[sk_gabor_img_1 < 30] = 0
        # 通过偏心率（>0.98）,连通区域面积来过滤一些非皱纹
        image_w = image.shape[1]
        bool_img = sk_frangi_img2 > 30
        label_image = measure.label(bool_img)
        count = 0
        image_xcs = []
        # thres=image.shape[0]*image.shape[1]/8000
        for region in measure.regionprops(label_image):
            image_xc = np.zeros((image_w, 2))
            # if region.area < thres:  # or region.area > 700
            #     x = region.coords
            #     for i in range(len(x)):
            #         sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            #     continue
            # print(region.axis_major_length)
            if region.axis_major_length < filter_w:
                x = region.coords
                for i in range(len(x)):
                    sk_frangi_img2[x[i][0]][x[i][1]] = 0
                continue
            # print(region.feret_diameter_max)
            # print(region.axis_major_length)
            # print(region.orientation)
            if region.orientation > -np.pi / 3 and region.orientation < np.pi / 3:
                x = region.coords
                for i in range(len(x)):
                    sk_frangi_img2[x[i][0]][x[i][1]] = 0
                continue
            if region.eccentricity > 0.98:
                count += 1
            else:
                x = region.coords
                for i in range(len(x)):
                    sk_frangi_img2[x[i][0]][x[i][1]] = 0
                continue
            x = region.coords
            for i in range(len(x)):
                image_xc[x[i][1], 0] += 1
                image_xc[x[i][1], 1] += x[i][0]
            image_xc[image_xc[:, 0] == 0, 0] = 1
            # print(image_xc)
            image_xc[:, 1] = image_xc[:, 1] / image_xc[:, 0]
            image_xcs.append(image_xc.copy())
        sk_frangi_img[(sk_frangi_img == 255) | (sk_frangi_img2 == 255)] = 255
        wrikle_list = []
        for lis in image_xcs:
            wrikle = []
            for i in range(len(lis)):
                if lis[i][1] > 1:
                    wrikle.append((i, lis[i][1]))
                    # image1[int(lis[i][1]), i] = 255
            # if len(wrikle) == 0:
            #     continue
            wrikle_list.append(wrikle.copy())
        wrikle_list_final = []
        sk_frangi_img2[:, :] = 0
        for i in range(len(wrikle_list)):
            # if i in delete_list:
            #     continue

            wrik = wrikle_list[i]
            # if len(wrik) < filter_w:
            #     continue
            wrikle_list_final.append(wrik)
            for (x, y) in wrik:
                if x < 1 and y < 1:
                    continue

                if int(y) - 1 > 0:
                    mask_g[int(y) - 1, int(x)] = 255
                # if int(y)-2>0:
                #     image3[int(y)-2,int(x)]=255
                mask_g[int(y), int(x)] = 255
                if int(y) + 1 < image_h:
                    mask_g[int(y) + 1, int(x)] = 255

    wrinkle_count = 0
    # for wrinkle in wrikle_list:
    #     wrinkle_count += len(wrinkle)
    # print(wrinkle_count/400/500.0)
    # cv2.imshow("r1", mask_g)
    # cv2.waitKey(0)
    # wrinkle_count = np.count_nonzero(mask_g)
    wrinkle_count = np.count_nonzero(sk_frangi_img)
    wrinkle_level = wrinkle_count / (image.shape[0] * image.shape[1]) / 0.11 * 100
    # print(wrinkle_count / (image.shape[0] * image.shape[1]))
    if wrinkle_level > 100:
        wrinkle_level = 100
    # mask_g = cv2.resize(mask_g, (i_shape[1], i_shape[0]))
    # sk_frangi_img = cv2.resize(sk_frangi_img, (i_shape[1], i_shape[0]))
    # print(image_o.shape)
    # print(image3.shape)
    image_o[mask_g > 100] = 255
    return image_o, 100 - int(wrinkle_level)

if __name__ == '__main__':

    path = r"E:\forehead\3.jpg"
    path = r"E:\faces\8(1).png"
    # path = r"E:\SU\sources\Wrinkles_detection\wrinkle\muouwen\8.PNG"
    # # path = r"E:\SU\sources\Wrinkles_detection\realtest\2.jpg"
    image = cv2.imread(path, -1)

    image1 ,_ = testGabor(image)
    cv2.namedWindow("show", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("show", image1)
    cv2.waitKey(0)