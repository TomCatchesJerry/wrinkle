import cv2 as cv
import cv2

import time
import numpy as np
from skimage.filters import frangi, gabor,hessian,meijering,sato
from skimage import measure, morphology,filters
import detect_face as dface
#算法试写、试调用的，合适了的算法需要移到wrinkle_detect.py文件

def testGabor(image):
    '''
    gabor滤波器对横向皱纹使用的参数frequency=0.085，theta=1.57
    theta与皱纹的方向有关，需要与皱纹方向一致
    @param image : 额头的图片
    @return: 皱纹列表,皱纹严重分数（0-100),越高越严重
    '''

    cv2.namedWindow("result",cv2.WINDOW_KEEPRATIO)
    i_shape = image.shape
    image_o=image.copy()
    image = cv.resize(image, (500, 150))
    filter_w = image.shape[1] / 10
    # print(image.shape)
    image1 = image.copy()
    image2 = image.copy()
    image3 = image.copy()
    g = cv2.split(image1)[1]
    # frequency谐波函数的空间频率，以像素来表示,越小越能展示细节，theta滤波器方向

    # g=cv.blur(g,(10,4))
    # g = cv.blur(g, (5, 5))
    # cv2.imshow("result", g)
    # cv2.waitKey(0)
    # g = cv.medianBlur(g, 3)
    g = cv.blur(g, (3, 3))
    # cv2.imshow("result", g)
    # cv2.waitKey(0)
    sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.084, theta = 1.57,sigma_x =6.2)#sigma_x =6.1,sigma_y = 0.12
    # sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.087, theta = 1.57, sigma_x = 6.1, sigma_y = 0.15)
    # sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.09, theta = 1.57,bandwidth = 1.5)
    # sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.09, theta = 1.57,bandwidth = 1.3) #用这个的话，虽说检测会更全一点，但误检较多
    cv2.imshow("result",sk_gabor_img_1)
    cv2.waitKey(0)
    image1[sk_gabor_img_1>30]=255
    # cv2.imshow("result", image1)
    # cv2.waitKey(0)
    # cv2.imshow("result", sk_gabor_1)
    # cv2.waitKey(0)
    # sk_gabor_img_1=sk_gabor_1
    # return sk_gabor_1,1
    # image1[sk_gabor_1 > 30] = 255
    # cv2.imshow("result", image1)
    # cv2.waitKey(0)
    # sk_gabor_img_1=sk_gabor_1
    sk_gabor_img_1[sk_gabor_img_1 < 30] = 0
    # 通过偏心率（>0.98）,连通区域面积来过滤一些非皱纹
    image_w = image.shape[1]
    bool_img = sk_gabor_img_1 > 30
    label_image = measure.label(bool_img)
    count = 0
    image_xcs = []
    for region in measure.regionprops(label_image):
        image_xc = np.zeros((image_w, 2))
        if region.area < 150:  # or region.area > 700
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
        # print(region.orientation)
        if region.orientation>-np.pi/2.5 and region.orientation<np.pi/2.5:
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
        if region.eccentricity > 0.98:
            count += 1
        else:
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
        x = region.coords
        for i in range(len(x)):
            image_xc[x[i][1], 0] += 1
            image_xc[x[i][1], 1] += x[i][0]
        image_xc[image_xc[:, 0] == 0, 0] = 1
        # print(image_xc)
        image_xc[:, 1] = image_xc[:, 1] / image_xc[:, 0]
        image_xcs.append(image_xc.copy())

    wrikle_list = []
    for lis in image_xcs:
        wrikle = []
        for i in range(len(lis)):
            if lis[i][1] > 1:
                wrikle.append((i, lis[i][1]))
                image1[int(lis[i][1]), i] = 255
        if len(wrikle) == 0:
            continue
        wrikle_list.append(wrikle.copy())

    # 存放皱纹像素(x,y)列表的列表
    wrikle_list.sort(key = len, reverse = True)
    remove_list = []
    wrikle_list_ab = np.zeros((len(wrikle_list), 2, 2))
    for i in range(len(wrikle_list)):
        wrikle_list_ab[i, 0] = wrikle_list[i][0]
        wrikle_list_ab[i, 1] = wrikle_list[i][-1]

    # print(wrikle_list_ab)
    for i in range(len(wrikle_list) - 1):
        j = i + 1
        while j < len(wrikle_list):
            if wrikle_list_ab[j, 0, 0] > wrikle_list_ab[i, 0, 0] - 1 and wrikle_list_ab[j, 1, 0] < wrikle_list_ab[
                i, 1, 0] + 1 and \
                    abs(wrikle_list_ab[i, 0, 1] - wrikle_list_ab[j, 0, 1]) < 10 and abs(wrikle_list_ab[i, 1, 1] - wrikle_list_ab[j, 1, 1]) < 10:
                remove_list.append(j)
            j += 1
    remove_list.sort(reverse = True)
    # print(remove_list)
    remove_list = np.asarray(remove_list)
    remove_list = np.unique(remove_list)
    remove_list = remove_list[::-1]
    # print(remove_list)
    for i in remove_list:
        wrikle_list.pop(i)
        wrikle_list_ab = np.delete(wrikle_list_ab, i, axis = 0)

    # print(wrikle_list_ab)
    for wrik in wrikle_list:
        for (x, y) in wrik:
            image2[int(y), int(x)] = 255
        # cv2.imshow("result", image2)
        # cv2.waitKey(0)

    len1 = len(wrikle_list)

    #把链接皱纹的算法去掉
    # delete_list = []
    # for i in range(len1):
    #     j = i + 1
    #     while (j < len1):
    #         r1 = (wrikle_list_ab[i, 0, 0] - wrikle_list_ab[j, 1, 0]) ** 2 + (
    #                 wrikle_list_ab[i, 0, 1] - wrikle_list_ab[j, 1, 1]) ** 2
    #         if r1 < 300:
    #             wrikle_list_ab[i, 0] = wrikle_list_ab[j, 0]
    #             wrikle_list[i] = wrikle_list[i] + wrikle_list[j]
    #             delete_list.append(j)
    #         r2 = (wrikle_list_ab[i, 1, 0] - wrikle_list_ab[j, 0, 0]) ** 2 + (
    #                 wrikle_list_ab[i, 1, 1] - wrikle_list_ab[j, 0, 1]) ** 2
    #         if r2 < 300:
    #             wrikle_list_ab[i, 1] = wrikle_list_ab[j, 1]
    #             wrikle_list[i] = wrikle_list[i] + wrikle_list[j]
    #             delete_list.append(j)
    #         j += 1
    wrikle_list_final = []
    image3[:,:]=0
    for i in range(len(wrikle_list)):
        # if i in delete_list:
        #     continue

        wrik = wrikle_list[i]
        if len(wrik) < filter_w:
            continue
        wrikle_list_final.append(wrik)
        for (x, y) in wrik:
            if x < 1 and y < 1:
                continue

            if int(y) - 1 > 0:
                image3[int(y) - 1, int(x)] = 255
            # if int(y)-2>0:
            #     image3[int(y)-2,int(x)]=255
            image3[int(y), int(x)] = 255
            if int(y) + 1 < 150:
                image3[int(y) + 1, int(x)] = 255

    wrinkle_count = 0
    for wrinkle in wrikle_list:
        wrinkle_count += len(wrinkle)
    # print(wrinkle_count/400/500.0)
    wrinkle_level = wrinkle_count / (image.shape[0] * image.shape[1]) / 0.09 * 100
    # print(wrinkle_count/(image.shape[0]*image.shape[0]))
    if wrinkle_level > 100:
        wrinkle_level = 100
    image3 = cv2.resize(image3, (i_shape[1], i_shape[0]))
    # print(image_o.shape)
    # print(image3.shape)
    image_o[image3>100]=255
    return image_o, 100 - int(wrinkle_level)
def testGabor1(image):
    '''

    多尺度检测
    1000*400 and 500*150
    大尺度的情况下可以检测出教轻微的皱纹
    gabor滤波器对横向皱纹使用的参数frequency=0.085，theta=1.57
    theta与皱纹的方向有关，需要与皱纹方向一致
    @param image : 额头的图片
    @return: 皱纹列表,皱纹严重分数（0-100),越高越严重
    '''

    cv2.namedWindow("result",cv2.WINDOW_KEEPRATIO)
    i_shape = image.shape
    image_o=image.copy()
    # image_w=1000
    image_w=i_shape[1]
    # image_h=400
    image_h=i_shape[0]

    image = cv.resize(image, (image_w, image_h))
    # filter_w = image.shape[1] / 200
    filter_w = image_w / 12.5
    # print(image.shape)
    image1 = image.copy()
    g = cv2.split(image1)[0]
    # frequency谐波函数的空间频率，以像素来表示,越小越能展示细节，theta滤波器方向
    g = cv.medianBlur(g, 3)



    # g = cv.blur(g, (3, 3))
    cv2.imshow("result", g)
    cv2.waitKey(0)

    g = g.astype(float)
    max_v = np.max(g)
    min_v = np.min(g)
    g[:, :] = (g[:, :] - min_v) * 255.0 / (max_v - min_v)

    g[g > 255] = 255
    g[g < 0] = 0
    g = g.astype(np.uint8)
    cv2.imshow("result", g)
    cv2.waitKey(0)
    #比较有用的参数修改sigmas，gama
    sk_frangi_img2 = frangi(g, sigmas = (3, 10, 2.5), alpha = 0.5, beta = 0.95, gamma = 1.2)
    cv2.imshow("result1", sk_frangi_img2)
    cv2.waitKey(0)
    thres=0.10
    sk_frangi_img2[sk_frangi_img2 > thres] = 255
    sk_frangi_img2[sk_frangi_img2 <= thres] = 0
    cv2.imshow("result", sk_frangi_img2)
    cv2.waitKey(0)
    image1[sk_frangi_img2==255]=255
    cv2.imshow("result1", image1)
    cv2.waitKey(0)
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
        print(region.feret_diameter_max)
        print(region.axis_major_length)
        # print(region.orientation)
        if region.orientation>-np.pi/3 and region.orientation<np.pi/3:
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
    sk_frangi_img=sk_frangi_img2.copy()
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
    sk_frangi_img2[:,:]=0
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
                sk_frangi_img2[int(y) - 1, int(x)] = 255
            # if int(y)-2>0:
            #     image3[int(y)-2,int(x)]=255
            sk_frangi_img2[int(y), int(x)] = 255
            if int(y) + 1 < image_h:
                sk_frangi_img2[int(y) + 1, int(x)] = 255

    wrinkle_count = 0
    for wrinkle in wrikle_list:
        wrinkle_count += len(wrinkle)
    # print(wrinkle_count/400/500.0)
    wrinkle_count=np.count_nonzero(sk_frangi_img2)
    wrinkle_level = wrinkle_count / (image.shape[0] * image.shape[1]) / 0.13 * 100
    print(wrinkle_count/(image.shape[0]*image.shape[1]))
    if wrinkle_level > 100:
        wrinkle_level = 100
    sk_frangi_img2 = cv2.resize(sk_frangi_img2, (i_shape[1], i_shape[0]))
    sk_frangi_img = cv2.resize(sk_frangi_img, (i_shape[1], i_shape[0]))
    # print(image_o.shape)
    # print(image3.shape)
    image_o[sk_frangi_img2>100]=255
    return image_o, 100 - int(wrinkle_level)
def testGabor2(image):
    '''
    gabor滤波器对横向皱纹使用的参数frequency=0.085，theta=1.57
    theta与皱纹的方向有关，需要与皱纹方向一致
    @param image : 额头的图片
    @return: 皱纹列表,皱纹严重分数（0-100),越高越严重
    '''

    cv2.namedWindow("result",cv2.WINDOW_KEEPRATIO)
    i_shape = image.shape
    image_o=image.copy()
    # image_w=i_shape[1]
    image_w=500
    # image_h=i_shape[0]
    image_h=150
    image = cv.resize(image, (image_w, image_h))
    filter_w = image_w / 12.5
    # print(image.shape)
    image1 = image.copy()
    # image2 = image.copy()
    # image3 = image.copy()
    g = cv2.split(image1)[0]
    # g = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    # frequency谐波函数的空间频率，以像素来表示,越小越能展示细节，theta滤波器方向
    # g = cv.blur(g, (3, 3))
    cv2.imshow("result", g)
    cv2.waitKey(0)

    #做些图像增强

    # g=cv2.equalizeHist(g)

    g = g.astype(float)
    max_v=np.max(g)
    min_v=np.min(g)
    g[:,:]=(g[:,:]-min_v)*255.0/(max_v-min_v)
    # g=g.astype(np.uint8)

    # mean_v=np.mean(g)
    # factor=0.5
    # g=g.astype(float)
    # g[:,:]=(g[:,:]-mean_v)*factor+g[:,:]
    g[g>255]=255
    g[g<0]=0
    g=g.astype(np.uint8)

    cv2.imshow("result3", g)
    cv2.waitKey(0)
    g1 = cv2.split(image1)[1]
    cv2.imshow("result1", g1)
    cv2.waitKey(0)
    g2 = cv2.split(image1)[2]
    cv2.imshow("result2", g2)
    cv2.waitKey(0)
    sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.084, theta = 1.57,sigma_x =6.2)#sigma_x =6.1,sigma_y = 0.12
    cv2.imshow("result",sk_gabor_img_1)
    cv2.waitKey(0)
    image1[sk_gabor_img_1>30]=255
    cv2.imshow("result", image1)
    cv2.waitKey(0)
    sk_frangi_img2 = frangi(g, sigmas = (3, 10, 2.5), alpha = 0.5, beta = 0.95, gamma = 1.2)
    cv2.imshow("result1", sk_frangi_img2)
    cv2.waitKey(0)
    # cv2.imshow("result", sk_gabor_1)
    # cv2.waitKey(0)
    # sk_gabor_img_1=sk_gabor_1
    # return sk_gabor_1,1
    # image1[sk_gabor_1 > 30] = 255
    # cv2.imshow("result", image1)
    # cv2.waitKey(0)
    # sk_gabor_img_1=sk_gabor_1
    sk_gabor_img_1[sk_gabor_img_1 < 30] = 0
    # 通过偏心率（>0.98）,连通区域面积来过滤一些非皱纹
    image_w = image.shape[1]
    bool_img = sk_gabor_img_1 > 30
    label_image = measure.label(bool_img)
    count = 0
    image_xcs = []
    for region in measure.regionprops(label_image):
        image_xc = np.zeros((image_w, 2))
        if region.axis_major_length < filter_w:
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
        # print(region.orientation)
        if region.orientation>-np.pi/2.5 and region.orientation<np.pi/2.5:
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
        if region.eccentricity > 0.98:
            count += 1
        else:
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
        x = region.coords
        for i in range(len(x)):
            image_xc[x[i][1], 0] += 1
            image_xc[x[i][1], 1] += x[i][0]
        image_xc[image_xc[:, 0] == 0, 0] = 1
        # print(image_xc)
        image_xc[:, 1] = image_xc[:, 1] / image_xc[:, 0]
        image_xcs.append(image_xc.copy())

    wrikle_list = []
    for lis in image_xcs:
        wrikle = []
        for i in range(len(lis)):
            if lis[i][1] > 1:
                wrikle.append((i, lis[i][1]))
                image1[int(lis[i][1]), i] = 255
        if len(wrikle) == 0:
            continue
        wrikle_list.append(wrikle.copy())

    # 存放皱纹像素(x,y)列表的列表
    wrikle_list.sort(key = len, reverse = True)
    remove_list = []
    wrikle_list_ab = np.zeros((len(wrikle_list), 2, 2))
    for i in range(len(wrikle_list)):
        wrikle_list_ab[i, 0] = wrikle_list[i][0]
        wrikle_list_ab[i, 1] = wrikle_list[i][-1]

    # print(wrikle_list_ab)
    for i in range(len(wrikle_list) - 1):
        j = i + 1
        while j < len(wrikle_list):
            if wrikle_list_ab[j, 0, 0] > wrikle_list_ab[i, 0, 0] - 1 and wrikle_list_ab[j, 1, 0] < wrikle_list_ab[
                i, 1, 0] + 1 and \
                    abs(wrikle_list_ab[i, 0, 1] - wrikle_list_ab[j, 0, 1]) < 10 and abs(wrikle_list_ab[i, 1, 1] - wrikle_list_ab[j, 1, 1]) < 10:
                remove_list.append(j)
            j += 1
    remove_list.sort(reverse = True)
    # print(remove_list)
    remove_list = np.asarray(remove_list)
    remove_list = np.unique(remove_list)
    remove_list = remove_list[::-1]
    # print(remove_list)
    for i in remove_list:
        wrikle_list.pop(i)
        wrikle_list_ab = np.delete(wrikle_list_ab, i, axis = 0)

    # print(wrikle_list_ab)
    sk_gabor_img_1[:,:]=0
    for wrik in wrikle_list:
        for (x, y) in wrik:
            sk_gabor_img_1[int(y), int(x)] = 255
        # cv2.imshow("result", image2)
        # cv2.waitKey(0)

    len1 = len(wrikle_list)


    wrikle_list_final = []
    # image3[:,:]=0
    for i in range(len(wrikle_list)):
        # if i in delete_list:
        #     continue

        wrik = wrikle_list[i]
        if len(wrik) < filter_w:
            continue
        wrikle_list_final.append(wrik)
        for (x, y) in wrik:
            if x < 1 and y < 1:
                continue

            if int(y) - 1 > 0:
                sk_gabor_img_1[int(y) - 1, int(x)] = 255
            # if int(y)-2>0:
            #     image3[int(y)-2,int(x)]=255
            sk_gabor_img_1[int(y), int(x)] = 255
            if int(y) + 1 < image_h:
                sk_gabor_img_1[int(y) + 1, int(x)] = 255

    wrinkle_count = 0
    for wrinkle in wrikle_list:
        wrinkle_count += len(wrinkle)
    # print(wrinkle_count/400/500.0)
    wrinkle_level = wrinkle_count / (image.shape[0] * image.shape[1]) / 0.09 * 100
    # print(wrinkle_count/(image.shape[0]*image.shape[0]))
    if wrinkle_level > 100:
        wrinkle_level = 100
    image3 = cv2.resize(sk_gabor_img_1, (i_shape[1], i_shape[0]))
    # print(image_o.shape)
    # print(image3.shape)
    image_o[image3>100]=255
    return image_o, 100 - int(wrinkle_level)


def testGabor3(image):
    '''


    gabor滤波器对横向皱纹使用的参数frequency=0.085，theta=1.57
    theta与皱纹的方向有关，需要与皱纹方向一致
    @param image : 额头的图片
    @return: 皱纹列表,皱纹严重分数（0-100),越高越严重
    '''

    cv2.namedWindow("result",cv2.WINDOW_KEEPRATIO)
    i_shape = image.shape
    image_o=image.copy()
    # image_w=1000
    image_w=i_shape[1]
    # image_h=300
    image_h=i_shape[0]
    # g=cv2.split(image)[0]
    # image = cv.resize(image, (image_w, image_h))
    # filter_w = image.shape[1] / 200
    filter_w = image_w / 12.5
    # print(image.shape)
    image1 = image.copy()
    # image1=cv.cvtColor(image1,cv.COLOR_BGR2HSV)
    g = cv2.split(image1)[0]
    # frequency谐波函数的空间频率，以像素来表示,越小越能展示细节，theta滤波器方向
    cv2.imshow("result", g)
    cv2.waitKey(0)
    # g = cv.blur(g, (3, 3))

    # g = cv2.equalizeHist(g)
    # clahe = cv.createCLAHE(clipLimit = 3, tileGridSize = (100, 100))
    #
    # g = clahe.apply(g)
    #


    g = g.astype(float)

    # max_v = np.max(g)
    # min_v = np.min(g)
    # g[:, :] = (g[:, :] - min_v) * 255.0 / (max_v - min_v)

    mean_v = np.mean(g)
    #0.4可能会过检
    factor=0.3

    g=g.astype(float)
    g[:,:]=(g[:,:]-mean_v)*factor+g[:,:]

    g[g > 255] = 255
    g[g < 0] = 0
    g = g.astype(np.uint8)
    cv2.imshow("result", g)
    cv2.waitKey(0)

    # g = cv.medianBlur(g,3)
    # cv2.imshow("result", g)
    # cv2.waitKey(0)

    mask_g=g.copy()
    mask_g[:,:]=0
    # g = cv.resize(g, (image_w, image_h))
    #比较有用的参数修改sigmas，gama
    # sk_frangi_img2 = frangi(g, sigmas = (3, 10, 2.5), alpha = 0.5, beta = 0.95, gamma = 1.2)
    # for the in (0.97,1.27,1.57,1.87,2.17):
    sk_frangi_img = g.copy()
    sk_frangi_img[:, :] = 0
    for the in (1.27,1.57,1.87):

        sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.084, theta = the, sigma_x = 6.2)#1.77,1.37
        # sk_gabor_img_2, sk_gabor_2 = gabor(g, frequency = 0.084, theta = 1.27, sigma_x = 6.2)#1.77,1.37
        # sk_gabor_img_3, sk_gabor_3 = gabor(g, frequency = 0.084, theta = 1.87, sigma_x = 6.2)#1.77,1.37
        sk_frangi_img2=sk_gabor_img_1
        cv2.imshow("result1", sk_frangi_img2)
        cv2.waitKey(0)
        # cv2.imshow("result2", sk_gabor_img_2)
        # cv2.waitKey(0)
        # cv2.imshow("result3", sk_gabor_img_3)
        # cv2.waitKey(0)
        thres=30
        sk_frangi_img2[sk_frangi_img2 > thres] = 255
        # sk_gabor_img_2[sk_gabor_img_2 > 30] = 255
        # sk_gabor_img_3[sk_gabor_img_3 > 30] = 255
        sk_frangi_img2[sk_frangi_img2 <= thres] = 0
        # sk_frangi_img2[sk_gabor_img_2==255]=255
        # sk_frangi_img2[sk_gabor_img_3==255]=255

        # cv2.imshow("result", sk_frangi_img2)
        # cv2.waitKey(0)
        image1[sk_frangi_img2==255]=255
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
            if region.orientation>-np.pi/3.5 and region.orientation<np.pi/3.5:
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
        sk_frangi_img[(sk_frangi_img==255)|(sk_frangi_img2==255)]=255
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
        sk_frangi_img2[:,:]=0
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
    cv2.imshow("r1",mask_g)
    cv2.waitKey(0)
    # wrinkle_count=np.count_nonzero(mask_g)
    wrinkle_count = np.count_nonzero(sk_frangi_img)
    wrinkle_level = wrinkle_count / (image.shape[0] * image.shape[1]) / 0.11 * 100
    print(wrinkle_count/(image.shape[0]*image.shape[1]))
    cv2.imshow("r1", sk_frangi_img)
    cv2.waitKey(0)
    wrinkle_count=np.count_nonzero(sk_frangi_img)
    print(wrinkle_count / (image.shape[0] * image.shape[1]))
    if wrinkle_level > 100:
        wrinkle_level = 100
    # mask_g = cv2.resize(mask_g, (i_shape[1], i_shape[0]))
    # sk_frangi_img = cv2.resize(sk_frangi_img, (i_shape[1], i_shape[0]))
    # print(image_o.shape)
    # print(image3.shape)
    cv2.imwrite("sk_gabor_img_1.png",mask_g)
    image_o[mask_g>100]=255
    return image_o, 100 - int(wrinkle_level)


def detect_no_wrinkle(image,left = True):
    '''

    检测法令纹
    gabor滤波器对横向皱纹使用的参数
    theta与皱纹的方向有关，需要与皱纹方向一致
    @param image : 额头的图片
    @param left 是否为左脸颊图片，右脸颊图片则left=False
    @return: 皱纹列表,皱纹严重分数（0-100),越高越严重
    '''

    cv2.namedWindow("result", cv2.WINDOW_KEEPRATIO)
    # i_shape=image.shape
    # print(i_shape)
    image_o=image.copy()
    filter_w=image.shape[0]/4.0
    image1=image.copy()
    cv2.imshow("result", image)
    cv2.waitKey(0)
    g = cv2.split(image1)[0]
    cv2.imshow("result", g)
    cv2.waitKey(0)
    g = g.astype(float)
    # max_v = np.max(g)
    # min_v = np.min(g)
    # g[:, :] = (g[:, :] - min_v) * 255.0 / (max_v - min_v)
    mean_v = np.mean(g)
    factor = 0.6
    g = g.astype(float)
    g[:, :] = (g[:, :] - mean_v) * factor + g[:, :]
    g[g > 255] = 255
    g[g < 0] = 0
    g = g.astype(np.uint8)
    cv2.imshow("result", g)
    cv2.waitKey(0)
    # print("g shape: ",g.shape)
    #左脸颊的theta = 0.68，右脸颊的theta = -0.68
    the=0.68
    if left==False:
        the=-0.68
    sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.08, theta = the,bandwidth = 1.1)
    sk_gabor_img_1[sk_gabor_img_1>30]=255
    cv2.imshow("result", sk_gabor_img_1)
    cv2.waitKey(0)
    sk_gabor_img_1[sk_gabor_img_1<30]=0
    #通过偏心率（>0.98）,连通区域面积来过滤一些非皱纹
    bool_img = sk_gabor_img_1>30
    label_image = measure.label(bool_img)
    count = 0
    for region in measure.regionprops(label_image):
        if region.axis_major_length < filter_w:  # or region.area > 700
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
        if region.eccentricity > 0.98:
            count += 1
        else:
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
    cv2.imwrite("sk_gabor_img_1.png",sk_gabor_img_1)
    wrinkle_count=np.count_nonzero(sk_gabor_img_1)
    # print(wrinkle_count/400/500.0)
    wrinkle_level=wrinkle_count/(image.shape[0]*image.shape[1])/0.08*100
    # print(wrinkle_count/(image.shape[0]*image.shape[0]))
    if wrinkle_level>100:
        wrinkle_level=100
    # image3=cv2.resize(sk_gabor_img_1,(i_shape[1],i_shape[0]))
    image_o[sk_gabor_img_1==255]=255
    return image_o, 100-int(wrinkle_level)


def delete_hair(image1,image2):
    """
    效果不好
    尝试如果检测出来的皱纹的像素均值<(整张图的均值-1.5*整张图的像素方差)，则认为是毛发
    :param image1:原图
    :param image2:皱纹的mask图
    :return:去掉毛发的皱纹mask图
    """

    label_image = measure.label(image2)

    b=cv2.split(image1)[2]
    mean_v = np.mean(b)
    print(mean_v)
    std_v = np.std(b)
    print(std_v)
    for region in measure.regionprops(label_image):
        vl = 0
        coords=region.coords
        for (x,y) in coords:
            vl+=b[x,y]
        m_v=vl/len(coords)
        print(m_v)
        if m_v<mean_v-std_v*1.5:

            x = region.coords
            for i in range(len(x)):
                image2[x[i][0]][x[i][1]] = 0
        # print(np.mean(b[coords]))
    return image2


def hes(image):
    image1=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    he=sato(image1,[2,4,6])
    cv2.namedWindow("show",cv2.WINDOW_KEEPRATIO)
    cv2.imshow("show",he)
    cv2.waitKey(0)
    # image1[he>0.5]=0
    # cv2.imshow("show", image1)
    # cv2.waitKey(0)

def trans(image):
    cv.namedWindow("show",cv.WINDOW_KEEPRATIO)
    image_hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    (h,s,v)=cv.split(image_hsv)
    cv.imshow("show",h)
    cv.waitKey(0)
    cv.imshow("show", s)
    cv.waitKey(0)
    cv.imshow("show", v)
    cv.waitKey(0)

    image_lab=cv.cvtColor(image,cv.COLOR_BGR2LAB)
    (l,a,b) = cv.split(image_lab)
    cv.imshow("show", l)
    cv.waitKey(0)
    cv.imshow("show", a)
    cv.waitKey(0)
    cv.imshow("show", b)
    cv.waitKey(0)
    image_luv = cv.cvtColor(image, cv.COLOR_BGR2LUV)
    (l, u,v) = cv.split(image_luv)
    cv.imshow("show", l)
    cv.waitKey(0)
    cv.imshow("show", u)
    cv.waitKey(0)
    cv.imshow("show", v)
    cv.waitKey(0)

#使用图像增强会是个很不错的方法
if __name__ == '__main__':

    # path = r"E:\faces\face\3.jpg"
    # # path = r"E:\forehead\3.jpg"
    # # path = r"E:\faces\42.png"
    # # path = r"E:\SU\sources\Wrinkles_detection\realtest\2.jpg"
    # image = cv2.imread(path, -1)
    # # hes(image)
    # path = r"sk_gabor_img_1.png"
    # image1 = cv2.imread(path, -1)
    # im=delete_hair(image,image1)
    # cv2.namedWindow("show",cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("show",im)
    # cv2.waitKey(0)

    # image1 ,lev = detect_no_wrinkle(image,True)
    # # image1 ,_ = testGabor(image)
    # cv2.namedWindow("show", cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("show", image1)
    # cv2.waitKey(0)
    # print(lev)


    path = r"E:\faces\face\1.jpg"
    path = r"E:\faces\face\1.jpg"
    # path = r"E:\SU\sources\Wrinkles_detection\wrinkle\muouwen\8.PNG"
    # path = r"E:\faces\1227\b\2.png"
    # path = r"E:\faces\41.png"
    # path = r"E:\faces\12.jpg"
    image = cv2.imread(path, -1)
    cv2.namedWindow("show", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("show", image)
    cv2.waitKey(0)
    # trans(image)
    image1 ,lev = testGabor3(image)

    cv2.imshow("show", image1)
    cv2.waitKey(0)
    print(lev)

    # test1 face_wrinkle testGabor draw_wrinkle
    # base_path = r"E:\SU\sources\Wrinkles_detection\realtest\\"
    # base_path = r"E:\forehead\\"
    # for i in range(8):
    #     index = 1 + i
    #     index = str(index)
    #     path = base_path + index + ".jpg"
    #     # path="5.jpg"
    #     # path="2885.png"
    #     image = cv2.imread(path, -1)
    #     wrinkle_lise, level = testGabor3(image)
    #     # wrinkle_lise, level = testGabor1(image)
    #     print(level)
    #     # image1 = draw_wrinkle(image, wrinkle_lise)
    #     cv.imshow("show", wrinkle_lise)
    #     cv.waitKey(0)