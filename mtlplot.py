import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
y = np.array([65,68,73,88,97,102,130,150,160,180,183,185,181,170,145,125,110,98,87,80,76,72,69,68])
z1 = np.polyfit(x, y, 8) # 用4次多项式拟合
p1 = np.poly1d(z1)
print(p1) # 在屏幕上打印拟合多项式
yvals=p1(x) # 也可以使用yvals=np.polyval(z1,x)
plot1=plt.plot(x, y, '*',color="k")
plot2=plt.plot(x, yvals, 'r',color='k')
plt.xlabel('图像坐标')
plt.ylabel('图像像素值')
plt.legend(loc=4) # 指定legend的位置,读者可以自己help它的用法
plt.title('polyfitting')
plt.show()
