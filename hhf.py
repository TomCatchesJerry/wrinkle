import cv2
import numpy as np


image_path=r"E:\faces\12.jpg"
image =cv2.imread(image_path,-1)
cv2.namedWindow("show",cv2.WINDOW_KEEPRATIO)
# cv2.imshow("show",image)
# cv2.waitKey(0)
image1=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
print(image)
cv2.imshow("show",image1)
cv2.waitKey(0)
image2=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
print(image2)
cv2.imshow("show",image2)
cv2.waitKey(0)
image3=cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
print(image3.astype(float))
cv2.imshow("show",image3)
cv2.waitKey(0)
kernel=np.array([[0,0,0],
                [-0.5,0,0.5],
                [0,0,0]])
output=cv2.filter2D(image1,-1,kernel)

# kernel1=np.array([[0,0,0],
#                 [1,-2,1],
#                 [0,0,0]])
# output1=cv2.filter2D(output,-1,kernel1)
#
# kernel2=np.array([[0,1,0],
#                 [0,-2,0],
#                 [0,1,0]])
# output2=cv2.filter2D(output,-1,kernel2)
#
# kernel3=np.array([[0,0,1],
#                 [0,-2,0],
#                 [1,0,0]])
# output3=cv2.filter2D(output,-1,kernel3)

cv2.imshow("show",output)
cv2.waitKey(0)


sig=9
M_ij=np.zeros((6*sig+1,6*sig+1))
G1_ij=np.zeros((6*sig+1,6*sig+1))
N_ij=np.zeros((6*sig+1,6*sig+1))
G2_ij=np.zeros((6*sig+1,6*sig+1))

for i in range(-3*sig,3*sig+1):
    M_ij[:, i+3*sig] = -3 * sig + i - 1
    N_ij[i+3*sig,:] = -3 * sig + i - 1

G1_ij[:,:]=1/(2*np.pi*sig**4)*(np.square(M_ij[:,:])/(sig**2)-1)*np.exp((-np.square(M_ij[:,:])+np.square(N_ij[:,:]))/(2*sig**2))
G2_ij[:,:]=1/(2*np.pi*sig**6)*(M_ij[:,:]*N_ij[:,:])*np.exp((-np.square(M_ij[:,:])+np.square(N_ij[:,:]))/(2*sig**2))

outputa=cv2.filter2D(output,-1,G1_ij)
outputb=cv2.filter2D(output,-1,G2_ij)
outputc=cv2.filter2D(output,-1,G1_ij.T)

lam1=0.5*(outputa+outputc+np.sqrt(np.square(outputa-outputc)+4*outputb*outputb))
lam2=0.5*(outputa+outputc-np.sqrt(np.square(outputa-outputc)+4*outputb*outputb))

lam2[lam2==0]=0.001
r_sig=np.square(lam1/lam2)
s_sig=np.square(lam1)+np.square(lam2)
beta1=0.5
beta2=15
e_sig=np.exp(-r_sig/(2*beta1**2))*(1-np.exp(-s_sig/(2*beta2**2)))
e_sig[lam2<0]=0

d_image=image1.copy()
d_image[e_sig<=0]=0
cv2.imshow("show",d_image)
cv2.waitKey(0)