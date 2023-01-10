import random
import cv2
from PIL import Image
import numpy as np

def pad_image(im, height, width): #(height width) are the target 
    im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # 把图片从cv2格式转换成Image
    w, h = im.size  

    if w>=h*width/height:
        h1 = int(h*width/w)
        im = im.resize((width, h1),  Image.BILINEAR)
        im_new = Image.new(mode='RGB', size=(width, height), color=0)
        im_new.paste(im, ( 0, (height-h1)//2 ) )
    else:
        w1 = int(w*height/h)
        im = im.resize((w1, height),  Image.BILINEAR)
        im_new = Image.new(mode='RGB', size=(width, height), color=0)
        im_new.paste(im, ( (width-w1)//2, 0 ) )

    im_new = cv2.cvtColor(np.asarray(im_new), cv2.COLOR_RGB2BGR)  # 将Image格式的图片转成np进而转换成cv2格式    
    return im_new

# print(im)
def write1(num):
    for i in range(301,1001):
        im = cv2.imread('PaddingIMG/'+"{j}.png".format(j = num))
        rand_w=random.randint(0,150)
        rand_h=random.randint(0,130)
        im=pad_image(im,320-rand_w,320-rand_h)
        im=pad_image(im,32,32)
        cv2.imwrite('trainsetv4/'+"{k}/{j}.png".format(j = i,k = num),im)

def write2(num,num_cnt):
    for j in range(1,num_cnt[num]+1):
        for i in range(300*j+701,300*j+1001):
            im = cv2.imread('PaddingIMG/'+"{m}/{k}.jpg".format(k = j,m = num))
            rand_w=random.randint(0,150)
            rand_h=random.randint(0,130)
            im=pad_image(im,320-rand_w,320-rand_h)
            im=pad_image(im,32,32)
            cv2.imwrite('trainsetv4/'+"{m}/{k}.png".format(k = i, m = num),im)

num_cnt=[10,19,5,13,15,15,6,4,4,4]

for num in range(0,10):
    write1(num)
    print("Processing--")

for num in range(0,10):
    write2(num,num_cnt)
    print("Processing--")

print('Done!')