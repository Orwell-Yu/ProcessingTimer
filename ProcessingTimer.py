from turtle import width
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imutils
from torchvision import transforms
from os import makedirs
from xlwt import Workbook
from torch.utils.data import DataLoader
from PIL import Image
from sys import stdout
from datetime import datetime
from TimerExpNetwork import *


custom_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5 ), (0.5))],
    )
classes = ('-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'empty')

torch.set_printoptions(profile="full")
def colorThres(img, gaugetype):
    if gaugetype != 'H' and gaugetype != 'M' and gaugetype != 'S':
        raise Exception('gaugetype error.')
    row, col = img.shape[:2]
    onechannel=np.zeros((row,col))
    for i in range(row):    
        for j in range(col):
            if img[i,j,1]>250 and img[i,j,0]<150 and img[i,j,2]<200:
                onechannel[i,j] = 1
    # kernel = np.ones((1,1), np.uint8)
    # onechannel = cv2.morphologyEx(onechannel, cv2.MORPH_OPEN, kernel)
    onechannel = cv2.dilate(onechannel, np.ones((2,2), np.uint8))
    return onechannel

def handleLR(mat,imgL):
    height, width = mat.shape
    while imgL<width and sum(mat[:,imgL])<=2:
        imgL+=1
    imgR=imgL
    while imgR<width and sum(mat[:,imgR])>2:
        imgR+=1
    # if imgR-imgL<10:
    #     # 还要判断是不是1
    #     sample = mat[:, imgL:imgR]
    #     sampleT, sampleB = handleTB(sample, 0, strictmode=False)
    #     if sampleB - sampleT < 3: #说明是一个杂点
    #         imgL=imgR
    #     else:
    #         break
    # else:
    #     break
    return imgL, imgR

def handleTB(mat, imgT, strictmode = True):
    height, width = mat.shape
    while True:
        while imgT<height and sum(mat[imgT,:])<=2:
            imgT+=1
        imgB=imgT
        while imgB<height and sum(mat[imgB,:])>2:
            imgB+=1    # im.show()
    # cv2.waitKey(10000)
        if strictmode and imgB-imgT<10:
            imgT=imgB
        else:
            break
    return imgT, imgB

def Split(mat):
    imgT, imgB = handleTB(mat, 0)
    mat = mat[imgT:imgB, :]
    imgL = 0
    postuplelist = []
    for i in range(5):
        imgL, imgR = handleLR(mat, imgL)
        if imgL == imgR:
            break
        postuplelist.append((imgL, imgR))
        imgL = imgR+1
    # test_mat = mat
    # for i, (l, r) in enumerate(postuplelist):
    #     cv2.imshow(str(i), test_mat[imgT:imgB, l:r])
        # test_mat = cv2.rectangle(test_mat, (l, b), (r, t), (255, 255, 255), 2)
    # cv2.imshow('test_mat', test_mat)
    return imgT, imgB, postuplelist

def pad_image(im, height, width): #(height width) are the target height and width.
    im = Image.fromarray(im)  # convert "im" from "array" to "PIL Image". Now "im" is 2D. 
    w, h = im.size  
    #print(h,w)
    # im.show()
    # cv2.waitKey(100)
    if w>=h*width/height:
        h1 = int(h*width/w)
        im = im.resize((width, h1),  Image.BILINEAR)
        im_new = Image.new(mode='L', size=(width, height), color=0)
        im_new.paste(im, ( 0, (height-h1)//2 ) )
    else:
        w1 = int(w*height/h)
        # print(w1)
        im = im.resize((w1, height),  Image.BILINEAR)
        im_new = Image.new(mode='L', size=(width, height), color=0)
        im_new.paste(im, ( (width-w1)//2, 0 ) )
    # im_new.show()
    im_new = np.asarray(im_new)  # convert "im_new" from "PIL Image" to "array"
    return im_new

def findMin(arr):
    min = 99999
    id = -1
    for i, num in enumerate(arr):
        if num < min:
            min = num
            id = i
    return id, min

def computeImageSize():
    return

def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    # im=Image.fromarray(edged)
    # im.show()
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)
 
    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)

def processImage(img, ned, gaugetype,rate,cnt):
    result = cv2.matchTemplate(img, ned, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    #print(max_loc)
    # print(img.shape)
    # print(gaugetype)
    #Note: img and ned are both in BGR, not in RGB!
    if gaugetype == "H":

        # original size of sample1, sample2, sample3: 26*26
        # for example, if img.shape= (360, 426, 3), ned.shape= (26, 21, 3), 
        # then result.shape= (360-(26-1)=335, 426-(21-1)=406)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, test_img = cv2.threshold(test_img, 150, 255, cv2.THRESH_BINARY)
        # 21*21
        # sample1 = test_img[max_loc[1]-int(258*rate) : max_loc[1]-int(113*rate), max_loc[0]-int(236*rate) : max_loc[0]-int(91*rate)] 
        # sample2 = test_img[max_loc[1]-int(258*rate) : max_loc[1]-int(113*rate), max_loc[0]-int(122*rate) : max_loc[0]-int(33*rate)]
        sample1 = test_img[max_loc[1]-int(258*rate) : max_loc[1]-int(113*rate), max_loc[0]-int(236*rate) : max_loc[0]-int(91*rate)] 
        sample2 = test_img[max_loc[1]-int(258*rate) : max_loc[1]-int(113*rate), max_loc[0]-int(122*rate) : max_loc[0]-int(33*rate)]
    # elif gaugetype == "Tm":
    #     # 多目标匹配取最左边那个0
    #     index = np.where(result > 0.8)
    #     if len(index[1]) > 0:
    #         yPos, _ = findMin(index[1])
    #         max_loc = (index[1][yPos], index[0][yPos])
    #     else:
    #         _, _, _, max_loc = cv2.minMaxLoc(result)
    #     test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     _, test_img = cv2.threshold(test_img, 100, 255, cv2.THRESH_BINARY)
    #     # 识别0标志的参数, 32*32
    #     sample1 = test_img[max_loc[1]+int(4*rate) : max_loc[1]+int(36*rate), max_loc[0]+int(27*rate) : max_loc[0]+int(59*rate)] 
    #     sample2 = test_img[max_loc[1]+int(4*rate) : max_loc[1]+int(36*rate), max_loc[0]+int(53*rate) : max_loc[0]+int(85*rate)]
    #     negsample = test_img[0:32, 0:32] #useless, since temperature cannot be negative
    #     sample3 = test_img[max_loc[1]+int(4*rate) : max_loc[1]+int(36*rate), max_loc[0]+int(81*rate) : max_loc[0]+int(113*rate)] 
        
    #     # 识别oC标志的参数，作为备用：32*23
    #     # sample1 = test_img[max_loc[1]+8 : max_loc[1]+40, max_loc[0]-87 : max_loc[0]-64] #之前是+9，+42，-84，-62
    #     # sample2 = test_img[max_loc[1]+8 : max_loc[1]+40, max_loc[0]-60 : max_loc[0]-37] # 之前是+7，+40，-57，-34
    #     # negsample = test_img[0:32, 0:32] #useless, since temperature cannot be negative
    #     # sample3 = test_img[max_loc[1]+8 : max_loc[1]+40, max_loc[0]-30 : max_loc[0]-7] #之前是+8，+41，-28，-5
    elif gaugetype == "M":
        _, _, _, max_loc = cv2.minMaxLoc(result)
        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, test_img = cv2.threshold(test_img, 150, 255, cv2.THRESH_BINARY)
        # 27*27
        sample1 = test_img[max_loc[1]-int(258*rate) : max_loc[1]-int(113*rate), max_loc[0]+int(42*rate) : max_loc[0]+int(187*rate)] 
        sample2 = test_img[max_loc[1]-int(258*rate) : max_loc[1]-int(113*rate), max_loc[0]+int(160*rate) : max_loc[0]+int(250*rate)]
    elif gaugetype == "S":
        _, _, _, max_loc = cv2.minMaxLoc(result)
        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, test_img = cv2.threshold(test_img, 150, 255, cv2.THRESH_BINARY)
        # 30*30
        sample1 = test_img[max_loc[1]-int(170*rate) : max_loc[1]-int(90*rate), max_loc[0]+int(300*rate) : max_loc[0]+int(370*rate)] 
        sample2 = test_img[max_loc[1]-int(170*rate) : max_loc[1]-int(90*rate), max_loc[0]+int(370*rate) : max_loc[0]+int(435*rate)] 
    
    sample1 = pad_image(sample1, 32, 32)
    sample2 = pad_image(sample2, 32, 32)
    # if gaugetype != "Tm":
    #     negsample = pad_image(negsample, 32, 32)
    if gaugetype == "Tm":
        kernel = np.ones((3,3), np.uint8)
    else:
        kernel = np.ones((1,1), np.uint8)
    sample1 = cv2.dilate(sample1, kernel)
    sample2 = cv2.dilate(sample2, kernel)
    # if(gaugetype == "S" and cnt>37):
    #     cv2.imwrite('PaddingIMG/3/'+"{j}.jpg".format(j = int(cnt-32)),sample1)
    #     sample1 = Image.fromarray(sample1)
    #     sample1.show()
    # if(gaugetype == "S" and cnt%10==8):
    #     cv2.imwrite('PaddingIMG/9/'+"{j}.jpg".format(j = int(cnt/10)+1),sample2)
        # sample2 = Image.fromarray(sample2)
        # sample2.show()
    # sample1 = Image.fromarray(sample1)
    # sample1.show()
    # sample2 = Image.fromarray(sample2)
    # sample2.show()
    return sample1, sample2

def drawImages_Gray(imgarr, prompt):
    print(prompt)
    for img in imgarr:
        plt.imshow(img, cmap="gray")
        plt.show()

def netRead(batch, net):
    images = batch[0]
    outputs = net(images)
    sm = nn.Softmax(dim=1)      
    sm_outputs = sm(outputs)
    probs, index = torch.max(sm_outputs, dim=1)
    first = classes[index[0]]
    second = classes[index[1]]
    # print(first,second)
    # neg = classes[index[2]]
    # third = classes[index[3]]
    return first, second

def isValidNum(first, second, caresign = True):
    if first == 'empty' or first == '-':
        return False
    if second == 'empty' or second == '-':
        return False
    return True

def loadNet(netpath):
    net = MyNetwork()
    net.load_state_dict(torch.load(netpath),False)
    return net

def readTempate():
    tlist = []
    ned = cv2.imread("./template images/E1.png")
    tlist.append(ned)
    ned = cv2.imread("./template images/0.png")
    tlist.append(ned)
    ned = cv2.imread("./template images/E2.png")
    tlist.append(ned)
    ned = cv2.imread("./template images/E3.png")
    tlist.append(ned)
    return tlist

def openVideo(videoNumber):
    # video = cv2.VideoCapture("./videos/" + videoNumber)
    video = cv2.VideoCapture(videoNumber)
    video.set(3,1280)
    video.set(4,720)
    #check if the video exists
    if not video.isOpened():
        raise Exception("Video cannot be opened.")
    return video

def showVideoInfo( video, sampling_rate):
    #show some info of video
    fps = video.get(5)
    totalframes = video.get(7)
    #print("Video: " + videoname)
    #print("Total frames: " + str(totalframes))
    print("FPS: " + str(int(fps)))
    #if sampling_rate = 10, it means every 0.1s we capture a photo
    print("Sampling_rate: " + str(sampling_rate) + "Hz") 
    # timeF: we pick one frame every 'timeF' frames.
    # Here we pick one frame every 5 frames. 
    timeF = fps / sampling_rate 
    #print("Frames needed to be extracted:" + str(int(totalframes/timeF)))
    return int(timeF), totalframes

def initExcelTable():
    #create and initialize an Excel table
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    # sheet1.write(0, 0, "Time/s")
    # sheet1.write(0, 1, "Resistance vacuum gauge/Pa")
    # sheet1.write(0, 2, "Temperature/C")
    # sheet1.write(0, 3, "Thermocouple vacuum gauge/Pa")
    # sheet1.write(0, 4, "Ionization vacuum gauge/Pa")
    sheet1.write(0, 0, "Hour/h")
    sheet1.write(0, 2, "Minute/min")
    sheet1.write(0, 4, "Second/s")
    return wb, sheet1

def splitSave(frame, splitarray, templateimglist, debugmode = [],cnt=0):
    width, length, _ = np.shape(frame)
    # Note：Resistance vacuum gauge--R， Thermocouple vacuum gauge--T， Ionization vacuum gauge--I
    #       temperature--Tm
    # Resistance vacuum gauge
    # marker = find_marker(frame)
    # width_outline = marker[1][0]
    marker = find_marker(frame)
    width_outline = marker[1][0]
    rate=width_outline/489
    #rate=1
    widthlower, widthupper, lengthlower, lengthupper = splitarray[0]
    subframe = frame[int(widthlower*width):int(widthupper*width), int(lengthlower*length):int(lengthupper*length)]
    first_H, second_H = processImage(subframe,templateimglist[0],"H",rate,cnt)
    # temperature
    # widthlower, widthupper, lengthlower, lengthupper = splitarray[1]
    # subframe = frame[int(widthlower*width):int(widthupper*width), int(lengthlower*length):int(lengthupper*length)]
    # first_Tm, second_Tm, neg_Tm, third_Tm = processImage(subframe,templateimglist[1],"Tm",rate)
    # # Thermocouple vacuum gauge
    widthlower, widthupper, lengthlower, lengthupper = splitarray[0]
    subframe = frame[int(widthlower*width):int(widthupper*width), int(lengthlower*length):int(lengthupper*length)]
    first_M, second_M = processImage(subframe,templateimglist[0],"M",rate,cnt)
    # # Ionization vacuum gauge
    widthlower, widthupper, lengthlower, lengthupper = splitarray[0]
    subframe = frame[int(widthlower*width):int(widthupper*width), int(lengthlower*length):int(lengthupper*length)]
    first_S, second_S = processImage(subframe,templateimglist[0],"S",rate,cnt)
    
    if 'H' in debugmode:
        drawImages_Gray([first_H, second_H], "H_Split:")
    # if 'Tm' in debugmode:
    #     drawImages_Gray([first_Tm, second_Tm, third_Tm], "Tm_Split:")
    if 'M' in debugmode:
        drawImages_Gray([first_M, second_M], "M_Split:")
    if 'S' in debugmode:
        drawImages_Gray([first_S, second_S], "S_Split:")
    return (first_H, second_H,
            first_M, second_M,
            first_S, second_S)

# def showProgress(cur, tot):
#     print("\r", end="")
#     print("progress: {:.1f}%".format(cur/tot*100), end="")
#     stdout.flush()

def readNumber(net, sheet, debugmode = [], show_result = False,  imageslist_H = None, 
                 imageslist_M = None, imageslist_S = None):#test three images
    if imageslist_H is None or imageslist_M is None or imageslist_S is None:
        raise Exception('At least one imageslist not found.')
    # imageslist_R[0].show()
    # H
    real_test_H = MyDataset_notfromdisk(imglist=imageslist_H, transform=custom_transform, mode="test")
    real_testloader_H = DataLoader(real_test_H, batch_size = 2, shuffle = False)
    for i, batch in enumerate(real_testloader_H, 1):
        first, second = netRead(batch, net)
        if 'H' in debugmode:
            drawImages_Gray(    [batch[0][0].permute(1, 2, 0), batch[0][1].permute(1, 2, 0), 
                                batch[0][2].permute(1, 2, 0), batch[0][3].permute(1, 2, 0)],
                                "H: first={}, second={}".format(first, second)
                            )
        if isValidNum(first, second):
            result = eval(first)*10 + eval(second)
        else:
            result = "NaN"
        # print(result)
        sheet.write(i, 0, result)
        # sheet.write(i, 0, i) #'result_cnt/10' means time(units of sec)
        if show_result:
            print(result, i/10) #for debug
    
    # Tm
    # real_test_Tm = MyDataset_notfromdisk(imglist=imageslist_Tm, transform=custom_transform, mode="test")
    # real_testloader_Tm = DataLoader(real_test_Tm, batch_size = 4, shuffle = False)
    # for i, batch in enumerate(real_testloader_Tm, 1):
    #     first, second, neg, third = netRead(batch, net)
    #     if 'Tm' in debugmode:
    #         drawImages_Gray(    [batch[0][0].permute(1, 2, 0), batch[0][1].permute(1, 2, 0), 
    #                              batch[0][3].permute(1, 2, 0)],
    #                             "Tm: first={}, second={}, third={}".format(first, second, third)
    #                         )
    #     if isValidNum(first, second, neg, third, False):
    #         result = eval(first)*100 + eval(second)*10 + eval(third)
    #     else:
    #         result = "NaN"
    #     sheet.write(i, 2, result)

    # M
    real_test_M = MyDataset_notfromdisk(imglist=imageslist_M, transform=custom_transform, mode="test")
    real_testloader_M = DataLoader(real_test_M, batch_size = 2, shuffle = False)
    for i, batch in enumerate(real_testloader_M, 1):
        first, second = netRead(batch, net)
        if 'M' in debugmode:
            drawImages_Gray(    [batch[0][0].permute(1, 2, 0), batch[0][1].permute(1, 2, 0), 
                                batch[0][2].permute(1, 2, 0), batch[0][3].permute(1, 2, 0)],
                                "M: first={}, second={}".format(first, second)
                            )
        if isValidNum(first, second):
            result = eval(first)*10 + eval(second)
        else:
            result = "NaN"
        # print(result)
        sheet.write(i, 2, result)

    # S
    real_test_S = MyDataset_notfromdisk(imglist=imageslist_S, transform=custom_transform, mode="test")
    real_testloader_S = DataLoader(real_test_S, batch_size = 2, shuffle = False)
    for i, batch in enumerate(real_testloader_S, 1):
        first, second = netRead(batch, net)
        if 'I' in debugmode: 
            drawImages_Gray(    [batch[0][0].permute(1, 2, 0), batch[0][1].permute(1, 2, 0), 
                                batch[0][2].permute(1, 2, 0), batch[0][3].permute(1, 2, 0)],
                                "S: first={}, second={}".format(first, second)
                            )
        if isValidNum(first, second):
            result = eval(first)*10 + eval(second)
        else:
            result = "NaN"
        sheet.write(i, 4, result)
        #print('Prossing')

def saveResult(wb):  
    makedirs("./resultData/", exist_ok=True)
    filename = str(datetime.now())[0:19].replace(":","-")+'.xls'
    wb.save("./resultData/"+filename)

from matplotlib.pyplot import imshow


def readExpData(netpath, splitarray,videoNumber, sampling_rate = 1, savetodisk = False, 
                fnReadNumDebug = [], fnSplitDebug = []):
    # load the trained convolution network
    net = loadNet(netpath)
    templateimglist = readTempate()
    video = openVideo(videoNumber)
    # timeF: we pick one frame every 'timeF' frames.
    # Here we pick one frame every 5 frames. 
    timeF, totalframes = showVideoInfo(video, sampling_rate)
    # print(timeF,totalframes)
    wb, sheet1 = initExcelTable()
    rval = True
    frame_cnt = 1
    imageslist_H = []
    # imageslist_Tm = []
    imageslist_M = []
    imageslist_S = []
    # Tempimageslist_Tm = []
    # cnt=0
    print("Splitting the image...")
    while rval: 
        frame_cnt+=1
        if cv2.waitKey(10) == ord("q"):
            break
        # Keep reading frames until rval=False(that is, end of file)
        rval, frame = video.read() # Note: frame is in BGR colorspace, not RGB!
        if(rval==0):
            break
        cv2.imshow("video", frame)
        marker = find_marker(frame)
        width_outline = marker[1][0]
        #print(width_outline)
        rate=width_outline/489
        width,length,_ = np.shape(frame)
        widthlower, widthupper, lengthlower, lengthupper = splitarray[0]
        subframe = frame[int(widthlower*width):int(widthupper*width), int(lengthlower*length):int(lengthupper*length)]
        width1, length1, _ = np.shape(subframe) 
        result = cv2.matchTemplate(subframe, templateimglist[0], cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        if(frame_cnt % timeF == 0 and (max_loc[1]-int(258*rate)<0 or max_loc[0]-int(236*rate)<0 or max_loc[0]+int(435*rate)>length1)):
            print("Can't Find")
            continue
        # widthlower, widthupper, lengthlower, lengthupper = splitarray[2]
        # subframe = frame[int(widthlower*width):int(widthupper*width), int(lengthlower*length):int(lengthupper*length)]
        # width1, length1, _ = np.shape(subframe) 
        # result = cv2.matchTemplate(subframe, templateimglist[0], cv2.TM_CCOEFF_NORMED)
        # _, _, _, max_loc = cv2.minMaxLoc(result)
        # if(frame_cnt % timeF == 0 and (max_loc[0]-int(74*rate)<0 or max_loc[0]+int(99*rate)>length1 or max_loc[1]+int(32*rate)>width1)):
        #     print("Can't Find")
        #     continue
        # widthlower, widthupper, lengthlower, lengthupper = splitarray[3]
        # subframe = frame[int(widthlower*width):int(widthupper*width), int(lengthlower*length):int(lengthupper*length)]
        # width1, length1, _ = np.shape(subframe) 
        # result = cv2.matchTemplate(subframe, templateimglist[0], cv2.TM_CCOEFF_NORMED)
        # _, _, _, max_loc = cv2.minMaxLoc(result)
        # if(frame_cnt % timeF == 0 and (max_loc[0]-int(74*rate)<0 or max_loc[0]+int(103*rate)>length1 or max_loc[1]+int(34*rate)>width1)):
        #     print("Can't Find")
        #     continue
        if (frame_cnt % timeF == 0 and rval): 
            # take down the data
            # print(frame_cnt)
            print('--Processing!--')
            (first_H, second_H, 
                first_M, second_M,
                first_S, second_S
            ) = splitSave(frame, splitarray, templateimglist, fnSplitDebug, frame_cnt/timeF)
            imageslist_H.append((first_H, second_H))
            # imageslist_Tm.append((first_Tm, second_Tm, neg_Tm, third_Tm))
            imageslist_M.append((first_M, second_M))
            imageslist_S.append((first_S, second_S))
            # Tempimageslist_Tm.append((first_Tm, second_Tm, neg_Tm, third_Tm))
            # real_test_Tm = MyDataset_notfromdisk(imglist=Tempimageslist_Tm, transform=custom_transform, mode="test")
            # real_testloader_Tm = DataLoader(real_test_Tm, batch_size = 4, shuffle = False)
            # for i, batch in enumerate(real_testloader_Tm, 1):
            #     first, second, neg, third = netRead(batch, net)
            # if isValidNum(first, second, neg, third, False):
            #     Print_Tm_result = eval(first)*100 + eval(second)*10 + eval(third)
            #     print(Print_Tm_result)
            print('------------')
            # Tempimageslist_Tm.clear()
        # if frame_cnt % 50 == 0:
        #     showProgress(frame_cnt, totalframes)
    #save the excel table
    print("\nReading the number...    ", end="")
    readNumber(net=net, sheet=sheet1, imageslist_H=imageslist_H, 
                 imageslist_M=imageslist_M, imageslist_S=imageslist_S,
                debugmode=fnReadNumDebug)
    saveResult(wb)
    print("Done!\n")

readExpData(    sampling_rate = 1,
                netpath="pth/attempt18.pth",
                videoNumber=2,
                fnReadNumDebug=[], 
                fnSplitDebug=[], 
                splitarray=[(1/5, 4/5, 1/5, 4/5), (0.4, 0.99999, 0, 0.5), (0, 0.5, 0.3, 0.67), (0, 0.5, 0.7, 0.99999)]
                )