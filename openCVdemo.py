import cv2
#引入库
cap = cv2.VideoCapture(2)
cap.set(3,1280)
cap.set(4,720)
while True:
    ret, frame = cap.read()
    cv2.imshow("Video", frame)
#读取内容
    if cv2.waitKey(10) == ord("q"):
        break
        
#随时准备按q退出
cap.release()
cv2.destroyAllWindows()
#停止调用，关闭窗口