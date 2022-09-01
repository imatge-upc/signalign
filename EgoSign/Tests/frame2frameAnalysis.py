import cv2

"""
This script is for testing frame to frame if there is any problem with the alignment of the videos
"""

#import the videos
cap = cv2.VideoCapture("test3_how2/_FzvMVnR_aU/2022-05-09_14-13-12_rgb_front.mkv")
cap2 = cv2.VideoCapture("test3_how2/_FzvMVnR_aU/2022-05-09_14-13-12_rgb_head.mkv")
cap3 = cv2.VideoCapture("test3_how2/_FzvMVnR_aU/2022-05-09_14-13-05_inside.mkv")
cap4 = cv2.VideoCapture("test3_how2/_FzvMVnR_aU/2022-05-09_14-13-12_rgb_side.mkv")

#4screen video
#cap5 = cv2.VideoCapture("test1-20220510T170226Z-001/test1/sync_videos/2022-05-06_11-29-24_total.mkv")

frm_nmb = 0
while True:
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    #ret5, frame5 = cap5.read()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Image2", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Image3", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Image4", cv2.WINDOW_NORMAL)
    #cv2.namedWindow("Image5", cv2.WINDOW_NORMAL)

    cv2.imshow("Image", frame)                              
    cv2.imshow("Image2", frame2)
    cv2.imshow("Image3", frame3)
    cv2.imshow("Image4", frame4)
    #cv2.imshow("Image5", frame5)

    print(frm_nmb)
    frm_nmb+=1
    cv2.waitKey(0)  

    if ret == False:
        break
