# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import animation
# import cv2
# import cv2
# import numpy as np
# import math
# from typing import List, Tuple
# import tensorflow as tf
# import glob
# import os
# import json
# import pandas as pd
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import genfromtxt
# import csv,cv2,os
# from sklearn.preprocessing import MinMaxScaler
# import PIL.Image
# import numpy as np 
# from celluloid import Camera
# import matplotlib.pyplot as plt

# global cnt
# datas = genfromtxt("C:\\Users\\won\\Desktop\\a.csv", delimiter=',')
# scaler = MinMaxScaler()
# x_data= scaler.fit_transform(np.reshape(datas[1:,1],(-1,1)))
# x_data2= scaler.fit_transform(np.reshape(datas[1:,2],(-1,1)))
# cnt = 0
# x = []
# y = []
# fig = plt.figure()
# ax1 = plt.subplot(2, 1, 1)
# ax2 = plt.subplot(2, 1, 2, sharex=ax1)
# line, = ax1.plot([], [], lw=3)
# ax1.tick_params(bottom= False,labelbottom = False)
# camera = Camera(fig)

# for cnt in range(0,len(x_data)-120):
#     x = np.linspace(cnt/30,(cnt+120)/30,120)
#     y = x_data[cnt:cnt+120,0]
#     plt.plot(x,y,linewidth=0.3)
#     plt.xticks([(cnt-120)//30,(cnt-90)//30,(cnt-60)//30,(cnt-30)//30,cnt//30])
#     camera.snap()
# anim = camera.animate()

# anim.save('filename.gif', writer='imagemagick')



from urllib import response
import cv2
import numpy as np
import math
from typing import List, Tuple
import tensorflow as tf
import glob
import os
import json
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import csv,cv2,os
from sklearn.preprocessing import MinMaxScaler
import PIL.Image
import numpy as np 
import matplotlib.pyplot as plt
import requests
import json
from skimage.transform import resize

def gazeTrackerV2(video_path=None,result_path=None):
    # load a video to capture
    video_cap = cv2.VideoCapture(video_path)
    print(video_path)
    # windows video codec
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # video frame per second
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    # set up video form and path to save a video file
    out = cv2.VideoWriter(result_path,fourcc,fps,(320,120))
    # read a first frame of a video
    success,image = video_cap.read()
    modified_video_list = []
    
    # extract all the frames of video
    while success:
       modified_video_list.append(image)
       success,image = video_cap.read()
    
    # resize and change the images color into GrayScale
    preds = []
    tmp = []
    cnt = 1
    
    video_np = np.asarray(modified_video_list)
    # detect nystagmus
    print(video_np.shape)
    
    
    
    contours = []
    srcs = []
    # extract pupils only from nystagmus
    for frame in video_np:
       #pred = np.reshape(pred,(160,320))
       dst = np.ones((120,320),dtype="uint8")
       dst.fill(255)
       src = frame.astype("uint8")
       #msk = (pred*255).astype("uint8")
       ret,bimage = cv2.threshold(frame,4,255,cv2.THRESH_BINARY_INV)
       #img = cv2.copyTo(src,bimage,dst)
       bimage = cv2.normalize(bimage, None, 3, 255, cv2.NORM_MINMAX)
      
       # make blurred images clear
       bimage = cv2.GaussianBlur(bimage, (15,15),0)
    
       # change all color into white except of black(0)
       mask = cv2.inRange(bimage,5,255)
       bimage[mask>0] = 255
    
    
       # resize image to 120 x 320 from 160 x 320
       bimage = resize(bimage,output_shape= (120, 320,1),preserve_range=True).astype("uint8")
       src =  resize(src,output_shape= (120, 320,1),preserve_range=True).astype("uint8")
       src = cv2.cvtColor(src,cv2.COLOR_GRAY2RGB)
    
       # find contour
       mode = cv2.RETR_EXTERNAL
       method =  cv2.CHAIN_APPROX_NONE   
       contour, hierarchy = cv2.findContours(bimage,mode,method)
    
    
       contours.append(contour)
       srcs.append(src)
    
    
    
    
    contours = np.asarray(contours)
    srcs = np.asarray(srcs)
    
    print("result I expect : ",end=' ')
    print(contours.shape,srcs.shape)
    
    max_radius_left = 0
    max_center_left = (0,0)
    
    max_radius_right = 0
    max_center_right = (0,0)
    
    max_area_left = 0
    max_area_right = 0
    
    
    
    
    for contour in contours:
    	for c in contour:
            (_,_),radius =  cv2.minEnclosingCircle(c)
            x,y,w,h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
    
            if radius < 8:
                continue
            
            
            if max_area_left < area and x < 160:
                max_radius_left = w//2 - 5
                max_center_left = (int(x+(w//2)),int(y+(h//2)))
                max_area_left = area
    
    
            elif max_area_right < area and x >= 160:
                max_radius_right = w//2 - 5
                max_center_right = (int(x+(w//2)),int(y+(h//2)))
                max_area_right = area
    
    
    
    print("max_area_left : ",max_area_left,"max_area_right : ",max_area_right)
    
    

    point_x_left = []
    point_y_left = []

    point_x_right = []
    point_y_right = []
    
    for contour,src in zip(contours,srcs):
        for c in contour:
            x,y,w,h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
    
    
            if x < 160:
                if max_area_left * 0.3 >= area or max_area_left < area:
                    continue
                
                elif max_area_left * 0.85 <= area:
                    cx = int(x + (w * 0.5))
                    cy = int(y + (h * 0.5))
                    rx = int(cx + (cx - max_center_left[0]))
                    ry = int(cy + (cy - max_center_left[1]))
                    src = cv2.circle(src,(cx,cy),int(w//2)-5,(0,0,255),1,lineType=8)
                    src = cv2.circle(src,(cx,cy),(int(w//2)-5)//3,(0,0,255),1,lineType=8)
                    src = cv2.arrowedLine(src,(cx,cy),(rx,ry),(153,255,0),1)
                    src = cv2.line(src,(max_center_left[0],max_center_left[1]),(cx,cy),(255,0,0),1,lineType=8)
                    point_x_left.append(cx)
                    point_y_left.append(cy)
    
                elif max_area_left * 0.5 <= area and max_area_left * 0.85 > area:
                    (mx,my),radius =  cv2.minEnclosingCircle(c)
                    rx = int(mx + (mx - max_center_left[0]))
                    ry = int(my + (my - max_center_left[1]))
                    src = cv2.circle(src,(int(mx),int(my)),int(w//2)-5,(0,0,255),1,lineType=8)
                    src = cv2.circle(src,(int(mx),int(my)),(int(w//2)-5)//3,(0,0,255),1,lineType=8)
                    src = cv2.arrowedLine(src,(int(mx),int(my)),(rx,ry),(153,255,0),1)
                    src = cv2.line(src,(max_center_left[0],max_center_left[1]),(int(mx),int(my)),(255,0,0),1,lineType=8)
                    point_x_left.append(int(mx))
                    point_y_left.append(int(mx))   
    
                elif max_area_left * 0.5 > area:
                    (mx,my),radius =  cv2.minEnclosingCircle(c)
                    rx = int(mx + (mx - max_center_left[0]))
                    ry = int(my + (my - max_center_left[1]))
                    src = cv2.circle(src,(int(mx),int(my)),int(radius),(0,0,255),1,lineType=8)
                    src = cv2.circle(src,(int(mx),int(my)),(int(w//2)-5)//3,(0,0,255),1,lineType=8)
                    src = cv2.arrowedLine(src,(int(mx),int(my)),(rx,ry),(153,255,0),1)
                    src = cv2.line(src,(max_center_left[0],max_center_left[1]),(int(mx),int(my)),(255,0,0),1,lineType=8)
                    point_x_left.append(int(mx))
                    point_y_left.append(int(mx))    
    
    
            elif x >= 160:
                if max_area_right * 0.3 >= area or max_area_right < area:
                    continue
                
                elif max_area_right * 0.85 <= area:
                    cx = int(x + (w * 0.5))
                    cy = int(y + (h * 0.5))
                    rx = int(cx + (cx - max_center_right[0]))
                    ry = int(cy + (cy - max_center_right[1]))
                    src = cv2.circle(src,(cx,cy),int(w//2)-5,(0,0,255),1,lineType=8)
                    src = cv2.circle(src,(cx,cy),(int(w//2)-5)//3,(0,0,255),1,lineType=8)
                    src = cv2.arrowedLine(src,(cx,cy),(rx,ry),(153,255,0),1)
                    src = cv2.line(src,(max_center_right[0],max_center_right[1]),(cx,cy),(255,0,0),1,lineType=8)
                    point_x_right.append(cx)
                    point_y_right.append(cy)
    
                elif max_area_right * 0.5 <= area and max_area_right * 0.85 > area:
                    (mx,my),radius =  cv2.minEnclosingCircle(c)
                    rx = int(mx + (mx - max_center_right[0]))
                    ry = int(my + (my - max_center_right[1]))
                    src = cv2.circle(src,(int(mx),int(my)),int(w//2)-5,(0,0,255),1,lineType=8)
                    src = cv2.circle(src,(int(mx),int(my)),(int(w//2)-5)//3,(0,0,255),1,lineType=8)
                    src = cv2.arrowedLine(src,(int(mx),int(my)),(rx,ry),(153,255,0),1)
                    src = cv2.line(src,(max_center_right[0],max_center_right[1]),(int(mx),int(my)),(255,0,0),1,lineType =8)
                    point_x_right.append(int(mx))
                    point_y_right.append(int(mx))  


                elif max_area_right * 0.5 > area:
                    (mx,my),radius =  cv2.minEnclosingCircle(c)
                    rx = int(mx + (mx - max_center_right[0]))
                    ry = int(my + (my - max_center_right[1]))
                    src = cv2.circle(src,(int(mx),int(my)),int(radius),(0,0,255),1,lineType=8)
                    src = cv2.circle(src,(int(mx),int(my)),(int(w//2)-5)//3,(0,0,255),1,lineType=8)
                    src = cv2.arrowedLine(src,(int(mx),int(my)),(rx,ry),(153,255,0),1)
                    src = cv2.line(src,(max_center_right[0],max_center_right[1]),(int(mx),int(my)),(255,0,0),1,lineType=8)
                    point_x_right.append(int(mx))
                    point_y_right.append(int(mx))  

    
    for src in srcs:
       out.write(src)
    
    print("max_radius_left : ",max_radius_left,"max_center_left : ",max_center_left,"max_radius_right : ",max_radius_right,"max_center_right : ",max_center_right)
    # save video
    out.release()

    return point_x_left,point_y_left,point_x_right,point_y_right

def gazeTracker(video_path=None,video_path_result=None,w=320,h=120,c=3,drawable = True,savable = True) -> Tuple[List, List]:
    #windows support codec
    cap = cv2.VideoCapture(video_path)
    fps = 15
    size_x=int(w/2)
    size_y=120
    idx = 1
    frame_idx = 1
    point_x = []
    point_y = []
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(video_path_result, fourcc, fps, (w, h))
    ll =[]
    ret, frame = cap.read()
    while ret:
        frames = [frame[:size_y,:size_x],frame[:size_y,size_x:]]
        if frame_idx % 2 == 1:
            for frame in frames:
                # frame = cv2.resize(frame, (size_x,size_y))
                frame_pupil = cv2.medianBlur(frame, 25)
                gray = cv2.cvtColor(frame_pupil,cv2.COLOR_BGR2GRAY)
                # make pupils's boundary clear(a little bit white -> pure white, a little bit black -> dark black)
                ret,bimage = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV)
                mode = cv2.RETR_TREE
                method =  cv2.CHAIN_APPROX_SIMPLE
                contours,_ = cv2.findContours(bimage,mode,method)
                flag = False

                if len(contours) == 0:
                    point_x.append(-355)
                    point_y.append(-355)
                    flag = True

                for c in contours:
                    (_,_),radius =  cv2.minEnclosingCircle(c)
                    x,y,w,h = cv2.boundingRect(c)
                    area = cv2.contourArea(c)
            
                    if radius < 8:
                        continue
                    max_center_left = (int(x+(w//2)),int(y+(h//2)))
                    max_area_left = cv2.contourArea(c)
            
            
            
                print("max_area_left : ",max_area_left)

                
                for c in contours:
                    (_,_),radius =  cv2.minEnclosingCircle(c)
                    x,y,w,h = cv2.boundingRect(c)
                    M = cv2.moments(c)
                    try:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    except:
                        continue
                    if drawable == True:
                        radius = math.sqrt(((int(size_x/2)-cX)**2)+((int(size_y/2)-cY)**2))
                        cx = int(x + (w * 0.5))
                        cy = int(y + (h * 0.5))
                        rx = int(cx + (cx - 30))
                        ry = int(cy + (cy - 30))
                        frame = cv2.circle(frame,(cx,cy),int(w//2)-5,(0,0,255),1,lineType=8)
                        frame = cv2.circle(frame,(cx,cy),(int(w//2)-5)//3,(0,0,255),1,lineType=8,thickness=)
                        frame = cv2.arrowedLine(frame,(cx,cy),(rx,ry),(153,255,0),1)
                        #src = cv2.line(src,(max_center_right[0],max_center_right[1]),(cx,cy),(255,0,0),1,lineType=8)
                        point_x.append(cx)
                        point_y.append(cy)
                        # ellipse = cv2.fitEllipse(contours[0])
                        # if cv2.contourArea(c) >= (math.pi*abs(int(size_x/2)-cX)*abs(int(size_y/2)-cY))*0.1:
                        #     cv2.line(frame,(int(size_x/2),int(size_y/2)),(cX,cY),(255,0,0),1)
                        #     r = (math.atan2(int(size_y/2)-cY,int(size_x/2)-cX)*180/math.pi)
                        #     x = int(cX-radius*math.cos(math.radians(r)))
                        #     y = int(cY-radius*math.sin(math.radians(r)))
                        #     cv2.circle(frame,(int(size_x/2),int(size_y/2)),3,(255,0,255),1)
                        #     cv2.circle(frame,(cX,cY),15,(0,0,255),1)
                        #     cv2.arrowedLine(frame,(cX,cY),(x,y),(153,255,0),1)
                    if idx%2 == 0:
                        if flag:
                            flag = False
                        else:
                            point_x.append(cX)
                            point_y.append(cY)

                if savable == True:
                    if idx %2 == 0:
                        out.write(cv2.hconcat([ll[len(ll)-1],frame]))
                    else:
                        ll.append(frame)
                idx+=1
        frame_idx += 1
        ret, frame = cap.read()

    cap.release()
    if savable == True:
        out.release()
    cv2.destroyAllWindows()

    return point_x, point_y

def point_to_csv(point_x, point_y,save_path):
    temp_x = point_x[0]
    temp_y = point_y[0]
    tp_x = 0
    tp_y = 0
    #point_x[0] = 0
    #point_y[0] = 0
    flag = False
    loc_idx = 0
    # print(len(point_x))
    # for i in range(1,len(point_x)):
    #     if point_x[i] <= -355:
    #         flag = True
    #         continue
    #     else:
    #         if flag:
    #             temp_x = point_x[i]
    #             temp_y = point_y[i]
    #             point_x[i] = 0
    #             point_y[i] = 0
    #             flag = False
    #         else:
    #             tp_x = point_x[i]
    #             tp_y = point_y[i]
    #             point_x[i] = point_x[i]-temp_x
    #             point_y[i] = point_y[i]-temp_y
    #             temp_x = tp_x
    #             temp_y = tp_y
    
    # tpx = 0
    # tpy = 0
    # flag = False
    # for i in range(1,len(point_x)):
    #     if point_x[i] <= -355:
    #         flag = True
    #         continue
    #     else:
    #         if flag:
    #             point_x[i] += tpx
    #             point_y[i] += tpy
    #             flag = False
    #         else:
    #             point_x[i] = point_x[i-1]+point_x[i]
    #             point_y[i] = point_y[i-1]+point_y[i]
    #             tpx = point_x[i]

    #             tpy = point_y[i]
    
    df = pd.DataFrame({"x":point_x ,"y":point_y})
    df.to_csv(save_path)

ptx,pty = gazeTracker("C:/Users/won/Desktop/ld.avi","C:/Users/won/Desktop/ld_out.avi")
point_to_csv(ptx,pty,"C:\\Users\\won\\Desktop\\ab.csv")

