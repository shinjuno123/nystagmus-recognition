from statistics import mode
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
import numpy as np
import os
import numpy as np 
import json
import csv

with open(os.getcwd() + '\\Custmodel\\setting.json') as FILE:
    DATA = json.load(FILE)
    CLASSES = DATA['CLASSES'] 
    CLASSES_LENGTH = DATA['CLASSES_LENGTH']
    SEGMENT_RANGE = DATA['SEGMENT_RANGE']



#  def graphing(csv_path,save_path):
#     datas = genfromtxt(csv_path, delimiter=',')
#     scaler = MinMaxScaler()
#     x_data= scaler.fit_transform(np.reshape(datas[1:,0],(-1,1)))
#     x_data2= scaler.fit_transform(np.reshape(datas[1:,1],(-1,1)))
#     cnt = 0
#     x = []
#     y = []
#     w,h = 1000,204
#     fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#     out = cv2.VideoWriter(save_path, fourcc, 30, (w,h))

#     while 120>cnt:
#         y = x_data[0:cnt,0]
#         y2 =  x_data2[0:cnt,0]
#         x = [x/30 for x in range(len( x_data[0:cnt,0]))]
#         f = plt.figure()
#         ax1 = plt.subplot(2, 1, 1)
#         ax1.tick_params(bottom= False,labelbottom = False)
#         plt.plot(x,y,linewidth=0.3)
#         plt.axis([0,4,0,1])
#         plt.xticks([0,1,2,3])
#         ax2 = plt.subplot(2, 1, 2, sharex=ax1)
#         plt.plot(x,y2,'r',linewidth=0.3)
#         plt.ylim(0,1)
#         plt.close()
#         f_arr = figure_to_array(f)
#         f_arr = cv2.resize(f_arr[50:,161:1151],(w,h))
#         f_arr = cv2.cvtColor(f_arr,cv2.COLOR_RGBA2BGR)
#         out.write(f_arr)
#         cnt+=1

#     while len(x_data)>cnt:
#         x = np.linspace((cnt-120)/30,cnt/30,120)
#         y2 =  x_data2[cnt-120:cnt,0]
#         y = x_data[cnt-120:cnt,0]
#         f = plt.figure()
#         ax1 = plt.subplot(2, 1, 1)
#         ax1.tick_params(bottom= False,labelbottom = False)
#         plt.plot(x,y,linewidth=0.3)
#         plt.ylim(0,1)
#         plt.xticks([(cnt-120)//30,(cnt-90)//30,(cnt-60)//30,(cnt-30)//30,cnt//30])
#         ax2 = plt.subplot(2, 1, 2, sharex=ax1)
#         plt.plot(x,y2,'r',linewidth=0.3)
#         plt.close()
#         f_arr = figure_to_array(f)
#         f_arr = cv2.resize(f_arr[50:,205:1108],(w,h))
#         f_arr = cv2.cvtColor(f_arr,cv2.COLOR_RGBA2BGR)
#         out.write(f_arr)
#         cnt+=1

#     out.release()
#     cv2.destroyAllWindows()



def gazeTrackerV2(video_path=None,video_path_result=None,drawable=True,writable =True,w=320,h=120) :
    #windows support codec
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    cap = cv2.VideoCapture(video_path)
    fps = 15
    size_x=w//2
    size_y=120
    c=3
    # save video, fourcc -> windows input codec
    out = cv2.VideoWriter(video_path_result, fourcc, fps, (w, h))
    idx = 1
    ret, frame = cap.read()
    point_x = []
    point_y = []
    ll =[]
    frame_idx = 1
    flag = False
    while ret:
        max_len_x = 14
        max_len_y = 14
        frames = [frame[:120,:160],frame[:120,160:]]
        if frame_idx % 2 == 1:
            for frame in frames:
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                gray = cv2.normalize(gray, None, 0, 50, cv2.NORM_MINMAX)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                ret,bimage = cv2.threshold(gray,3,255,cv2.THRESH_BINARY_INV)
                mode = cv2.RETR_TREE
                method =  cv2.CHAIN_APPROX_SIMPLE
                contours,_ = cv2.findContours(bimage,mode,method)

                if len(contours) == 0:
                    point_x.append(-355)
                    point_y.append(-355)
                    flag = True

                for c in contours:
                    _,_,w,h = cv2.boundingRect(c)
                    if drawable == True:
                        (cX,cY),radius = cv2.minEnclosingCircle(c)
                        if cv2.contourArea(c) >= ((12*3.141592)**2)/9:
                            cX,cY, radius = int(cX), int(cY),int(radius)
                            if max_len_x < (w*0.5):
                                max_len_x = w//2
                            if max_len_y < (h*0.5):
                                max_len_y = h//2
                            r = (math.atan2(int(size_y/2)-cY,int(size_x/2)-cX)*180/math.pi)
                            cv2.circle(frame,(cX,cY),2,(0,255,0),-1)
                            cv2.circle(frame,(cX,cY),max_len_x,(0,0,255),2)
                    if idx%2 == 0:
                        if flag:
                            flag = False
                        else:
                            point_x.append(cX)
                            point_y.append(cY)
                if writable == True:
                    if idx %2 == 0:
                        out.write(cv2.hconcat([ll[len(ll)-1],frame]))
                        idx+=1
                    else:
                        ll.append(frame)
                        idx+=1
        frame_idx += 1
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return point_x, point_y


def mkcsv(point_x, point_y,path):
    df = pd.DataFrame({"x":point_x ,"y":point_y})
    df.to_csv(path)

def point_to_csv(point_x, point_y,path):
    temp_x = point_x[0]
    temp_y = point_y[0]
    tp_x = 0
    tp_y = 0
    point_x[0] = 0
    point_y[0] = 0
    flag = False
    for i in range(1,len(point_x)):
        if point_x[i] <= -355:
            flag = True
            continue
        else:
            if flag:
                temp_x = point_x[i]
                temp_y = point_y[i]
                point_x[i] = 0
                point_y[i] = 0
                flag = False
            else:
                tp_x = point_x[i]
                tp_y = point_y[i]
                point_x[i] = point_x[i]-temp_x
                point_y[i] = point_y[i]-temp_y
                temp_x = tp_x
                temp_y = tp_y
    
    tpx = 0
    tpy = 0
    flag = False
    for i in range(1,len(point_x)):
        if point_x[i] <= -355:
            flag = True
            continue
        else:
            if flag:
                point_x[i] += tpx
                point_y[i] += tpy
                flag = False
            else:
                point_x[i] = point_x[i-1]+point_x[i]
                point_y[i] = point_y[i-1]+point_y[i]
                tpx = point_x[i]
                tpy = point_y[i]
    
    df = pd.DataFrame({"x":point_x ,"y":point_y})
    df.to_csv(path)


def getModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(36,return_sequences=True), input_shape=(15, 2)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1024,activation = "relu"))
    model.add(tf.keras.layers.Dense(512,activation = "relu"))
    model.add(tf.keras.layers.Dense(128,activation = "relu"))
    model.add(tf.keras.layers.Dense(3,activation = "softmax"))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0004)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
    model.summary()
    return model


def makePredictCSV(video_path,model,sliced_point):
    #[rt_nystagmus, normal, lt_nystagmus]
    print(sliced_point.shape)
    preds = model.predict(sliced_point)
    results = []

    for pred in preds:
        tmp = [0,0,0]
        max_value = np.amax(pred)
        max_index = np.where(pred == max_value)[0][0]
        tmp[max_index] = 1
        results.append(tmp)

    rt_nystagmus = [result[0] for result in results]
    normal = [result[1] for result in results]
    lt_nystagmus = [result[2] for result in results]
    df = pd.DataFrame({"rt_nystagmus":rt_nystagmus ,"normal":normal,"lt_nystagmus":lt_nystagmus})
    df.to_csv(video_path.split("\\")[0] + "/pred/"+ video_path.split("\\")[-1].replace(".avi","_pred.csv"))


def ImgSlicer(video_path = None,point_x=None,point_y=None,model=None):
    sliced_point =[]
    sliced_img= []
    sliced_pred = []
    index = 0
    center_point = np.array((point_y,point_x)).T
    # ex) (1000,120,320,3) => 1000/5(segment_range)=> 200-(15/5)-1 = 198 => (198,15,120,320,3)
    loop = center_point.shape[0] //5 - 2 
    
    model_wait = 30
    for _ in range(int(loop)):
    # segment_range = 5 => [[0~15],[5,20],[10,25]…]
        sliced_point.append(center_point[(index*SEGMENT_RANGE):(index*SEGMENT_RANGE)+15])
        index=index+1
    
    makePredictCSV(video_path,model,np.array(sliced_point))



    return np.array(sliced_point),np.array(sliced_img)





def makeContextVector(model = None,video_list = None,behavior_class=None,approval_range=3,point_x=None,point_y=None) -> List:
    # initialize
    behavior_dict={}
    for c in behavior_class:
        behavior_dict[c] = [-1 for _ in range(CLASSES_LENGTH)]
    patient_behavior_dict={}
  
    # make Sliced Video (N,15,320,120,3)
    for video in video_list:
        sliced_point,_ =ImgSlicer(video,point_x=point_x,point_y=point_y,model=model)
        result = model.predict(sliced_point)
        preds = np.asarray(result,dtype=np.float16)
        patient_behavior_dict[os.path.basename(video)]=preds
    #patient_behavior_dict -> {bl.avi:[[0.5,0,5],[0.7,0.3]....](Prediction in 15segment)}
  
    lst = [0 for _ in range(CLASSES_LENGTH)] #[0,0,0]
    lst[CLASSES.index("normal")]=1
  
    lst = [0 for _ in range(CLASSES_LENGTH)] #[0,0,0]

    for key in patient_behavior_dict.keys(): #[rt_nystagmus, normal, lt_nystagmus]
        store=[patient_behavior_dict[key][x].argmax() for x in range(0,len(patient_behavior_dict[key]))]
        store = np.array(store)
        for i in range(len(patient_behavior_dict[key])-approval_range):
            if np.where(store[i:i+approval_range]==store[i],True,False).all() ==True:
                lst[patient_behavior_dict[key][i].argmax()] = 1
                behavior_dict[key.split('\\')[-1]] = lst
        lst = [0 for _ in range(CLASSES_LENGTH)]
    
    context_vector = list(behavior_dict.values())
    return context_vector


def predictClass(model_weight_path = None,video_path = None, approval_range = 3,point_x =None,point_y=None) -> List:
    model = getModel()
    model.load_weights(model_weight_path)
    video_path_list = glob.glob(video_path)
    behavior_class = [os.path.basename(x) for x in video_path_list]
    ctv =  makeContextVector(model,video_path_list,behavior_class,approval_range,point_x,point_y)

    return ctv
