from PyQt5.QtWidgets import *
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import QDir, Qt, QUrl,QTime,QRect,QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import glob,os
from cv2 import split
from matplotlib.collections import LineCollection
from matplotlib.widgets import Widget
import numpy as np
import pandas
global width, height
global filename
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from graph_area import MyGraph
import sys
from tkinter import E
from PyQt5.QtWidgets import * #(QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QTextEdit)
from PyQt5.QtGui import QPalette, QKeySequence, QIcon,QColor
from PyQt5.QtCore import QDir, Qt, QUrl,QTime,QRect,QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import glob,os
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from matplotlib.colors import ListedColormap, BoundaryNorm


        


class MyVideo(QDialog,QWidget):

    def __init__(self,filename,width,height):
        super().__init__()
        self.initUI(filename,width,height)
        self.setGeometry(0,50,width,int(height*0.95))
        self.show()

    def initUI(self,filename,width,height):
        #video area
        self.mediaplayer = QMediaPlayer(None,QMediaPlayer.VideoSurface)
        self.mediaplayer_gaze = QMediaPlayer(None,QMediaPlayer.VideoSurface)
        self.videowidget = QVideoWidget()
        self.videowidget_gaze = QVideoWidget()
        self.width = width
        self.height = height
        self.filename = filename
        # Main Interface -> v_layout
        self.v_layout =QGridLayout()
        
        #video set Area(original,gaze_Tracker) in h
        self.h = QHBoxLayout()
        self.h.addWidget(self.videowidget)
        self.h.addWidget(self.videowidget_gaze)
        self.h.geometry()
        self.v_layout.addLayout(self.h,0,0,2,1)
        
   

        #video set Area(graph) in v
        self.v = QHBoxLayout()
        self.v_layout.addLayout(self.v,3,0)

        #video handling Area(play,stop,backward,forward) in adjust uint
        self.slow_btn = QPushButton()
        self.slow_btn.setFixedWidth(70)
        self.slow_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.slow_btn.clicked.connect(self.slow_play)
        self.slow_btn.setEnabled(True)

        self.fast_btn = QPushButton()
        self.fast_btn.setFixedWidth(70)
        self.fast_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.fast_btn.clicked.connect(self.fast_play)
        self.fast_btn.setEnabled(True)

        self.playbtn = QPushButton()
        self.playbtn.setFixedWidth(70)
        self.playbtn.setEnabled(False)
        self.playbtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playbtn.clicked.connect(self.play_video)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0,100)
        self.slider.valueChanged.connect(self.setPosition)
        self.slider.setSingleStep(200)
        self.slider.setPageStep(100)
        self.slider.setAttribute(Qt.WA_TranslucentBackground, True)
        self.slider.setStyleSheet(stylesheet(self))
        self.slider.setTickInterval(10)
        

        self.lbl = QLineEdit('00:00:00')
        self.lbl.setReadOnly(True)
        self.lbl.setFixedWidth(65)
        self.lbl.setUpdatesEnabled(True)
        self.lbl.selectionChanged.connect(lambda: self.lbl.setSelection(0, 0))
        
        self.elbl = QLineEdit('00:00:00')
        self.elbl.setReadOnly(True)
        self.elbl.setFixedWidth(65)
        self.elbl.setUpdatesEnabled(True)
        self.elbl.selectionChanged.connect(lambda: self.elbl.setSelection(0, 0))

        self.splbl = QLabel("속도 : 1.0x")
        self.splbl.setFixedWidth(65)

        adjust_uint = QHBoxLayout()
        adjust_uint.addWidget(self.slow_btn)
        adjust_uint.addWidget(self.playbtn)
        adjust_uint.addWidget(self.fast_btn)
        adjust_uint.addWidget(self.splbl)
        adjust_uint.addWidget(self.lbl)
        adjust_uint.addWidget(self.slider)
        adjust_uint.addWidget(self.elbl)

        self.v_layout.addLayout(adjust_uint,3,0)
        self.setWindowTitle('Video View Area')
        self.setGeometry(0, 0, width, height)
        self.setLayout(self.v_layout)
        self.show()

        menu_layout = QVBoxLayout()
        self.video_list = QListWidget()
        self.video_list.setStyleSheet(stylesheet(self))
        if os.path.isdir(filename):
            self.basepath = filename
        else:
            self.basepath = os.path.dirname(filename)
        for video in glob.glob(self.basepath+"/*avi"):
            idx = glob.glob(self.basepath+"/*avi").index(video)
            self.video_list.insertItem(idx,os.path.basename(video))
        self.video_list.itemDoubleClicked.connect(self.dbClickList)

        menu_layout.addWidget(self.video_list)
        self.v_layout.addLayout(menu_layout,0,1,0,3)
        self.mediaplayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(filename)))

        filename_gaze = self.basepath+"/pred/"+os.path.basename(filename).replace(".avi","_gaze.avi")
        self.mediaplayer_gaze.setMedia(
                    QMediaContent(QUrl.fromLocalFile(filename_gaze)))


        self.playbtn.setEnabled(True)
        self.mediaplayer.setVideoOutput(self.videowidget)
        self.mediaplayer_gaze.setVideoOutput(self.videowidget_gaze)
        self.mediaplayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaplayer.positionChanged.connect(self.positionChanged)
        self.mediaplayer.durationChanged.connect(self.durationChanged)
        self.mediaplayer.error.connect(self.handleError)
        # self.mediaplayer_gaze.stateChanged.connect(self.mediaStateChanged)
        # self.mediaplayer_gaze.positionChanged.connect(self.positionChanged)
        # self.mediaplayer_gaze.durationChanged.connect(self.durationChanged)
        self.shortcut = QShortcut(QKeySequence("x"), self)
        self.shortcut.activated.connect(self.forwardSlider)
        self.shortcut = QShortcut(QKeySequence("z"), self)
        self.shortcut.activated.connect(self.backSlider)
        self.shortcut = QShortcut(Qt.Key_Space, self)
        self.shortcut.activated.connect(self.play_video)
        
        items = [self.video_list.item(x) for x in range(self.video_list.count())]
        for item in items:
            if item.text() == os.path.basename(filename):
                self.video_list.setCurrentItem(item)
                break
        self.vlayout =QVBoxLayout()

        self.widget = QWidget()
        self.widget.setLayout(self.vlayout)
        self.widget.setFixedHeight(400)
        self.v_layout.addWidget(self.widget,2,0)

        self.myfilename = self.basepath+"/pred/"+os.path.basename(filename).replace(".avi","_out.csv")
        data = genfromtxt(self.myfilename, delimiter=',')
        scaler = MinMaxScaler()
        self.xp= scaler.fit_transform(np.reshape(data[1:,1],(-1,1)))
        self.yp= scaler.fit_transform(np.reshape(data[1:,2],(-1,1)))


        self.fig = plt.figure()

        self.ax = plt.subplot(2,1,1)
        self.ax2 = plt.subplot(2,1,2,sharex=self.ax)

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        #adjust uint layout
        self.explain = QLabel("범위를 입력하세요.(초)")
        self.explain.setFixedWidth(130)

        self.range_change_start_lte = QLineEdit()
        self.range_change_start_lte.setFixedWidth(50)

        self.rng = QLabel("  ~  ")
        self.rng.setFixedWidth(50)
        self.rng.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.range_change_end_lte = QLineEdit()
        self.range_change_end_lte.setFixedWidth(50)

        self.range_change_btn = QPushButton("범위 설정")
        self.range_change_btn.setFixedWidth(80)
        self.range_change_btn.clicked.connect(self.range_change)
        self.range_rollback_btn = QPushButton("되돌리기")
        self.range_rollback_btn.setFixedWidth(80)
        self.range_rollback_btn.clicked.connect(self.range_rollback)

        self.hlayout = QHBoxLayout()
        self.hlayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.hlayout.addWidget(self.explain)
        self.hlayout.addWidget(self.range_change_start_lte)
        self.hlayout.addWidget(self.rng)
        self.hlayout.addWidget(self.range_change_end_lte)
        self.hlayout.addWidget(self.range_change_btn)
        self.hlayout.addWidget(self.range_rollback_btn)      
        self.vlayout.addLayout(self.hlayout)


        self.vlayout.addWidget(self.toolbar)
        self.vlayout.addWidget(self.canvas)

        self.pt_range = np.linspace(0,len(self.xp)/15,len(self.xp))

        color_collection = ['r','g','b']

        # ['rt_nystagmus','lt_nystagmus','normal','up_nystagmus','down_nystagmus']
        self.predcsv_path = self.basepath+"/pred/"+os.path.basename(filename).replace(".avi","_pred.csv")

        predcsv = pandas.read_csv(self.predcsv_path)[['rt_nystagmus','normal','lt_nystagmus']]


        if predcsv.iloc[0,0] == 1:
            self.ax.plot(self.pt_range[0:16],self.xp[0:16,0],'r',linewidth=0.3)
        elif predcsv.iloc[0,1] == 1:
            self.ax.plot(self.pt_range[0:16],self.xp[0:16,0],'g',linewidth=0.3)
        else:
            self.ax.plot(self.pt_range[0:16],self.xp[0:16,0],'b',linewidth=0.3)
            
        xp_idx = 15
        for i in range(1,predcsv.shape[0]):
            pred = list(predcsv.iloc[i,:])
            print([xp_idx,xp_idx+5])
            if pred[0] == 1:
                self.ax.plot(self.pt_range[xp_idx:xp_idx+6],self.xp[xp_idx:xp_idx+6,0],'r',linewidth=0.3)
            elif pred[1] == 1:
                self.ax.plot(self.pt_range[xp_idx:xp_idx+6],self.xp[xp_idx:xp_idx+6,0],'g',linewidth=0.3)
            else:
                self.ax.plot(self.pt_range[xp_idx:xp_idx+6],self.xp[xp_idx:xp_idx+6,0],'b',linewidth=0.3)

            xp_idx += 5
        
        plt.subplots_adjust(left=0.02, bottom=0.2,  right=0.98, top=0.9, wspace=0.2, hspace=0.55)

        # self.ax.plot(self.pt_range[:10],self.xp[:10,0],'r',linewidth=0.3)
        # self.ax.plot(self.pt_range[10:],self.xp[10:,0],'b',linewidth=0.3)

        self.ax2.plot(self.pt_range,self.yp[:,0],'g',linewidth=0.3)
        self.ax.set_title("X Point",fontsize=8)
        self.ax2.set_title("Y Point",fontsize=8)
        self.ax2.set_xlabel('Time(Sec)',fontsize=8)
        plt.xticks([x/15 for x in range(0,len(self.pt_range),15)])
        self.ax.legend()
        self.ax2.legend()
        self.canvas.draw()

    def handleError(self):
        self.playbtn.setEnabled(False)
        print("Error: ", self.mediaplayer.errorString())

    def dbClickList(self):
        items = self.video_list.selectedItems()
        self.video_list.setCurrentItem(items[len(items)-1])
        item = items[len(items)-1].text()
        self.mediaplayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(self.basepath+"/"+item)))

        filename_gaze = self.basepath+"/pred/"+os.path.basename(item).replace(".avi","_gaze.avi")
        self.mediaplayer_gaze.setMedia(
                    QMediaContent(QUrl.fromLocalFile(filename_gaze)))


    def range_rollback(self):
        self.ax.clear()
        self.ax2.clear()
        self.ax.plot(self.pt_range,self.xp[:,0],linewidth=0.3)
        self.ax2.plot(self.pt_range,self.yp[:,0],'r',linewidth=0.3)
        plt.xticks([x/30 for x in range(0,len(self.pt_range),30)])
        self.ax.legend()
        self.ax2.legend()
        self.ax.set_title("X Point", fontsize=8)
        self.ax2.set_title("Y Point",fontsize=8)
        self.ax2.set_xlabel('Time(Sec)',fontsize=8)
        self.range_change_start_lte.setText("")
        self.range_change_end_lte.setText("")
        self.canvas.draw()
    
    def range_change(self):
        range_start = int(self.range_change_start_lte.text())
        range_end = int(self.range_change_end_lte.text())
        if range_start - range_end > 0:
            range_start,range_end = range_end,range_start
        if range_start*30<=len(self.pt_range) and range_end*30<=len(self.pt_range) and range_start>=0 and range_end>=0:
            changed_range = self.pt_range[range_start*30:(range_end*30)+1]
            self.ax.clear()
            self.ax2.clear()
            self.ax.plot(changed_range,self.xp[range_start*30:(range_end*30)+1,0],linewidth=0.3)
            self.ax2.plot(changed_range,self.yp[range_start*30:(range_end*30)+1,0],'r',linewidth=0.3)
            plt.xticks([x/30 for x in range(range_start*30,range_end*30+1,30)])
            self.ax.legend()
            self.ax2.legend()
            self.ax.set_title("X Point", fontsize=8)
            self.ax2.set_title("Y Point", fontsize=8)
            self.ax2.set_xlabel('Time(Sec)',fontsize=8)
            self.canvas.draw()
        
    def slow_play(self):
        if self.mediaplayer.playbackRate()> 0.1:
            self.mediaplayer.setPlaybackRate(self.mediaplayer.playbackRate()-0.1)
            self.mediaplayer_gaze.setPlaybackRate(self.mediaplayer.playbackRate()-0.1)
            self.slow_btn.setEnabled(True)
            self.fast_btn.setEnabled(True)
            self.splbl.setText("속도 : "+str(round(self.mediaplayer.playbackRate(),1))+"x")
        else:
            self.slow_btn.setEnabled(False)

    def fast_play(self):
        if self.mediaplayer.playbackRate()< 2.0:
            self.mediaplayer.setPlaybackRate(self.mediaplayer.playbackRate()+0.1)
            self.mediaplayer_gaze.setPlaybackRate(self.mediaplayer.playbackRate()+0.1)
            self.fast_btn.setEnabled(True)
            self.slow_btn.setEnabled(True)
            self.splbl.setText("속도 : "+str(round(self.mediaplayer.playbackRate(),1))+"x")
        else:
            self.fast_btn.setEnabled(False)

    def forwardSlider(self):
        self.mediaplayer.setPosition(self.mediaplayer.position() + 200)
        self.mediaplayer_gaze.setPosition(self.mediaplayer_gaze.position() + 200)

    def backSlider(self):
        self.mediaplayer.setPosition(self.mediaplayer.position() - 200)
        self.mediaplayer_gaze.setPosition(self.mediaplayer_gaze.position() - 200)

    def play_video(self):
        if self.mediaplayer.state() == QMediaPlayer.PlayingState:
            self.mediaplayer.pause()
            self.mediaplayer_gaze.pause()
        else:
            self.mediaplayer.play()
            self.mediaplayer_gaze.play()
     
    def positionChanged(self, position):
        self.slider.setValue(position)
        mtime = QTime(0,0,0,0)
        mtime = mtime.addMSecs(self.mediaplayer.position())
        time = mtime.toString("hh:mm:ss").split(":")
        
        range_start = (int(time[1])*60)+int(time[2])
        range_end = (int(time[1])*60)+int(time[2])+4
        if range_start*15<=len(self.pt_range) and range_end*15<=len(self.pt_range) and range_start>=0 and range_end>=0:
            changed_range = self.pt_range[range_start*15:(range_end*15)+1]
            self.ax.clear()
            self.ax2.clear()
            self.ax.plot(changed_range,self.xp[range_start*15:(range_end*15)+1,0],linewidth=0.3)
            self.ax2.plot(changed_range,self.yp[range_start*15:(range_end*15)+1,0],'r',linewidth=0.3)
            self.ax.set_ylim([0, 1])
            self.ax2.set_ylim([0, 1])
            plt.xticks([x/15 for x in range(range_start*15,range_end*15+1,15)])
            self.ax.set_title("X Point", fontsize=8)
            self.ax2.set_title("Y Point", fontsize=8)
            self.ax2.set_xlabel('Time(Sec)',fontsize=8)
            self.canvas.draw()
        self.lbl.setText(mtime.toString())

    def durationChanged(self, duration):
        self.slider.setRange(0, duration)
        mtime = QTime(0,0,0,0)
        mtime = mtime.addMSecs(self.mediaplayer.duration())
       
        self.elbl.setText(mtime.toString())

    def setPosition(self, position):
        self.mediaplayer.setPosition(position)
        self.mediaplayer_gaze.setPosition(position)

    def mediaStateChanged(self, state):
        if self.mediaplayer.state() == QMediaPlayer.PlayingState:
            self.playbtn.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playbtn.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

def stylesheet(self):
    return """
QListWidget
{
background: #FFFFFF;
color: #000000;
border: 0px solid #076100;
font-size: 15pt;
font-weight: bold;
text-align: center;
}
    """