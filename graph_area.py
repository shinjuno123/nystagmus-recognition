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


class MyGraph(QDialog,QWidget):

    def __init__(self,filename,width,height):
        super(MyGraph,self).__init__()
        self.initUI(filename,width,height)
        self.show()

    def initUI(self,filename,width,height):
        
        # Main Interface -> v_layout
        self.vlayout =QVBoxLayout()
        self.setLayout(self.vlayout)
        self.setGeometry(0,200, width, int(height*0.5))
        self.filename = filename
        self.setWindowTitle("Graph about [" + os.path.basename(self.filename).split("_out.")[0]+"]")
        data = genfromtxt(self.filename, delimiter=',')
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
        self.explain.setFixedWidth(120)

        self.range_change_start_lte = QLineEdit()
        self.range_change_start_lte.setFixedWidth(50)

        self.rng = QLabel(" ~ ")
        self.rng.setFixedWidth(30)
        self.rng.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.range_change_end_lte = QLineEdit()
        self.range_change_end_lte.setFixedWidth(50)

        self.range_change_btn = QPushButton("범위 설정")
        self.range_change_btn.setFixedWidth(80)
        self.range_change_btn.clicked.connect(self.range_change)
        self.range_rollback_btn = QPushButton("되돌리기")
        self.range_rollback_btn.setFixedWidth(80)
        self.range_rollback_btn.clicked.connect(self.range_rollback)

        self.indent = QLabel()
        self.indent.setFixedWidth(950)
        self.hlayout = QHBoxLayout()
        self.hlayout.addWidget(self.explain)
        self.hlayout.addWidget(self.range_change_start_lte)
        self.hlayout.addWidget(self.rng)
        self.hlayout.addWidget(self.range_change_end_lte)
        self.hlayout.addWidget(self.range_change_btn)
        self.hlayout.addWidget(self.range_rollback_btn)
        self.hlayout.addWidget(self.indent)
        self.vlayout.addLayout(self.hlayout)


        self.vlayout.addWidget(self.toolbar)
        self.vlayout.addWidget(self.canvas)

        self.pt_range = np.linspace(0,len(self.xp)/30,len(self.xp))
        print(self.xp)
        print(self.yp)
        
        plt.subplots_adjust(left=0.05, bottom=0.1,  right=0.95, top=0.9, wspace=0.2, hspace=0.35)
        self.ax.plot(self.pt_range,self.xp[:,0],linewidth=0.3)
        self.ax2.plot(self.pt_range,self.yp[:,0],'r',linewidth=0.3)
        self.ax.set_title("X Point")
        self.ax2.set_title("Y Point")
        self.ax2.set_xlabel('Time(Sec)')
        
        plt.xticks([x/30 for x in range(0,len(self.pt_range),30)])
        self.ax.legend()
        self.ax2.legend()
        self.canvas.draw()
        
    def range_rollback(self):
        self.ax.clear()
        self.ax2.clear()
        self.ax.plot(self.pt_range,self.xp[:,0],linewidth=0.3)
        self.ax2.plot(self.pt_range,self.yp[:,0],'r',linewidth=0.3)
        plt.xticks([x/30 for x in range(0,len(self.pt_range),30)])
        self.ax.legend()
        self.ax2.legend()
        self.ax.set_title("X Point")
        self.ax2.set_title("Y Point")
        self.ax2.set_xlabel('Time(Sec)')
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
            self.ax.set_title("X Point")
            self.ax2.set_title("Y Point")
            self.ax2.set_xlabel('Time(Sec)')
            self.canvas.draw()
