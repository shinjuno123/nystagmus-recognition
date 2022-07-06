import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt,QSize
from PyQt5.QtGui import QIcon,QFont
import os,glob
from secondwindow import sencondwindow
import sys
import os
from PyQt5.QtWidgets import *
global filename
global width, height
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
import glob
from video_area import MyVideo
from Custmodel.mainUtility import *

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):

        self.myform = QtWidgets.QFormLayout()

        self.btn_set = QPushButton()

        self.btn_set.clicked.connect(self.clickSet)

        self.label1 = QLabel("  인공지능 진단")
        self.label1.setStyleSheet("QLabel{border-style: outset;border-width: 1px;border-radius: 10px;border-color: #DCDCDC;margin: 1px;}")
    
        self.label1.setFixedHeight(40)
        self.font = QFont()
        self.font.setPointSizeF(15)
        self.label1.setFont(self.font)
        self.list = QListWidget()
        self.list.setStyleSheet(stylesheet(self))

        self.label2 = QLabel("폴더추가")
        self.label2.setStyleSheet(stylesheet(self))

        self.btn_add = QPushButton("추가")
        self.btn_add.setFixedWidth(90)
        self.btn_add.setStyleSheet(stylesheet(self))

        self.btn_add.clicked.connect(self.clickAdd)

        self.btn_predict = QPushButton("예측")
        self.btn_predict.setFixedWidth(90)
        self.btn_predict.setStyleSheet(stylesheet(self))
        self.btn_predict.clicked.connect(self.clickPredict)

        self.btn_del = QPushButton("삭제")
        self.btn_del.setFixedWidth(90)
        self.btn_del.setStyleSheet(stylesheet(self))
        self.btn_del.clicked.connect(self.clickDel)

        self.btn_clear = QPushButton("모두 삭제")
        self.btn_clear.setFixedWidth(90)
        self.btn_clear.setStyleSheet(stylesheet(self))
        self.btn_clear.clicked.connect(self.clickClr)

        self.label3 = QLabel("예측 결과")
        self.label3.setStyleSheet(stylesheet(self))

        self.mygroupbox = QScrollArea()
        self.mygroupbox.setStyleSheet(stylesheet(self))

        #self.pred_v_layout = QVBoxLayout()
       
 

        self.btn_predict.setEnabled(False)
        self.btn_del.setEnabled(False)

        # mainLayout
        self.main_h_layout = QHBoxLayout()
        self.sub_navi_layout = QVBoxLayout()
        self.sub_navi_box = QGroupBox()
        self.sub_navi_box.setStyleSheet("""QGroupBox { background-color:#D0D0D0; border:#D0D0D0; padding:5px; }""")
        self.sub_navi_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.sub_navi_box.setLayout(self.sub_navi_layout)
        self.sub_content_layout = QVBoxLayout()
        self.sub_content_layout.setContentsMargins(10,0,10,30)
        self.main_h_layout.addWidget(self.sub_navi_box)
        #self.main_h_layout.addLayout(self.sub_navi_layout)
        self.main_h_layout.addLayout(self.sub_content_layout)
        self.setLayout(self.main_h_layout)
        self.setGeometry(int(width/2-(width*0.3/2)),int(height/2-(height*0.6/2)),int(width*0.3),int(height*0.6))

        # HBox : 3-button (add, predict, delete, clear)
        self.btn_set.setFixedSize(50,50)
        self.btn_set.setIcon(QIcon("../front_end/set.png"))
        self.btn_set.setStyleSheet("""QPushButton { background-color:#D0D0D0; border:#D0D0D0; padding:5px; }""")
        self.btn_set.setIconSize(QSize(40,40))
        self.sub_navi_layout.addWidget(self.btn_set)

        self.sub_content_layout.addWidget(self.label1)
        self.sub_content_layout.addWidget(self.label2)
        self.sub_content_layout.addWidget(self.list)
        self.h = QHBoxLayout()
        self.h.addWidget(self.btn_add)
        self.h.addWidget(self.btn_predict)
        self.h.addWidget(self.btn_del)
        self.h.addWidget(self.btn_clear)

 

        self.sub_content_layout.addLayout(self.h)
        # VBox : Menu setting button
        self.v = QVBoxLayout()
        #self.v setStyleSheet("background-color:black;")

        # grid에는 setStyleSheet가 안됨
        #self.v.setStyleSheet("background-color: rgb(223, 223, 223)")
        self.label1.setStyleSheet("background-color: rgb(255, 255, 255)")
        self.sub_content_layout.addWidget(self.label3)
        self.sub_content_layout.addWidget(self.mygroupbox)

    def clickClr(self):
        [self.myform.removeRow(0) for x in range(self.myform.rowCount())]

    #  추가 버튼
    def clickAdd(self):   
        self.file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.videolist = glob.glob(self.file+"/*.avi")
        if self.videolist:
            cnt = len(self.videolist)
            row = self.list.count()
            for i in range(row, row+cnt):
                if self.videolist[i-row] != "All Files (*)":
                    if len(self.list.findItems(os.path.basename(self.videolist[i-row]),Qt.MatchExactly))==0:
                        self.list.addItem(os.path.basename(self.videolist[i-row]))
                        # 파일위치저장
                        self.basepath = os.path.dirname(self.videolist[i-row])
            self.list.setCurrentRow(0)
        if len(self.videolist) == 0:
            self.btn_predict.setEnabled(False)
            self.btn_del.setEnabled(False)
        else:
            self.btn_predict.setEnabled(True)
            self.btn_del.setEnabled(True)
    
    def openVideo(self):
        sending_button = self.sender()
        video_name = str(sending_button.objectName())
        self.graph_area = MyVideo(self.file+"/"+video_name,width,height)
        self.graph_area.exec_()

    # 삭제 버튼
    def clickDel(self):
        row = self.list.currentRow()
        del_item = self.list.takeItem(row)
        if self.list.count() == 0:
            self.btn_predict.setEnabled(False)
            self.btn_del.setEnabled(False)
        else:
            self.btn_predict.setEnabled(True)
            self.btn_del.setEnabled(True)
    def clickSet(self):
        self.hide()
        second = sencondwindow()
        second.exec()
        self.show()

    # 예측 버튼 함수
    def clickPredict(self):
        #result
        [self.myform.removeRow(0) for x in range(self.myform.rowCount())]
        row = self.list.count()
        self.myform.setVerticalSpacing(20)
        self.groupbox = QtWidgets.QGroupBox()
        self.pred_path = self.file+"/pred/"
        self.filelist = []
        if len(self.videolist) != 0 :
            for video in self.videolist:
                if len(self.list.findItems(os.path.basename(video),Qt.MatchExactly))==1:
                    #search_graph_file = self.file+"/pred/"+os.path.basename(video).replace(".avi","_graph.avi")
                    search_gaze_file = self.file+"/pred/"+os.path.basename(video).replace(".avi","_gaze.avi")
                    search_csv_file = self.file+"/pred/"+os.path.basename(video).replace(".avi","_out.csv")
                    if os.path.exists(search_gaze_file) == True and os.path.exists(search_csv_file)==True:#os.path.exists(search_graph_file) == True and
                        self.filename_lbl = QtWidgets.QLabel()
                        self.filename_lbl.setFixedWidth(int((self.main_h_layout.geometry().width()/7)*4.65))
                        self.filename_btn = QtWidgets.QPushButton()
                        self.filename_btn.setText("예측 결과 보기")
                        self.filename_btn.setObjectName(os.path.basename(video))
                        self.filename_btn.clicked.connect(self.openVideo)
                        self.filename_lbl.setText(os.path.basename(video))
                        self.filename_lbl.setStyleSheet(stylesheet(self))
                        self.filelist.append((self.filename_lbl,self.filename_btn))
                    else:
                        point_x,point_y = gazeTrackerV2(video_path=video,video_path_result=search_gaze_file)
                        mkcsv(point_x=point_x,point_y=point_y,path=search_csv_file)

                        f = open(self.file+"/pred/"+os.path.basename(video).replace(".avi","_out.txt"), 'w')
                        [f.write(str(x)) for x in predictClass("./weight/Classifier.h5",video,3,point_x,point_y)]
                        f.close()
                        # graphing(search_csv_file,search_graph_file)
                        self.filename_lbl = QtWidgets.QLabel()
                        self.filename_lbl.setFixedWidth(int((self.main_h_layout.geometry().width()/7)*4.65))
                        self.filename_btn = QtWidgets.QPushButton()
                        self.filename_btn.setText("예측 결과 보기")
                        self.filename_btn.setObjectName(os.path.basename(video))
                        self.filename_btn.clicked.connect(self.openVideo)
                        self.filename_lbl.setText(os.path.basename(video))
                        self.filelist.append((self.filename_lbl,self.filename_btn))
        
        for i in range(len(self.filelist)):
            self.myform.addRow(self.filelist[i][0])
            self.myform.addRow(self.filelist[i][1])
        self.groupbox.setLayout(self.myform)
        self.mygroupbox.setWidget(self.groupbox)


def stylesheet(self):
        return """
   QPushButton{
    background-color:#E8E8E8;
    border-style: outset;
    border-width: 1px;
    border-radius: 10px;
    border-color: #DCDCDC;
    font: 14px;
    margin: 1px;
    padding: 5px;}
    

   QLabel{
    font: 15px;
    font-family:arial;
    margin:5px;
    }

   QListWidget{
    background-color:white;
    border-style: outset;
    border-width: 1px;
    border-radius: 20px;
    border-color: #DCDCDC;
    font: 14px;
    margin: 1px;
    padding: 5px;}

    QScrollArea{
    background-color:white;
    border-style: outset;
    border-width: 1px;
    border-radius: 20px;
    border-color: #DCDCDC;
    font: 14px;
    margin: 1px;
    padding: 5px;}
"""       

if __name__ == "__main__":
    app = QApplication(sys.argv)
    screen_size = app.desktop().screenGeometry()
    width,height = screen_size.width(),screen_size.height()
    screen_rect = app.desktop()
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()

