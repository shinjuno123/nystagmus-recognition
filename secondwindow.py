import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_secondwindow = uic.loadUiType("secondwindow.ui")[0]

class sencondwindow(QDialog,QWidget,form_secondwindow):
    def __init__(self):
        super(sencondwindow,self).__init__()
        self.initUI()
        self.show()
        
        self.widget.setStyleSheet("background-color:rgb(220,220,220)")
        self.btn_home.setStyleSheet("background-image:url(home.png)")
        
    def initUI(self):
        self.setupUi(self)
        self.btn_home.clicked.connect(self.Home)
        
    def Home(self):
        self.close()