from PyQt5 import QtCore, QtWidgets

class Ui_Sisan4(object):
    def setupUi(self, Sisan4):
        Sisan4.setObjectName("Sisan4")
        Sisan4.resize(777, 680)
        Sisan4.setGeometry( 535, 230, 777, 680 )
        self.centralwidget = QtWidgets.QWidget(Sisan4)
        self.centralwidget.setObjectName("centralwidget")
        #self.centralwidget.setStyleSheet(("background-color: grey;"))
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 781, 691))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.resultsBox = QtWidgets.QGroupBox(self.tab)
        self.resultsBox.setGeometry(QtCore.QRect(0, 0, 771, 391))
        #self.resultsBox.setGeometry(QtCore.QRect(0, 230, 771, 391))
        self.resultsBox.setObjectName("resultsBox")
        
        #self.resultsText = QtWidgets.QPlainTextEdit(self.resultsBox)
        #self.resultsText.setGeometry(QtCore.QRect(0, 20, 771, 371))
        #self.resultsText.setObjectName("resultsText")
        
        self.resultsTable = QtWidgets.QTableWidget(self.resultsBox)
        self.resultsTable.setGeometry(QtCore.QRect(0, 20, 771, 371))
        self.resultsTable.setObjectName("resultsTable")
        
        
        self.sizeBox = QtWidgets.QGroupBox(self.tab)
        self.sizeBox.setGeometry(QtCore.QRect(200, 385, 191, 231))
        self.sizeBox.setObjectName("sizeBox")
        self.spinSize_x1 = QtWidgets.QSpinBox(self.sizeBox)
        self.spinSize_x1.setGeometry(QtCore.QRect(30, 140, 31, 22))
        self.spinSize_x1.setObjectName("spinSize_x1")
        self.spinSize_x2 = QtWidgets.QSpinBox(self.sizeBox)
        self.spinSize_x2.setGeometry(QtCore.QRect(80, 140, 31, 22))
        self.spinSize_x2.setObjectName("spinSize_x2")
        self.spinSize_x3 = QtWidgets.QSpinBox(self.sizeBox)
        self.spinSize_x3.setGeometry(QtCore.QRect(130, 140, 31, 22))
        self.spinSize_x3.setObjectName("spinSize_x3")
        self.label_6 = QtWidgets.QLabel(self.sizeBox)
        self.label_6.setGeometry(QtCore.QRect(40, 110, 21, 20))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.sizeBox)
        self.label_7.setGeometry(QtCore.QRect(90, 110, 21, 20))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.sizeBox)
        self.label_8.setGeometry(QtCore.QRect(140, 110, 21, 20))
        self.label_8.setObjectName("label_8")
        self.spinSize_Y = QtWidgets.QSpinBox(self.sizeBox)
        self.spinSize_Y.setGeometry(QtCore.QRect(70, 60, 51, 22))
        self.spinSize_Y.setObjectName("spinSize_Y")
        self.label_9 = QtWidgets.QLabel(self.sizeBox)
        self.label_9.setGeometry(QtCore.QRect(90, 30, 20, 20))
        self.label_9.setObjectName("label_9")
        self.polinomsBox = QtWidgets.QGroupBox(self.tab)
        self.polinomsBox.setGeometry(QtCore.QRect(390, 385, 191, 231))
        self.polinomsBox.setObjectName("polinomsBox")
        self.spinRang_x1 = QtWidgets.QSpinBox(self.polinomsBox, minimum=0, maximum=999)
        self.spinRang_x1.setGeometry(QtCore.QRect(16, 110, 42, 22))
        self.spinRang_x1.setObjectName("spinRang_x1")
        self.spinRang_x2 = QtWidgets.QSpinBox(self.polinomsBox, minimum=0, maximum=999)
        self.spinRang_x2.setGeometry(QtCore.QRect(70, 110, 42, 22))
        self.spinRang_x2.setObjectName("spinRang_x2")
        self.spinRang_x3 = QtWidgets.QSpinBox(self.polinomsBox, minimum=0, maximum=999)
        self.spinRang_x3.setGeometry(QtCore.QRect(130, 110, 42, 22))
        self.spinRang_x3.setObjectName("spinRang_x3")
        self.label_1 = QtWidgets.QLabel(self.polinomsBox)
        self.label_1.setGeometry(QtCore.QRect(30, 80, 21, 20))
        self.label_1.setObjectName("label_1")
        self.label_2 = QtWidgets.QLabel(self.polinomsBox)
        self.label_2.setGeometry(QtCore.QRect(80, 80, 21, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.polinomsBox)
        self.label_3.setGeometry(QtCore.QRect(130, 80, 21, 20))
        self.label_3.setObjectName("label_3")
        self.comboBox = QtWidgets.QComboBox(self.polinomsBox)
        self.comboBox.setGeometry(QtCore.QRect(40, 40, 111, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.spinDelta = QtWidgets.QSpinBox(self.polinomsBox)
        self.spinDelta.setGeometry(QtCore.QRect(110, 160, 41, 22))
        self.spinDelta.setObjectName("spinDelta")
        self.checkBox = QtWidgets.QCheckBox(self.polinomsBox)
        self.checkBox.setGeometry(QtCore.QRect(30, 160, 101, 20))
        self.checkBox.setObjectName("checkBox")
        self.dataBox = QtWidgets.QGroupBox(self.tab)
        self.dataBox.setGeometry(QtCore.QRect(0, 385, 201, 231))
        self.dataBox.setObjectName("dataBox")
        self.spinSize = QtWidgets.QSpinBox(self.dataBox, minimum=0, maximum=10000)
        self.spinSize.setGeometry(QtCore.QRect(111, 40, 81, 22))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinSize.sizePolicy().hasHeightForWidth())
        self.spinSize.setSizePolicy(sizePolicy)
        self.spinSize.setObjectName("spinSize")
        self.viborka = QtWidgets.QLabel(self.dataBox)
        self.viborka.setGeometry(QtCore.QRect(20, 40, 91, 21))
        self.viborka.setObjectName("viborka")
        self.file_x = QtWidgets.QLabel(self.dataBox)
        self.file_x.setGeometry(QtCore.QRect(20, 80, 71, 21))
        self.file_x.setObjectName("file_x")
        self.lineFile_from_x = QtWidgets.QLineEdit(self.dataBox)
        self.lineFile_from_x.setGeometry(QtCore.QRect(80, 80, 101, 22))
        self.lineFile_from_x.setObjectName("lineFile_from_x")
        self.lineFile_from_y = QtWidgets.QLineEdit(self.dataBox)
        self.lineFile_from_y.setGeometry(QtCore.QRect(80, 120, 101, 22))
        self.lineFile_from_y.setObjectName("lineFile_from_y")
        self.file_y = QtWidgets.QLabel(self.dataBox)
        self.file_y.setGeometry(QtCore.QRect(20, 120, 71, 21))
        self.file_y.setObjectName("file_y")
        self.isMulti = QtWidgets.QCheckBox(self.dataBox)
        self.isMulti.setGeometry(QtCore.QRect(40, 160, 141, 20))
        self.isMulti.setObjectName("isMulti")
        self.Button = QtWidgets.QPushButton(self.tab)
        self.Button.setGeometry(QtCore.QRect(0, 600, 771, 41))
        self.Button.setObjectName("Button")
        
        self.predictBox = QtWidgets.QGroupBox(self.tab)
        self.predictBox.setGeometry(QtCore.QRect(580, 385, 191, 231))
        self.predictBox.setObjectName("predictBox")
        
        
        
        self.clearButton = QtWidgets.QPushButton(self.predictBox)
        self.clearButton.setGeometry(QtCore.QRect(30, 160, 135, 28))
        self.clearButton.setObjectName("clearButton")
        self.stopButton = QtWidgets.QPushButton(self.predictBox)
        self.stopButton.setGeometry(QtCore.QRect(30, 190, 65, 28))
        self.stopButton.setObjectName("stopButton")
        self.startButton = QtWidgets.QPushButton(self.predictBox)
        self.startButton.setGeometry(QtCore.QRect(100, 190, 65, 28))
        self.startButton.setObjectName("startButton")
        self.spinSpeed = QtWidgets.QSpinBox(self.predictBox)
        self.spinSpeed.setGeometry(QtCore.QRect(130, 130, 51, 22))
        self.spinSpeed.setObjectName("spinSpeed")
        self.spinSpeed = QtWidgets.QSpinBox(self.predictBox, minimum=0, maximum=10000)
        self.spinSpeed.setGeometry(QtCore.QRect(130, 130, 51, 22))
        
        self.label_17 = QtWidgets.QLabel(self.predictBox)
        self.label_17.setGeometry(QtCore.QRect(30, 130, 91, 21))
        self.label_17.setObjectName("label_17")
        self.spinWindow1 = QtWidgets.QSpinBox(self.predictBox)
        self.spinWindow1.setGeometry(QtCore.QRect(100, 20, 61, 22))

        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinWindow1.sizePolicy().hasHeightForWidth())
        self.spinWindow1.setSizePolicy(sizePolicy)
        self.spinWindow1.setObjectName("spinWindow1")
        self.spinWindow1 = QtWidgets.QSpinBox(self.predictBox, minimum=0, maximum=10000)
        self.spinWindow1.setGeometry(QtCore.QRect(100, 20, 61, 22))
        
        self.label_14 = QtWidgets.QLabel(self.predictBox)
        self.label_14.setGeometry(QtCore.QRect(9, 20, 91, 21))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.predictBox)
        self.label_15.setGeometry(QtCore.QRect(9, 50, 91, 21))
        self.label_15.setObjectName("label_15")
        self.spinWindow2 = QtWidgets.QSpinBox(self.predictBox)
        self.spinWindow2.setGeometry(QtCore.QRect(100, 50, 61, 22))
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinWindow2.sizePolicy().hasHeightForWidth())
        self.spinWindow2.setSizePolicy(sizePolicy)
        self.spinWindow2.setObjectName("spinWindow2")
        self.spinWindow2 = QtWidgets.QSpinBox(self.predictBox, minimum=0, maximum=10000)
        self.spinWindow2.setGeometry(QtCore.QRect(100, 50, 61, 22))
        
        self.label_16 = QtWidgets.QLabel(self.predictBox)
        self.label_16.setGeometry(QtCore.QRect(34, 80, 131, 20))
        self.label_16.setAcceptDrops(False)
        self.label_16.setObjectName("label_16")
        
        #self.radioButton_1 = QtWidgets.QRadioButton(self.predictBox)
        #self.radioButton_1.setGeometry(QtCore.QRect(20, 100, 21, 20))
        #self.radioButton_1.setObjectName("radioButton_1")
        self.radioButton_2 = QtWidgets.QRadioButton(self.predictBox)
        self.radioButton_2.setGeometry(QtCore.QRect(30, 100, 21, 20))
        self.radioButton_2.setObjectName("radioButton_2")
        #self.radioButton_3 = QtWidgets.QRadioButton(self.predictBox)
        #self.radioButton_3.setGeometry(QtCore.QRect(60, 100, 21, 20))
        #self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.predictBox)
        self.radioButton_4.setGeometry(QtCore.QRect(70, 100, 21, 20))
        self.radioButton_4.setObjectName("radioButton_4")
        #self.radioButton_5 = QtWidgets.QRadioButton(self.predictBox)
        #self.radioButton_5.setGeometry(QtCore.QRect(100, 100, 21, 20))
        #self.radioButton_5.setObjectName("radioButton_5")
        self.radioButton_6 = QtWidgets.QRadioButton(self.predictBox)
        self.radioButton_6.setGeometry(QtCore.QRect(110, 100, 21, 20))
        self.radioButton_6.setObjectName("radioButton_6")
        #self.radioButton_7 = QtWidgets.QRadioButton(self.predictBox)
        #self.radioButton_7.setGeometry(QtCore.QRect(140, 100, 21, 20))
        #self.radioButton_7.setObjectName("radioButton_7")
        self.radioButton_8 = QtWidgets.QRadioButton(self.predictBox)
        self.radioButton_8.setGeometry(QtCore.QRect(150, 100, 21, 20))
        self.radioButton_8.setObjectName("radioButton_8")

        
        self.tabWidget.addTab(self.tab, "")
        
        self.resultsBox.raise_()
        self.sizeBox.raise_()
        self.polinomsBox.raise_()
        self.dataBox.raise_()
        self.Button.raise_()
        Sisan4.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Sisan4)
        self.statusbar.setObjectName("statusbar")
        Sisan4.setStatusBar(self.statusbar)
        self.retranslateUi(Sisan4)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Sisan4)        
        
        def add_tab(self, tabs, i):
            self.tab = QtWidgets.QWidget()
            self.tab.setObjectName("tab_"+str(i))
            self.image_label = QtWidgets.QLabel(self.tab)
            self.image_label.setGeometry(QtCore.QRect(10, 0, 531, 521))
            self.image_label.setText("")
            self.image_label.setObjectName("image_label_"+str(i + 1))
            self.tabWidget.addTab(self.tab, "Plot "+str(i + 1))
            tabs.append(self.tab)
        
    def remove_tabs(self, tabs):
            n=len(tabs)
            for i in range(1,n+1):
                to_del = tabs[n-i]
                tabs.remove(to_del)
                to_del.deleteLater()  
        
    def retranslateUi(self, Sisan4):
        _translate = QtCore.QCoreApplication.translate
        Sisan4.setWindowTitle(_translate("Sisan4", "MainWindow"))
        self.resultsBox.setTitle(_translate("Sisan4", "Прогрес"))
        self.sizeBox.setTitle(_translate("Sisan4", "Вектори"))
        self.label_6.setText(_translate("Sisan4", "x1"))
        self.label_7.setText(_translate("Sisan4", "x2"))
        self.label_8.setText(_translate("Sisan4", "x3"))
        self.label_9.setText(_translate("Sisan4", "Y"))
        self.polinomsBox.setTitle(_translate("Sisan4", "Поліноми"))
        self.label_1.setText(_translate("Sisan4", "x1"))
        self.label_2.setText(_translate("Sisan4", "x2"))
        self.label_3.setText(_translate("Sisan4", "x3"))
        self.comboBox.setItemText(0, _translate("Sisan4", "Чебишева"))
        self.comboBox.setItemText(1, _translate("Sisan4", "Ерміта"))
        self.comboBox.setItemText(2, _translate("Sisan4", "Лежандра"))
        self.comboBox.setItemText(3, _translate("Sisan4", "Лагерра"))
        self.checkBox.setText(_translate("Sisan4", "Точність"))
        self.dataBox.setTitle(_translate("Sisan4", "Введення данних"))
        self.viborka.setText(_translate("Sisan4", "Вибірка"))
        self.file_x.setText(_translate("Sisan4", "Файл x"))
        self.file_y.setText(_translate("Sisan4", "Файл y"))
        self.isMulti.setText(_translate("Sisan4", "Мультиплікативна"))
        self.Button.setText(_translate("Sisan4", "Запуск"))
        self.predictBox.setTitle(_translate("Sisan4", "Прогнозування"))
        self.clearButton.setText(_translate("Sisan4", "Очистити історію"))
        self.stopButton.setText(_translate("Sisan4", "Пауза"))
        self.startButton.setText(_translate("Sisan4", "Продовж"))
        self.label_17.setText(_translate("Sisan4", "Швидкість (мс)"))
        self.label_14.setText(_translate("Sisan4", "Вікно періоду"))
        self.label_15.setText(_translate("Sisan4", "Вікно прогнозу"))
        self.label_16.setText(_translate("Sisan4", "Адекватність замірів"))
        #self.radioButton_1.setText(_translate("Sisan4", "RadioButton"))
        self.radioButton_2.setText(_translate("Sisan4", "RadioButton"))
        #self.radioButton_3.setText(_translate("Sisan4", "RadioButton"))
        self.radioButton_4.setText(_translate("Sisan4", "RadioButton"))
        #self.radioButton_5.setText(_translate("Sisan4", "RadioButton"))
        self.radioButton_6.setText(_translate("Sisan4", "RadioButton"))
        #self.radioButton_7.setText(_translate("Sisan4", "RadioButton"))
        self.radioButton_8.setText(_translate("Sisan4", "RadioButton"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Sisan4", "Програма"))
