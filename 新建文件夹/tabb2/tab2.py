# -*- coding: utf-8 -*-
import sys

# Form implementation generated from reading ui file 'F:\xianyu\moxing\tab2.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFrame
from qfluentwidgets import BodyLabel, CardWidget, LineEdit, PushButton, SubtitleLabel, TitleLabel
from tabb2.main import dz2
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(900, 700)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.CardWidget = CardWidget(Form)
        self.CardWidget.setMinimumSize(QtCore.QSize(20, 0))
        self.CardWidget.setObjectName("CardWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.CardWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.TitleLabel = TitleLabel(self.CardWidget)
        self.TitleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.TitleLabel.setObjectName("TitleLabel")
        self.verticalLayout.addWidget(self.TitleLabel)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.LineEdit_11 = LineEdit(self.CardWidget)
        self.LineEdit_11.setObjectName("LineEdit_11")
        self.gridLayout.addWidget(self.LineEdit_11, 7, 1, 1, 1)
        self.BodyLabel_7 = BodyLabel(self.CardWidget)
        self.BodyLabel_7.setObjectName("BodyLabel_7")
        self.gridLayout.addWidget(self.BodyLabel_7, 3, 0, 1, 1)
        self.BodyLabel_17 = BodyLabel(self.CardWidget)
        self.BodyLabel_17.setObjectName("BodyLabel_17")
        self.gridLayout.addWidget(self.BodyLabel_17, 6, 0, 1, 1)
        self.BodyLabel_19 = BodyLabel(self.CardWidget)
        self.BodyLabel_19.setObjectName("BodyLabel_19")
        self.gridLayout.addWidget(self.BodyLabel_19, 7, 0, 1, 1)
        self.BodyLabel_2 = BodyLabel(self.CardWidget)
        self.BodyLabel_2.setObjectName("BodyLabel_2")
        self.gridLayout.addWidget(self.BodyLabel_2, 0, 2, 1, 1)
        self.BodyLabel = BodyLabel(self.CardWidget)
        self.BodyLabel.setObjectName("BodyLabel")
        self.gridLayout.addWidget(self.BodyLabel, 0, 0, 1, 1)
        self.LineEdit_3 = LineEdit(self.CardWidget)
        self.LineEdit_3.setObjectName("LineEdit_3")
        self.gridLayout.addWidget(self.LineEdit_3, 2, 1, 1, 1)
        self.BodyLabel_4 = BodyLabel(self.CardWidget)
        self.BodyLabel_4.setText("")
        self.BodyLabel_4.setObjectName("BodyLabel_4")
        self.gridLayout.addWidget(self.BodyLabel_4, 1, 2, 1, 1)
        self.BodyLabel_18 = BodyLabel(self.CardWidget)
        self.BodyLabel_18.setObjectName("BodyLabel_18")
        self.gridLayout.addWidget(self.BodyLabel_18, 6, 2, 1, 1)
        self.LineEdit_9 = LineEdit(self.CardWidget)
        self.LineEdit_9.setObjectName("LineEdit_9")
        self.gridLayout.addWidget(self.LineEdit_9, 5, 1, 1, 1)
        self.BodyLabel_9 = BodyLabel(self.CardWidget)
        self.BodyLabel_9.setObjectName("BodyLabel_9")
        self.gridLayout.addWidget(self.BodyLabel_9, 4, 0, 1, 1)
        self.BodyLabel_6 = BodyLabel(self.CardWidget)
        self.BodyLabel_6.setObjectName("BodyLabel_6")
        self.gridLayout.addWidget(self.BodyLabel_6, 2, 2, 1, 1)
        self.LineEdit_10 = LineEdit(self.CardWidget)
        self.LineEdit_10.setObjectName("LineEdit_10")
        self.gridLayout.addWidget(self.LineEdit_10, 6, 1, 1, 1)
        self.BodyLabel_20 = BodyLabel(self.CardWidget)
        self.BodyLabel_20.setObjectName("BodyLabel_20")
        self.gridLayout.addWidget(self.BodyLabel_20, 7, 2, 1, 1)
        self.BodyLabel_10 = BodyLabel(self.CardWidget)
        self.BodyLabel_10.setText("")
        self.BodyLabel_10.setObjectName("BodyLabel_10")
        self.gridLayout.addWidget(self.BodyLabel_10, 4, 2, 1, 1)
        self.BodyLabel_16 = BodyLabel(self.CardWidget)
        self.BodyLabel_16.setObjectName("BodyLabel_16")
        self.gridLayout.addWidget(self.BodyLabel_16, 5, 2, 1, 1)
        self.LineEdit_2 = LineEdit(self.CardWidget)
        self.LineEdit_2.setObjectName("LineEdit_2")
        self.gridLayout.addWidget(self.LineEdit_2, 1, 1, 1, 1)
        self.LineEdit_5 = LineEdit(self.CardWidget)
        self.LineEdit_5.setObjectName("LineEdit_5")
        self.gridLayout.addWidget(self.LineEdit_5, 4, 1, 1, 1)
        self.LineEdit_4 = LineEdit(self.CardWidget)
        self.LineEdit_4.setObjectName("LineEdit_4")
        self.gridLayout.addWidget(self.LineEdit_4, 3, 1, 1, 1)
        self.BodyLabel_15 = BodyLabel(self.CardWidget)
        self.BodyLabel_15.setObjectName("BodyLabel_15")
        self.gridLayout.addWidget(self.BodyLabel_15, 5, 0, 1, 1)
        self.BodyLabel_3 = BodyLabel(self.CardWidget)
        self.BodyLabel_3.setObjectName("BodyLabel_3")
        self.gridLayout.addWidget(self.BodyLabel_3, 1, 0, 1, 1)
        self.BodyLabel_5 = BodyLabel(self.CardWidget)
        self.BodyLabel_5.setObjectName("BodyLabel_5")
        self.gridLayout.addWidget(self.BodyLabel_5, 2, 0, 1, 1)
        self.BodyLabel_8 = BodyLabel(self.CardWidget)
        self.BodyLabel_8.setObjectName("BodyLabel_8")
        self.gridLayout.addWidget(self.BodyLabel_8, 3, 2, 1, 1)
        self.LineEdit = LineEdit(self.CardWidget)
        self.LineEdit.setObjectName("LineEdit")
        self.gridLayout.addWidget(self.LineEdit, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.PushButton = PushButton(self.CardWidget)
        self.PushButton.setObjectName("PushButton")
        self.horizontalLayout_2.addWidget(self.PushButton)
        self.PushButton_2 = PushButton(self.CardWidget)
        self.PushButton_2.setObjectName("PushButton_2")
        self.horizontalLayout_2.addWidget(self.PushButton_2)
        self.PushButton_3 = PushButton(self.CardWidget)
        self.PushButton_3.setObjectName("PushButton_3")
        self.horizontalLayout_2.addWidget(self.PushButton_3)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.CardWidget_2 = CardWidget(self.CardWidget)
        self.CardWidget_2.setObjectName("CardWidget_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.CardWidget_2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.SubtitleLabel = SubtitleLabel(self.CardWidget_2)
        self.SubtitleLabel.setObjectName("SubtitleLabel")
        self.horizontalLayout_4.addWidget(self.SubtitleLabel)
        self.LineEdit_6 = LineEdit(self.CardWidget_2)
        self.LineEdit_6.setMinimumSize(QtCore.QSize(20, 0))
        self.LineEdit_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.LineEdit_6.setObjectName("LineEdit_6")
        self.horizontalLayout_4.addWidget(self.LineEdit_6)
        self.verticalLayout.addWidget(self.CardWidget_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(2, 2)
        self.verticalLayout.setStretch(3, 2)
        self.verticalLayout.setStretch(4, 1)
        self.horizontalLayout.addWidget(self.CardWidget)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.TitleLabel.setText(_translate("Form", "模型二-八因素"))
        self.BodyLabel_7.setText(_translate("Form", "地下水位深度"))
        self.BodyLabel_17.setText(_translate("Form", "剪切波速"))
        self.BodyLabel_19.setText(_translate("Form", "震级"))
        self.BodyLabel_2.setText(_translate("Form", "m"))
        self.BodyLabel.setText(_translate("Form", "土层埋深"))
        self.BodyLabel_18.setText(_translate("Form", "m/s"))
        self.BodyLabel_9.setText(_translate("Form", "循环剪应力比"))
        self.BodyLabel_6.setText(_translate("Form", "%"))
        self.BodyLabel_20.setText(_translate("Form", ""))
        self.BodyLabel_16.setText(_translate("Form", "g"))
        self.BodyLabel_15.setText(_translate("Form", "门槛加速度"))
        self.BodyLabel_3.setText(_translate("Form", "修正标贯基数"))
        self.BodyLabel_5.setText(_translate("Form", "细粒含量"))
        self.BodyLabel_8.setText(_translate("Form", "m"))
        self.PushButton.setText(_translate("Form", "清除结果"))
        self.PushButton_2.setText(_translate("Form", "快速计算"))
        self.PushButton_3.setText(_translate("Form", "全部清除"))
        self.SubtitleLabel.setText(_translate("Form", "概率"))






class New_tab2(QFrame):

    def __init__(self):
        super(New_tab2, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.PushButton.clicked.connect(self.clear_results)
        self.ui.PushButton_3.clicked.connect(self.clear)
        self.ui.PushButton_2.clicked.connect(self.dz)

    def dz(self):
        x1 = float(self.ui.LineEdit.text())
        x2 = float(self.ui.LineEdit_2.text())
        x3 = float(self.ui.LineEdit_3.text())
        x4 = float(self.ui.LineEdit_4.text())
        x5 = float(self.ui.LineEdit_5.text())
        x6 = float(self.ui.LineEdit_9.text())
        x7 = float(self.ui.LineEdit_10.text())
        x8 = float(self.ui.LineEdit_11.text())
        res = dz2(x1,x2,x3,x4,x5,x6,x7,x8)
        self.ui.LineEdit_6.setText(str(res))
    def clear_results(self):
        # 清空输入字段
        self.ui.LineEdit_6.clear()

    def clear(self):
        # 定义清除函数
        self.ui.LineEdit.clear()
        self.ui.LineEdit_2.clear()
        self.ui.LineEdit_3.clear()
        self.ui.LineEdit_4.clear()
        self.ui.LineEdit_5.clear()
        self.ui.LineEdit_11.clear()
        self.ui.LineEdit_9.clear()
        self.ui.LineEdit_10.clear()
        self.ui.LineEdit_6.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = New_tab2()
    w.show()

    app.exec()