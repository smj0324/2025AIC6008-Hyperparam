import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt

class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mobilenet Hyperparameter UI")
        self.setGeometry(100, 100, 800, 500)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(self.main_tab(), "Main")
        self.tabs.addTab(self.train_tab(), "Train")
        self.tabs.addTab(self.result_tab(), "Results")

    def main_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Main 화면입니다."))  # 여기는 자유롭게 구성
        widget.setLayout(layout)
        return widget

    def train_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        info_label = QLabel("Username님의 Mobilenet 모델의 ?버전이 학습 중입니다.")
        info_label.setAlignment(Qt.AlignCenter)

        hyper_label = QLabel("현재 설정한 hyperparameter입니다.")
        hyper_label.setAlignment(Qt.AlignLeft)

        # 왼쪽 표 (hyperparameter)
        table1 = QTableWidget(3, 2)
        table1.setHorizontalHeaderLabels(["Param", "Value"])
        table1.setItem(0, 0, QTableWidgetItem("lr"))
        table1.setItem(0, 1, QTableWidgetItem("0.001"))
        table1.setItem(1, 0, QTableWidgetItem("epochs"))
        table1.setItem(1, 1, QTableWidgetItem("10"))
        table1.setItem(2, 0, QTableWidgetItem("batch_size"))
        table1.setItem(2, 1, QTableWidgetItem("32"))

        # 오른쪽 표 (Home/Loss/Accurcy)
        table2 = QTableWidget(0, 3)
        table2.setHorizontalHeaderLabels(["Home", "Loss", "accrucy"])

        hbox = QHBoxLayout()
        hbox.addWidget(table1)
        hbox.addWidget(table2)

        layout.addWidget(info_label)
        layout.addWidget(hyper_label)
        layout.addLayout(hbox)
        widget.setLayout(layout)
        return widget

    def result_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        info_label = QLabel("Username님\nMobilenet 모델의 ?버전 학습을 기반으로 추천하는 하이퍼파라미터 조합이에요.")
        info_label.setAlignment(Qt.AlignLeft)

        hbox = QHBoxLayout()

        # 두 개의 박스
        for _ in range(2):
            box = QVBoxLayout()
            box.addWidget(QLabel("[하이퍼파라미터 추천 이미지 영역]"))
            btn = QPushButton("선택")
            box.addWidget(btn)
            hbox.addLayout(box)

        layout.addWidget(info_label)
        layout.addLayout(hbox)
        widget.setLayout(layout)
        return widget


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MainUI()
    ui.show()
    sys.exit(app.exec_())
