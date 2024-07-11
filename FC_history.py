import sys
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QListWidget, QTableWidget, \
    QTableWidgetItem, QAbstractItemView, QHeaderView, QLabel, QListWidgetItem, QFrame, QFontDialog, QScrollArea, \
    QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from pymongo import MongoClient
from datetime import datetime


class MessageItemWidget(QWidget):
    def __init__(self, role, content, font):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        frame = QFrame()
        frame.setFrameShape(QFrame.Panel)
        frame.setFrameShadow(QFrame.Raised)
        if role == "user":
            frame.setStyleSheet("color: #000000 ;background-color: #EEF7FF; border-radius: 10px;")
        elif role == "assistant":
            frame.setStyleSheet("color: #FFFFFF ;background-color: #028391; border-radius: 10px;")

        content_label = QLabel(content)
        content_label.setFont(font)
        content_label.setWordWrap(True)  # Enable word wrapping

        frame_layout = QVBoxLayout()
        frame_layout.addWidget(content_label)
        frame.setLayout(frame_layout)

        layout.addWidget(frame)
        self.setLayout(layout)


class MongoViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.mongo_client = MongoClient("mongodb://localhost:27017/")
        self.db = self.mongo_client["chat_database"]
        self.collection = self.db["conversations"]

        self.font = QFont("Roboto", 11, 600)  # Default font and size
        self.font2 = QFont("Roboto", 10, 400)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('MongoDB Viewer')
        self.setGeometry(100, 100, 1200, 804)
        QApplication.setFont(self.font2)

        # Create the main layout
        main_layout = QHBoxLayout()

        # Create the layout for the thread list and exit button
        thread_layout = QVBoxLayout()

        # Create the list widget for thread_id selection
        self.thread_list = QListWidget()
        self.thread_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded)  # Ensure vertical scroll bar appears as needed
        self.thread_list.itemClicked.connect(self.load_thread_data)

        # Add the thread list to the thread layout
        thread_layout.addWidget(self.thread_list)

        # Create the exit button
        exit_button = QPushButton("Exit")
        exit_button.setFixedSize(120, 80)  # Set fixed size for the exit button
        exit_button.setStyleSheet("""
                    QPushButton {
                        font-size: 20pt;
                        font-weight: 700;
                        color: white;
                        background-color: red;
                        border: 5px solid red;
                        border-radius: 12px;
                    }
                    QPushButton:hover {
                        color: red;
                        background-color: white;
                    }
                    QPushButton:pressed {
                        color: red;
                        background-color: white;
                        border: 5px solid red;
                    }
                """)
        exit_button.clicked.connect(self.exit_function)

        # Center the exit button horizontally
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(exit_button)
        button_layout.addStretch()

        # Add the button layout to the thread layout
        thread_layout.addLayout(button_layout)

        # Create a container widget for the thread layout
        thread_widget = QWidget()
        thread_widget.setLayout(thread_layout)
        thread_widget.setFixedWidth(400)  # Set fixed width for thread_widget

        # Add the thread widget to the main layout
        main_layout.addWidget(thread_widget)

        # Create the table for displaying messages
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Role', 'Content', 'Timestamp'])
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.setColumnWidth(0, 100)  # Set width for Role column
        self.table.setColumnWidth(1, 500)  # Set width for Content column
        self.table.setColumnWidth(2, 150)  # Set width for Timestamp column
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Ensure vertical scroll bar appears as needed
        main_layout.addWidget(self.table)

        # Set the main layout to the central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.load_thread_ids()

    def load_thread_ids(self):
        # Load all thread_ids from MongoDB
        threads = self.collection.find({}, {"thread_id": 1, "messages.timestamp": 1})
        for thread in threads:
            thread_id = thread["thread_id"]
            # Get the last timestamp
            last_message = max(thread["messages"], key=lambda x: x["timestamp"]) if thread.get("messages") else {
                "timestamp": "No messages"}
            last_timestamp = last_message["timestamp"]
            self.add_thread_item(thread_id, last_timestamp)

    def add_thread_item(self, thread_id, last_timestamp):
        # Create a widget to hold the thread_id and last_timestamp
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)  # Set spacing between items

        frame = QFrame()
        frame.setFrameShape(QFrame.Panel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setStyleSheet("background-color: {}; border-radius: 10px;".format(self.random_pastel_color()))

        thread_id_label = QLabel(thread_id)
        last_timestamp_label = QLabel(
            datetime.fromtimestamp(last_timestamp).strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_timestamp, (
                int, float)) else last_timestamp)

        frame_layout = QVBoxLayout()
        frame_layout.addWidget(thread_id_label)
        frame_layout.addWidget(last_timestamp_label)
        frame.setLayout(frame_layout)

        layout.addWidget(frame)
        widget.setLayout(layout)

        # Create a QListWidgetItem and set the widget as its item widget
        item = QListWidgetItem(self.thread_list)
        item.setSizeHint(widget.sizeHint())
        self.thread_list.addItem(item)
        self.thread_list.setItemWidget(item, widget)

    def load_thread_data(self, item):
        # Load the messages of the selected thread_id
        selected_widget = self.thread_list.itemWidget(item)
        selected_thread_id = selected_widget.layout().itemAt(0).widget().layout().itemAt(0).widget().text()
        document = self.collection.find_one({"thread_id": selected_thread_id})
        messages = document.get("messages", [])

        # Update the table with messages
        self.table.setRowCount(len(messages))
        for row, message in enumerate(messages):
            self.table.setItem(row, 0, QTableWidgetItem(message["role"]))
            content_widget = MessageItemWidget(message["role"], message["content"], self.font)
            self.table.setCellWidget(row, 1, content_widget)
            self.table.setItem(row, 2, QTableWidgetItem(
                datetime.fromtimestamp(message["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')))

        self.table.resizeRowsToContents()  # Resize rows to fit content

    def random_pastel_color(self):
        """Generate a random pastel color."""
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        return QColor(r, g, b).name()

    def change_font(self):
        font, ok = QFontDialog.getFont(self.font, self)
        if ok:
            self.font = font
            self.load_thread_data(self.thread_list.currentItem())

    def exit_function(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Xác nhận thoát')
        msg_box.setText('Bạn có muốn tắt ứng dụng?')
        # Thêm các nút và đặt nhãn cho chúng
        yes_button = msg_box.addButton(QMessageBox.Yes)
        yes_button.setText('Exit')
        no_button = msg_box.addButton(QMessageBox.No)
        no_button.setText('Cancel')
        for button in msg_box.buttons():
            button.setCursor(Qt.PointingHandCursor)
        # msg_box.setStyleSheet(BoxStyle)
        # Hiển thị hộp thoại và chờ phản hồi
        reply = msg_box.exec()
        if reply == QMessageBox.Yes:
            sys.exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MongoViewer()
    viewer.show()
    sys.exit(app.exec_())
