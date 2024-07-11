import subprocess
import sys, os, shutil, time
import cv2
import requests
import pygame
import threading
import markdown
import speech_recognition as sr
from PyQt5.QtGui import QImage, QPixmap
from gtts import gTTS
from PyQt5.QtWidgets import QApplication,QFileDialog, QInputDialog, QMainWindow, QMessageBox, QLineEdit, QDesktopWidget, QTableWidgetItem, QAbstractItemView
from PyQt5.QtCore import Qt, QTimer, QDateTime, QThread, pyqtSignal
from PyQt5 import QtGui
from PyQt5 import QtCore
from GUI_DOCTOR import Ui_DOCTOR_MainWindow
from openai import OpenAI
from pymongo import MongoClient
from ultralytics import YOLO
from io import BytesIO
from FC_skin_detect import load_model, predict, get_label

import torch

model_path = 'model/model_weights.pth'
image_path = 'image/skin/captured_image.jpg'

# Load the model
model = load_model(model_path=model_path)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lock_global = False
flag_global = False
text_global = False

faceModel = YOLO('model/yolov8n-face.pt')
image = cv2.imread("image/admin/image_admin.png")

telegram_bot_token = ''
telegram_chat_id = ''

# Connect to MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["chat_database"]
collection = db["conversations"]

# Initialize OpenAI client
openai_client = OpenAI(api_key="")
EMO_ID = "asst_Z7wIX1goi3Pgv9pv21fSG4oN"
DOCTOR_ID = "asst_mQsY2Up5deKj46HUdts1Lrzt"

def create_new_thread_EMO():
    new_thread = openai_client.beta.threads.create(
        tool_resources={
            "file_search": {
                "vector_store_ids": ["vs_dZ76OOIyz8MjIb5Il7fQe4w8"]
            },

        },
    )
    print(f"This is the new thread object: {new_thread} \n")
    # Create a new document for the new conversation
    conversation_id = collection.insert_one({"thread_id": new_thread.id, "messages": []}).inserted_id
    return new_thread, conversation_id

def create_new_thread_DOCTOR():
    new_thread = openai_client.beta.threads.create()
    print(f"This is the new thread object: {new_thread} \n")
    # Create a new document for the new conversation
    conversation_id = collection.insert_one({"thread_id": new_thread.id, "messages": []}).inserted_id
    return new_thread, conversation_id

def getWeather():
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": "Hanoi",
        "appid": '',
        "units": "metric",
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200:
            weather = response.json()
            icon_id = weather['weather'][0]['icon']
            temp = weather['main']['temp']
            description = weather['weather'][0]['description']
            city = weather['name']
            country = weather['sys']['country']
            icon_url = f"http://openweathermap.org/img/wn/{icon_id}@2x.png"
            return (icon_url,temp,description,city,country)
        else:
            return {"error": f"Failed to fetch weather data for Hanoi"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Request error: {e}"}

BoxStyle = """
            QMessageBox {
                background-color: white;
                font-size: 28px;

            }
            QPushButton {
                background-color: #ED1A2E;
                color: white;
                border-radius: 10px;
                padding: 15px 30px; 
                font-size: 20px; 
            }
        """

class CameraThread(QThread):
    emo_changed_signal = pyqtSignal(str)
    name_changed_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        print("INIT MICRO")

        self.user_name = ""
        global flag_global


    def update_username(self, new_username):
        self.user_name = new_username

    def run(self):
        global flag_global
        emo_list = [0, 0, 0, 0, 0, 0, 0]
        # video = cv2.VideoCapture(0)


class MicroThread(QThread):
    message_received = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        print("INIT MICRO")
        self.running = False
        self.recognizer = sr.Recognizer()
        global lock_global


    def run(self):
        global lock_global
        with sr.Microphone() as source:
            while True:
                print("Thực hiện trong MicroThread")
                if self.running:
                    if lock_global:
                        print("Listening...")
                        self.recognizer.adjust_for_ambient_noise(source)
                        print("CHECK1")
                        try:
                            audio = self.recognizer.listen(source)
                        except Exception as e:
                            print(f"Except in listen {e}")
                        print("CHECK2")
                        try:
                            message = self.recognizer.recognize_google(audio, language='vi-VN')
                            print(message)
                            self.message_received.emit(message)
                        except sr.UnknownValueError:
                            print("Sorry, I could not understand what you said.")
                        except sr.RequestError as e:
                            print(f"Sorry, I could not understand what you said.{e}")
                else:
                    print("Thoát khỏi MicroThread")
                    break

    def start_micro(self):
        self.running = True

    def stop_micro(self):
        self.running = False

class SpeakThread(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = ""

        global lock_global
        print("INIT SPEAK THREAD")



    def setnewText(self, text):
        self.text = text
        print("------------***-------------")

    def run(self):
        global lock_global, text_global
        print("1")
        lock_global = False
        with openai_client.audio.speech.with_streaming_response.create(model="tts-1",
                                                                voice="shimmer",
                                                                speed=1.5,
                                                                input=text_global,
                                                                ) as response:
            response.stream_to_file("speech.mp3")
        print("2")
        pygame.init()
        pygame.mixer.music.load("speech.mp3")
        print("3")
        pygame.mixer.music.play()
        print("4")
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
        print("5")
        os.remove("speech.mp3")
        print("6")
        lock_global = True
        print("7")


class DOCTOR_MainWindow(QMainWindow, Ui_DOCTOR_MainWindow):
    def __init__(self, controller):

        super(DOCTOR_MainWindow, self).__init__()
        self.bot_response = None
        self.emotion_text = None
        self.name = None
        self.emotion_cam = None
        self.setupUi(self)
        self.controller = controller

        self.index = 1

        self.message = ""
        self.message_text = ""
        self.user_name = "ADMIN"
        self.emotionx = ""
        self.skin_text.setText("")
        self.precision_label.setText("100")

        global lock_global, flag_global
        lock_global = True
        flag_global = True
        self.send_button.clicked.connect(self.send_function)
        self.reset_button.clicked.connect(self.reset_function)
        self.switch_button.clicked.connect(self.switch_function)
        self.history_button.clicked.connect(self.history_function)
        self.upload_button.clicked.connect(self.upload_function)
        self.exit_button.clicked.connect(self.exit_function)
        self.emer_button.clicked.connect(self.emer_function)
        self.input_text.textChanged.connect(self.check_input_start)
        self.capture_button.clicked.connect(self.capture_image)
        # self.input_text.returnPressed.connect(self.send_function)
        self.setup_clock()
        self.setup_weather_clock()
        self.timer_cam = QTimer(self)
        self.timer_cam.timeout.connect(self.update_frame)

        self.cap = cv2.VideoCapture(0)

        # Bắt đầu timer
        self.timer_cam.start(30)

        try:
            self.DOCTOR_assistant = openai_client.beta.assistants.retrieve(DOCTOR_ID)
            print(self.DOCTOR_assistant)
        except Exception as e:
            print(f"Error in load assistant:{e}")

        try:
            self.current_thread_DOCTOR, self.conversation_id_DOCTOR = create_new_thread_DOCTOR()
            print(self.current_thread_DOCTOR)
        except Exception as e:
            print(f"Error in load thread:{e}")



        # Luồng camera
        self.camera_thread = CameraThread()
        # self.camera_thread.update_flag(self.flag)
        self.camera_thread.update_username(self.user_name)
        self.camera_thread.start()
        # self.camera_thread.emo_changed_signal.connect(self.send_function) # Kết nối tín hiệu để cập nhật biến emo, boss
        self.camera_thread.name_changed_signal.connect(self.update_name) # Bắt buộc phải gửi tín hiệu đến send_message để thực hiện liên tục
        # Luồng micro
        self.micro_thread = MicroThread()
        # self.micro_thread.update_lock(self.lock)
        self.micro_thread.start()
        self.micro_thread.message_received.connect(self.get_message)

        self.camera_thread.emo_changed_signal.connect(self.get_emotion)
        # Luồng speak

        self.speak_thread = SpeakThread()
        # self.speak_thread.update_lock(self.lock)
        # self.DOCTOR_window = DOCTOR_MainWindow(self)
        # self.history_window = history_MainWindow(self)
        # self.upload_window = upload_MainWindow(self)

        # self.lock = threading.Lock()
        # self.start_all_services()
        # self.init_doctor()
        self.switch_function()

    def show_tts_running_message(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Thông báo')
        msg_box.setText('Đang chạy TTS')
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Chuyển đổi hình ảnh từ BGR (OpenCV) sang RGB (PyQt)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Chuyển đổi hình ảnh thành QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Hiển thị hình ảnh trên video_label
            self.webcam_label.setPixmap(QPixmap.fromImage(q_img))

    def capture_image(self):
        if not lock_global:
            self.show_tts_running_message()
            return
        ret, frame = self.cap.read()
        if ret:
            # Chuyển đổi hình ảnh từ BGR (OpenCV) sang RGB (PyQt)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Chuyển đổi hình ảnh thành QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Hiển thị hình ảnh trên captured_label
            self.webcam_label_2.setPixmap(QPixmap.fromImage(q_img))

            # Lưu hình ảnh
            save_path = "image/skin/captured_image.jpg"
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Lưu hình ảnh
            cv2.imwrite(save_path, frame)
            print("Da luu anh")

            predicted_label_idx, max_probability = predict(model, image_path, device)
            predicted_label = get_label(predicted_label_idx)

            # Print the prediction
            print(f'Predicted label: {predicted_label}, with confidence: {max_probability:.2f}%')
            self.skin_text.setText(predicted_label)
            pre = f"{max_probability:.2f}"
            self.reset_function()
            self.precision_label.setText(pre)

            self.input_text.setText(predicted_label)
            self.send_function()



    def check_input_start(self):
        global  flag_global
        if self.input_text.toPlainText().strip():
            flag_global = True
            # self.camera_thread.update_flag(flag_global)

    def update_emostr(self,emostr):
        self.emotion = emostr
    def update_name(self, name):  # Phương thức để cập nhật nhãn với giá trị boss mới
        self.name = name
        self.user_lineEdit.setText(name)

    def get_message(self, message):
        print("Thực hiện trong get_message")
        self.message = message
        self.input_text.setText(message)
        print("message: ", message)
        self.send_function()

    def get_emotion(self, emo_str):
        print("Thực hiện trong get_emo")
        self.emotionx = emo_str
        print("emo_str: ", emo_str)

    def send_function(self):
        global  flag_global, text_global, lock_global
        if not lock_global:
            self.show_tts_running_message()
            return

        if self.skin_text.text() == "":
            self.init_doctor()
            return
        try:

            print("SEND")

            flag_global = False
            self.message_text = self.input_text.toPlainText().strip()
            # self.camera_thread.update_flag(flag_global)
            print("Thực hiện trong send_message")

            flag_global = True
            # self.camera_thread.update_flag(flag_global)

            self.response_text.clear()
            self.message = self.message_text
            print("COMBINE MESSAGE",self.message)

            my_thread_message = openai_client.beta.threads.messages.create(
                thread_id=self.current_thread_DOCTOR.id,
                role="user",
                content=self.message,
            )
            print(f"This is the message object: {my_thread_message} \n")

            # Update MongoDB with the user's message
            collection.update_one(
                {"_id": self.conversation_id_DOCTOR},
                {"$push": {"messages": {"role": "user", "content": self.message, "timestamp": time.time()}}}
            )

            # Step 4: Run the Assistant
            my_run = openai_client.beta.threads.runs.create(
                thread_id=self.current_thread_DOCTOR.id,
                assistant_id=DOCTOR_ID,
                instructions=self.DOCTOR_assistant.instructions,
                model="gpt-4o-2024-05-13"
            )

            print(f"This is the run object: {my_run} \n")

            # Step 5: Periodically retrieve the Run to check on its status to see if it has moved to completed
            while my_run.status in ["queued", "in_progress"]:
                keep_retrieving_run = openai_client.beta.threads.runs.retrieve(
                    thread_id=self.current_thread_DOCTOR.id,
                    run_id=my_run.id,
                )
                print(f"Run status: {keep_retrieving_run.status}")

                if keep_retrieving_run.status == "completed":
                    print("\n")

                    # Step 6: Retrieve the Messages added by the Assistant to the Thread
                    all_messages = openai_client.beta.threads.messages.list(
                        thread_id= self.current_thread_DOCTOR.id
                    )
                    assistant_message = all_messages.data[0].content[0].text.value
                    self.bot_response = assistant_message
                    # Update MongoDB with the assistant's message
                    collection.update_one(
                        {"_id": self.conversation_id_DOCTOR},
                        {"$push": {
                            "messages": {"role": "assistant", "content": assistant_message, "timestamp": time.time()}}}
                    )
                    print("------------------------------------------------------------ \n")
                    print(f"User: {self.message}")
                    print(f"Assistant: {assistant_message}")

                    break
                elif keep_retrieving_run.status in ["queued", "in_progress"]:
                    pass
                else:
                    print(f"Run status: {keep_retrieving_run.status}")
                    break
            self.response_text.setHtml(markdown.markdown(self.bot_response))
            # self.response_text.setText(self.bot_response)
            # Cập nhật text cho SpeakThread
            text_global = self.bot_response
            self.speak_thread.start()
            while True:
                print("Check")
                if lock_global:
                    self.input_text.clear()
                    break

        except Exception as e:

            print("Chưa nhận được tín hiệu từ luồng emotion_thread!")

            print(f'ERROR in run:{e}')
            self.message_text = self.input_text.toPlainText().strip()


            flag_global = True
            # self.camera_thread.update_flag(flag_global)

            self.input_text.clear()
            self.response_text.clear()
            self.message = self.message_text
            print("COMBINE MESSAGE", self.message)

            my_thread_message = openai_client.beta.threads.messages.create(
                thread_id=self.current_thread_DOCTOR.id,
                role="user",
                content=self.message,
            )
            print(f"This is the message object: {my_thread_message} \n")

            # Update MongoDB with the user's message
            collection.update_one(
                {"_id": self.conversation_id_DOCTOR},
                {"$push": {"messages": {"role": "user", "content": self.message, "timestamp": time.time()}}}
            )

            # Step 4: Run the Assistant
            my_run = openai_client.beta.threads.runs.create(
                thread_id=self.current_thread_DOCTOR.id,
                assistant_id=DOCTOR_ID,
                instructions=self.DOCTOR_assistant.instructions,
                tools=[{"type": "file_search"}],
                model="gpt-4o-2024-05-13"
            )

            print(f"This is the run object: {my_run} \n")

            # Step 5: Periodically retrieve the Run to check on its status to see if it has moved to completed
            while my_run.status in ["queued", "in_progress"]:
                keep_retrieving_run = openai_client.beta.threads.runs.retrieve(
                    thread_id=self.current_thread_DOCTOR.id,
                    run_id=my_run.id,
                )
                print(f"Run status: {keep_retrieving_run.status}")

                if keep_retrieving_run.status == "completed":
                    print("\n")

                    # Step 6: Retrieve the Messages added by the Assistant to the Thread
                    all_messages = openai_client.beta.threads.messages.list(
                        thread_id=self.current_thread_DOCTOR.id
                    )
                    assistant_message = all_messages.data[0].content[0].text.value
                    self.bot_response = assistant_message
                    # Update MongoDB with the assistant's message
                    collection.update_one(
                        {"_id": self.conversation_id_DOCTOR},
                        {"$push": {
                            "messages": {"role": "assistant", "content": assistant_message, "timestamp": time.time()}}}
                    )
                    print("------------------------------------------------------------ \n")
                    print(f"User: {self.message}")
                    print(f"Assistant: {assistant_message}")

                    break
                elif keep_retrieving_run.status in ["queued", "in_progress"]:
                    pass
                else:
                    print(f"Run status: {keep_retrieving_run.status}")
                    break

            self.response_text.setText(self.bot_response)

            # Cập nhật text cho SpeakThread
            text_global = self.bot_response
            self.speak_thread.start()
            self.input_text.clear()

    def reset_function(self):
        global lock_global
        if not lock_global:
            self.show_tts_running_message()
            return
        print("RESET")
        self.input_text.clear()
        self.response_text.clear()
        try:
            self.current_thread_DOCTOR, self.conversation_id_DOCTOR = create_new_thread_DOCTOR()
            print(self.current_thread_DOCTOR)
        except Exception as e:
            print(f"Error in load thread:{e}")
        pass

    def switch_function(self):
        print("SWITCH")
        global flag_global, lock_global
        global lock_global
        if not lock_global:
            self.show_tts_running_message()
            return
        if self.index == 1:
            print("1")
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("image/icon/off.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.switch_button.setIcon(icon)
            self.switch_label.setPixmap(QtGui.QPixmap("image/icon/keyboard.png"))
            self.input_text.setReadOnly(False)
            if self.micro_thread.isRunning():
                self.micro_thread.stop_micro()
                self.micro_thread.start()
            self.index = 0
            flag_global = True
            # self.camera_thread.update_flag(flag_global)

        else:
            print("2")
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("image/icon/on.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
            self.switch_button.setIcon(icon)
            self.switch_label.setPixmap(QtGui.QPixmap("image/icon/micro.png"))
            self.input_text.setReadOnly(True)
            if not self.micro_thread.isRunning():
                self.micro_thread.start_micro()
                self.micro_thread.start()
            self.index = 1
            flag_global = True
            # self.camera_thread.update_flag(flag_global)


    def history_function(self):
        print("HISTORY")
        global lock_global
        if not lock_global:
            self.show_tts_running_message()
            return
        subprocess.run([sys.executable, 'FC_history.py'])


    def upload_function(self):
        print("UPLOAD")
        global lock_global
        if not lock_global:
            self.show_tts_running_message()
            return
        global image
        # Mở hộp thoại chọn tệp
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Chọn ảnh để tải lên", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            try:
                predicted_label_idx, max_probability = predict(model, file_path, device)
                predicted_label = get_label(predicted_label_idx)
                self.webcam_label_2.setPixmap(QtGui.QPixmap(file_path))
                # Print the prediction
                print(f'Predicted label: {predicted_label}, with confidence: {max_probability:.2f}%')
                self.skin_text.setText(predicted_label)
                pre = f"{max_probability:.2f}"
                self.reset_function()
                self.precision_label.setText(pre)
                self.input_text.setText(predicted_label)
                self.send_function()

            except Exception as e:
                QMessageBox.critical(None, "Lỗi", f"Đã xảy ra lỗi: {str(e)}")

        pass

    def emer_function(self):
        print("EMERGENCY")
        global telegram_chat_id
        global telegram_bot_token
        apiToken = telegram_bot_token
        chatID = telegram_chat_id

        apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

        try:
            response = requests.post(apiURL, json={ 'chat_id': chatID,
                                                    'text': '''*TIN NHẮN CẢNH BÁO NGUY HIỂM, TÔI CẦN TRỢ GIÚP*''',
                                                    'parse_mode': 'Markdown'})
            print(response.text)
        except Exception as e:
            print(e)
        pass

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
        msg_box.setStyleSheet(BoxStyle)
        # Hiển thị hộp thoại và chờ phản hồi
        reply = msg_box.exec()
        if reply == QMessageBox.Yes:
            sys.exit()

    def setup_clock(self):
        self.update_clock()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)

    def setup_weather_clock(self):
        self.update_weather()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_weather)
        self.timer.start(1000*3600)
    def update_clock(self):
        current_datetime = QDateTime.currentDateTime()
        formatted_time = current_datetime.toString("hh:mm")
        self.time_label.setText(formatted_time)
        formatted_day = current_datetime.toString("dddd, dd/MM/yyyy")

        # Chuyển đổi thứ từ tiếng Anh sang tiếng Việt
        days_in_vietnamese = {
            "Monday": "T2",
            "Tuesday": "T3",
            "Wednesday": "T4",
            "Thursday": "T5",
            "Friday": "T6",
            "Saturday": "T7",
            "Sunday": "CN"
        }
        for english_day, vietnamese_day in days_in_vietnamese.items():
            formatted_day = formatted_day.replace(english_day, vietnamese_day)
        self.day_label.setText(formatted_day)

    def update_weather(self):
        weather_result = getWeather()
        try:
            wt_url, wt_temp, wt_des, wt_city, wt_country = weather_result
        except:
            wt_url, wt_temp, wt_des, wt_city, wt_country = (None, 25, None, "Hanoi", None)
        self.temp_label.setText(f'{wt_temp}')
        if wt_city == "Hanoi":
            self.location_label.setText('TP. Hà Nội')
        else:
            self.location_label.setText(f'{wt_city}')

        if wt_url == None:
            print("Khong co thong tin")
        else:
            wt_response = requests.get(wt_url)
            image = QImage()
            image.loadFromData(BytesIO(wt_response.content).read())

            # Convert QImage to QPixmap
            pixmap = QPixmap(image)

            # Update the label with the QPixmap
            self.weather_label.setPixmap(pixmap)

    def init_doctor(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Thông báo')
        msg_box.setText('Hãy chụp ảnh lên hoặc tải ảnh lên!')
        # Thêm các nút và đặt nhãn cho chúng
        yes_button = msg_box.addButton(QMessageBox.Yes)
        yes_button.setText('OK')
        for button in msg_box.buttons():
            button.setCursor(Qt.PointingHandCursor)
        msg_box.setStyleSheet(BoxStyle)
        # Hiển thị hộp thoại và chờ phản hồi
        reply = msg_box.exec()


class Controller:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.DOCTOR_window = DOCTOR_MainWindow(self)
        # DOCTOR_Class = DOCTOR_MainWindow()

    def show_DOCTOR_window(self):
        self.DOCTOR_window.setWindowTitle('DOCTOR Chatbot by HUST')
        self.DOCTOR_window.setFixedSize(1200,804)
        self.DOCTOR_window.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        desktop_geometry = QDesktopWidget().availableGeometry()
        window_geometry = self.DOCTOR_window.frameGeometry()

        # Calculate the center point of the screen
        center_point = desktop_geometry.center()

        # Move the window to the center of the screen
        window_geometry.moveCenter(center_point)
        self.DOCTOR_window.move(window_geometry.topLeft())
        self.DOCTOR_window.show()


    def run(self):
        self.show_DOCTOR_window()
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    controller = Controller()
    controller.run()


