import re
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
from GUI_EMO import Ui_EMO_MainWindow
from openai import OpenAI
from pymongo import MongoClient
from ultralytics import YOLO
from io import BytesIO
from deepface import DeepFace
from FC_recognition_emotion_text import predict_sentiment, model, tokenizer
from FC_calculator_emotion import weight_text, weight_cam, emotion_calculation, change_pos
import markdown2
from bs4 import BeautifulSoup


def extract_text_from_markdown(markdown_text):
    # Chuyển đổi Markdown thành HTML
    html = markdown2.markdown(markdown_text)

    # Sử dụng BeautifulSoup để trích xuất nội dung văn bản từ HTML
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()

    return text

lock_global = None
flag_global = None
text_global = None

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
        video = cv2.VideoCapture(0)
        while True:
            _, frame = video.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_faces = DeepFace.extract_faces(frame_rgb, enforce_detection=False, detector_backend="opencv")
            face_result = faceModel.predict(frame, conf=0.4)
            face_count = 0

            # Xác định cảm xúc
            for info in face_result:
                parameters = info.boxes
                for box in parameters:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    h, w = y2 - y1, x2 - x1
                    flags = frame[y1:y1 + h, x1:x1 + w]
                    objs = DeepFace.analyze(flags, actions=['emotion'], enforce_detection=False,
                                            detector_backend='opencv')
                    emos = []
                    for emo in objs[0]['emotion']:
                        emos.append((objs[0]['emotion'][emo]) / 100)
                    face_count += 1

                    if flag_global == True:
                        print("Thực hiện trong true")
                        for i in range(len(emo_list)):
                            emo_list[i] += emos[i]
                        print(emo_list)
                        emo_str = ', '.join(map(str, emo_list))
                        self.emo_changed_signal.emit(emo_str)

                if flag_global == False:
                        print("Thực hiện trong false")
                        emo_str = ', '.join(map(str, emo_list))

                        self.emo_changed_signal.emit(emo_str)  # Phát tín hiệu khi có giá trị emo mới
                        emo_list = [0, 0, 0, 0, 0, 0, 0]
                        flag_global = True

            # Xác định khuôn mặt
            if len(detected_faces) > 0:
                result = DeepFace.verify(image, frame_rgb, enforce_detection=False, model_name="GhostFaceNet")
                if result["verified"]:
                    name = self.user_name
                    print(name)
                else:
                    name = ""
                    print("Người lạ!")
            else:
                name = ""
                print("Không phát hiện khuôn mặt!")

            self.name_changed_signal.emit(name)

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


class EMO_MainWindow(QMainWindow, Ui_EMO_MainWindow):
    def __init__(self, controller):

        super(EMO_MainWindow, self).__init__()
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
        self.setup_clock()
        self.setup_weather_clock()


        try:
            self.EMO_assistant = openai_client.beta.assistants.retrieve(EMO_ID)
            print(self.EMO_assistant)
        except Exception as e:
            print(f"Error in load assistant:{e}")

        try:
            self.current_thread_EMO, self.conversation_id_EMO = create_new_thread_EMO()
            print(self.current_thread_EMO)
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

        self.switch_function()

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
        try:
            global  flag_global, text_global
            global lock_global
            if not lock_global:
                self.show_tts_running_message()
                return
            print("SEND")
            flag_global = False
            # self.camera_thread.update_flag(flag_global)

            print("Thực hiện trong send_message")
            # Kết thúc quá trình nhập liệu
            self.emotion_cam = self.emotionx.split(', ')
            print("SEND1",self.emotion_cam)
            self.message_text = self.input_text.toPlainText().strip()
            print("SEND2",self.message_text)
            _, self.emotion_text = predict_sentiment(model, tokenizer, self.message)
            print("SEND3",self.emotion_text)
            # self.emotion_text = change_pos(self.emotion_text)
            print("SEND4")
            self.emotion = emotion_calculation(weight_cam, weight_text, self.emotion_cam, self.emotion_text)
            print("SEND5")
            if self.emotion == "angry":
                self.robot_label.setPixmap(QtGui.QPixmap("image/emotion/angry.png"))
            elif self.emotion == "disgust":
                self.robot_label.setPixmap(QtGui.QPixmap("image/emotion/disgust.png"))
            elif self.emotion == "happy":
                self.robot_label.setPixmap(QtGui.QPixmap("image/emotion/happy.png"))
            elif self.emotion == "fear":
                self.robot_label.setPixmap(QtGui.QPixmap("image/emotion/fear.png"))
            elif self.emotion == "sad":
                self.robot_label.setPixmap(QtGui.QPixmap("image/emotion/sad.png"))
            elif self.emotion == "surprise":
                self.robot_label.setPixmap(QtGui.QPixmap("image/emotion/surprise.png"))
            else:
                self.robot_label.setPixmap(QtGui.QPixmap("image/emotion/neutral.png"))

            flag_global = True
            # self.camera_thread.update_flag(flag_global)

            self.response_text.clear()
            self.message = "{" + self.emotion + "} - " + self.message_text
            print("COMBINE MESSAGE",self.message)

            my_thread_message = openai_client.beta.threads.messages.create(
                thread_id=self.current_thread_EMO.id,
                role="user",
                content=self.message,
            )
            print(f"This is the message object: {my_thread_message} \n")

            # Update MongoDB with the user's message
            collection.update_one(
                {"_id": self.conversation_id_EMO},
                {"$push": {"messages": {"role": "user", "content": self.message, "timestamp": time.time()}}}
            )

            # Step 4: Run the Assistant
            my_run = openai_client.beta.threads.runs.create(
                thread_id=self.current_thread_EMO.id,
                assistant_id=EMO_ID,
                instructions=self.EMO_assistant.instructions,
                tools=[{"type": "file_search"}],
                model="gpt-4o-2024-05-13"
            )

            print(f"This is the run object: {my_run} \n")

            # Step 5: Periodically retrieve the Run to check on its status to see if it has moved to completed
            while my_run.status in ["queued", "in_progress"]:
                keep_retrieving_run = openai_client.beta.threads.runs.retrieve(
                    thread_id=self.current_thread_EMO.id,
                    run_id=my_run.id,
                )
                print(f"Run status: {keep_retrieving_run.status}")

                if keep_retrieving_run.status == "completed":
                    print("\n")

                    # Step 6: Retrieve the Messages added by the Assistant to the Thread
                    all_messages = openai_client.beta.threads.messages.list(
                        thread_id= self.current_thread_EMO.id
                    )
                    assistant_message = all_messages.data[0].content[0].text.value
                    self.bot_response = assistant_message
                    # Update MongoDB with the assistant's message
                    collection.update_one(
                        {"_id": self.conversation_id_EMO},
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
            # text_global = self.bot_response
            text_global = extract_text_from_markdown(self.bot_response)
            print("CHECKK",text_global)
            self.speak_thread.start()
            self.input_text.clear()
        except Exception as e:

            print("Chưa nhận được tín hiệu từ luồng emotion_thread!")

            print(f'ERROR in run:{e}')
            self.message_text = self.input_text.toPlainText().strip()
            self.emotion = "neutral"
            self.robot_label.setPixmap(QtGui.QPixmap("image/emotion/neutral.png"))

            flag_global = True
            # self.camera_thread.update_flag(flag_global)

            self.input_text.clear()
            self.response_text.clear()
            self.message = "{" + self.emotion + "} - " + self.message_text
            print("COMBINE MESSAGE", self.message)

            my_thread_message = openai_client.beta.threads.messages.create(
                thread_id=self.current_thread_EMO.id,
                role="user",
                content=self.message,
            )
            print(f"This is the message object: {my_thread_message} \n")

            # Update MongoDB with the user's message
            collection.update_one(
                {"_id": self.conversation_id_EMO},
                {"$push": {"messages": {"role": "user", "content": self.message, "timestamp": time.time()}}}
            )

            # Step 4: Run the Assistant
            my_run = openai_client.beta.threads.runs.create(
                thread_id=self.current_thread_EMO.id,
                assistant_id=EMO_ID,
                instructions=self.EMO_assistant.instructions,
                tools=[{"type": "file_search"}],
                model="gpt-4o-2024-05-13",
                max_completion_tokens=100
            )

            print(f"This is the run object: {my_run} \n")

            # Step 5: Periodically retrieve the Run to check on its status to see if it has moved to completed
            while my_run.status in ["queued", "in_progress"]:
                keep_retrieving_run = openai_client.beta.threads.runs.retrieve(
                    thread_id=self.current_thread_EMO.id,
                    run_id=my_run.id,
                )
                print(f"Run status: {keep_retrieving_run.status}")

                if keep_retrieving_run.status == "completed":
                    print("\n")

                    # Step 6: Retrieve the Messages added by the Assistant to the Thread
                    all_messages = openai_client.beta.threads.messages.list(
                        thread_id=self.current_thread_EMO.id
                    )
                    assistant_message = all_messages.data[0].content[0].text.value
                    self.bot_response = assistant_message
                    # Update MongoDB with the assistant's message
                    collection.update_one(
                        {"_id": self.conversation_id_EMO},
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
            self.bot_response = re.sub(r'【.*?】', '', self.bot_response)
            self.response_text.setText(self.bot_response)

            # Cập nhật text cho SpeakThread
            text_global = self.bot_response
            self.speak_thread.start()
            self.input_text.clear()

    def reset_function(self):
        print("RESET")
        global lock_global
        if not lock_global:
            self.show_tts_running_message()
            return
        self.input_text.clear()
        self.response_text.clear()
        try:
            self.current_thread_EMO, self.conversation_id_EMO = create_new_thread_EMO()
            print(self.current_thread_EMO)
        except Exception as e:
            print(f"Error in load thread:{e}")
        pass

    def switch_function(self):
        print("SWITCH")
        global flag_global, lock_global

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
        msg_box = QMessageBox()
        msg_box.setWindowTitle('UPLOAD')
        msg_box.setText('Bạn muốn tải lên ảnh người dùng hay lịch trình và các thông tin cần nhớ?')
        # Thêm các nút và đặt nhãn cho chúng
        taianh_button = msg_box.addButton(QMessageBox.Yes)
        taianh_button.setText('Tải lên ảnh')
        taifile_button = msg_box.addButton(QMessageBox.Ok)
        taifile_button.setText('Tải lên lịch trình')
        thoat_button = msg_box.addButton(QMessageBox.Apply)
        thoat_button.setText('Thoát')

        for button in msg_box.buttons():
            button.setCursor(Qt.PointingHandCursor)
        msg_box.setStyleSheet(BoxStyle)
        # Hiển thị hộp thoại và chờ phản hồi
        reply = msg_box.exec()
        if msg_box.clickedButton() == taianh_button:
            self.upload_photo()
        elif msg_box.clickedButton() == taifile_button:
            self.upload_task()
        elif msg_box.clickedButton() == thoat_button:
            print("Dialog closed without action out")
        else:
            print("Dialog closed without action")

    def upload_photo(self):
        print("UPLOAD PHOTO")
        global image
        global lock_global
        if not lock_global:
            self.show_tts_running_message()
            return
        # Mở hộp thoại chọn tệp
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Chọn ảnh để tải lên", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            try:
                # Yêu cầu người dùng nhập tên tệp
                text, ok = QInputDialog.getText(None, 'User name', 'Nhập tên người dùng:')
                if ok and text:
                    # Thư mục lưu ảnh
                    save_dir = "uploaded_images"
                    os.makedirs(save_dir, exist_ok=True)

                    # Đặt tên ảnh theo tên người dùng nhập
                    new_file_name = f"{text}.png"  # Thêm phần mở rộng .png
                    new_file_path = os.path.join(save_dir, new_file_name)

                    # Sao chép ảnh vào thư mục chỉ định
                    shutil.copy(file_path, new_file_path)

                    # Đặt lại ảnh nguồn và tên người dùng
                    image = cv2.imread(new_file_path)
                    self.user_lineEdit.setText(text)
                    self.user_label.setPixmap(QtGui.QPixmap(f'uploaded_images/{new_file_name}'))

                    # Hiển thị thông báo thành công
                    QMessageBox.information(None, "Thành công",
                                            f"Ảnh đã được tải lên thành công và lưu tại: {new_file_path}")
                else:
                    QMessageBox.warning(None, "Cảnh báo", "Tên ảnh không được để trống.")
            except Exception as e:
                QMessageBox.critical(None, "Lỗi", f"Đã xảy ra lỗi: {str(e)}")

        pass

    def upload_task(self):
        print("UPLOAD TASK")
        global lock_global
        if not lock_global:
            self.show_tts_running_message()
            return
        global image
        # Mở hộp thoại chọn tệp
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Chọn lịch trình tải lên", "",
                                                   "Doc Files (*.doc *.txt *.json *.docx)", options=options)
        print(file_path)
        if file_path:
            try:
                VECTOR_STORE_ID = "vs_dZ76OOIyz8MjIb5Il7fQe4w8"
                vector_store = openai_client.beta.vector_stores.retrieve(vector_store_id=VECTOR_STORE_ID)

                print(vector_store)
                file_paths = [f"{file_path}"]
                print(file_paths)
                file_streams = [open(path, "rb") for path in file_paths]
                # Ensure the file is opened and closed properly
                file_batch = openai_client.beta.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store.id, files=file_streams
                )

                # You can print the status and the file counts of the batch to see the result of this operation.
                print(file_batch.status)
                print(file_batch.file_counts)

                # Hiển thị thông báo thành công
                QMessageBox.information(None, "Thành công",
                                        f"Dữ liệu đã được tải lên thành công và lưu tại: {VECTOR_STORE_ID}")

            except Exception as e:
                QMessageBox.critical(None, "Lỗi", f"Đã xảy ra lỗi: {str(e)}")

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

class Controller:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.EMO_window = EMO_MainWindow(self)
        # EMO_Class = EMO_MainWindow()

    def show_EMO_window(self):
        self.EMO_window.setWindowTitle('EMO Chatbot by HUST')
        self.EMO_window.setFixedSize(1200,804)
        self.EMO_window.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        desktop_geometry = QDesktopWidget().availableGeometry()
        window_geometry = self.EMO_window.frameGeometry()

        # Calculate the center point of the screen
        center_point = desktop_geometry.center()

        # Move the window to the center of the screen
        window_geometry.moveCenter(center_point)
        self.EMO_window.move(window_geometry.topLeft())
        self.EMO_window.show()


    def run(self):
        self.show_EMO_window()
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    controller = Controller()
    controller.run()


