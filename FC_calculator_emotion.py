import numpy as np

weight_cam = 0.8
weight_text = 0.2

id2label = {
  0:'angry',
  1:'disgust',
  2:'happy',
  3:'fear',
  4:'sad',
  5:'surprise',
  6:'neutral'
}

def change_pos(emo_text):
    emo_text[0], emo_text[1], emo_text[2], emo_text[3], emo_text[4], emo_text[5], emo_text[6] = emo_text[0], emo_text[1], emo_text[3], emo_text[2], emo_text[5], emo_text[6], emo_text[4]
    return emo_text
def emotion_calculation(weight_cam, weight_text, emo_cam, emo_text):
    print("Thực hiện trong emotion calculation")
    emo_cam = [float(x) for x in emo_cam]
    emo_cam = [x / sum(emo_cam) for x in emo_cam]
    emo_cam = np.array(emo_cam)
    emo_text = np.array(emo_text)
    emo_cam = np.round(emo_cam, decimals=4)
    emo_text = np.round(emo_text, decimals=4)
    emotion = weight_cam*emo_cam + weight_text*emo_text
    emotion = np.round(emotion, decimals=4)
    max_index = np.argmax(emotion)
    label= id2label[max_index]
    print(label)
    return label