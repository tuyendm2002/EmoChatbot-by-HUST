import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer



import underthesea
def word_segmentation(text):
    output = underthesea.word_tokenize(text)
    return output

def text_preprocess(text):
    text = word_segmentation(text) # required for PhoBERT
    text = ' '.join(text) # return 1 string
    return text

config = {
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'MODEL_NAME': 'PhoBERT-Sentiment',
    'PRETRAINED_NAME': 'vinai/phobert-base-v2',
    'NUM_CLASSES': 7,
    'DROPOUT': 0.5,

    'TRAINING': {
        'N_EPOCHS': 20,
        'INIT_LR': 1e-5,
        'WEIGHT_DECAY': 1e-2,
        'BATCH_SIZE': 8,
        'WORKERS': 0,
        'METRIC_SAVE_BEST': 'f1_score',
        'PATIENCE': 3
    },

    'TOKENIZER': {
      'PADDING': 'max_length',
      'MAX_INPUT_LENGTH': 200,
      'TRUNCATION': True,
      'RETURN_ATTENTION_MASK': True,
      'ADD_SPECIAL_TOKENS': True,
    },
}

id2label = {
  0:'angry',
  1:'disgust',
  2:'happy',
  3:'fear',
  4:'neutral',
  5:'sad',
  6:'surprise'
}

class SentimentClassifier(nn.Module):
    def __init__(self, config):
        super(SentimentClassifier, self).__init__()
        self.basemodel = AutoModel.from_pretrained(config['PRETRAINED_NAME'])
        self.drop1 = nn.Dropout(p=config['DROPOUT'])
        self.fc1 = nn.Linear(self.basemodel.config.hidden_size, 256)
        self.drop2 = nn.Dropout(p=config['DROPOUT'])
        self.fc2 = nn.Linear(256, config['NUM_CLASSES'])

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.basemodel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        x = self.drop1(output)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x

def predict_sentiment(model, tokenizer, input_sentence):
    model.eval()
    # Tokenize và mã hóa câu nhập vào
    encoding = tokenizer.encode_plus(
        input_sentence,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=200,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities_list = torch.sigmoid(logits)[0].tolist()
        probabilities_tensor = torch.tensor(probabilities_list)
        max_probabilities, predicted_labels = torch.max(probabilities_tensor, dim=0)
        label_list = (probabilities_tensor / sum(probabilities_tensor)).tolist()

    return predicted_labels, label_list

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = SentimentClassifier(config)
model.load_state_dict(torch.load('model/sentiment-model.pth', map_location=torch.device('cpu')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
'''
text = "Mưa gì mưa lắm thế không biết"
text = text_preprocess(text)
predict, label_list = predict_sentiment(model, tokenizer, text)
emotion = id2label[predict.item()]
print("Cảm xúc của câu nói là: ", emotion)
print(label_list)
'''