import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class ViT(nn.Module):
    def __init__(self, config=ViTConfig(), num_labels=8,
                 model_checkpoint='google/vit-base-patch16-224-in21k'):
        super(ViT, self).__init__()
        self.vit = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x):
        x = self.vit(x)['last_hidden_state']
        # Use the embedding of [CLS] token
        output = self.classifier(x[:, 0, :])
        return output

def load_model(model_path='model/model_weights.pth', num_labels=8):
    model = ViT(num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def prepare_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image_path, device=torch.device("cpu")):
    model.to(device)
    image = prepare_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        output = F.softmax(output, dim=-1)  # Convert to probabilities
        print(output)
        predicted_label_idx = output.argmax(dim=1).item()
        max_probability = output.max(dim=1).values.item() * 100  # Get the highest probability
    return predicted_label_idx, max_probability

id2label = {0: 'Melanocytic nevus - Nốt ruồi lành tính',
            1: 'Melanoma - Ung thư hắc tố da',
            2: 'Benign keratosis - Dày sừng tiết bã',
            3: 'Dermatofibroma - U sợi bì',
            4: 'Basal cell carcinoma - Ung thư biểu mô tế bào đáy',
            5: 'Squamous cell carcinoma - Ung thư biểu mô tế bào vảy',
            6: 'Varcular lesion - U/Dị dạng mạch máu dưới da',
            7: 'Actinic keratosis - Dày sừng ánh sáng'}

def get_label(predicted_label_idx):
    return id2label.get(predicted_label_idx, "Unknown")

if __name__ == "__main__":
    # Example usage
    model_path = 'model/model_weights.pth'
    image_path = 'image/skin/captured_image.jpg'

    model = load_model(model_path=model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_label_idx, max_probability = predict(model, image_path, device)
    predicted_label = get_label(predicted_label_idx)

    print(f'Predicted label: {predicted_label}, with confidence: {max_probability:.2f}%')
