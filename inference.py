import torch
from PIL import Image
from torchvision import transforms
from pefomed.models.pefomed_model import PeFoMedModel

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

model = PeFoMedModel(pretrained_path="outputs/best_model_epoch_1.pth").to(device)
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_path = "my_dataset/test/images/patient123/study5/view1_frontal.jpg"
img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

with torch.no_grad():
    print("\nGenerated Report:\n")
    print(model.generate(img))
