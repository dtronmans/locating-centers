import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from src.unet import UNet


def predict(model_path, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3)

    # Load the pre-trained model state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    output_to_visualize = output.squeeze(0).cpu().numpy()

    plt.imshow(output_to_visualize[0], cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Predicted Heatmap')
    plt.show()


if __name__ == "__main__":
    predict("../model.pt", "images/cat_998.png")
    predict("../model.pt", "images/cat_999.png")
