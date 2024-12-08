import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.config import Config
from src.thresholding import basic_threshold
from src.unet import UNet


def predict(model_path, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_points = basic_threshold(output)

    output_to_visualize = output.squeeze(0).cpu().numpy()

    plt.imshow(output_to_visualize[0], cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Predicted Heatmap')

    image_np = np.array(image)
    for point in predicted_points[0]:
        y, x = point.numpy()
        image_np = cv2.circle(image_np, (x, y), radius=1, color=(255, 0, 0), thickness=-1)

    plt.imshow(image_np)
    plt.title('Image with Predicted Points')
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    config = Config("config.json")

    predict("model.pt", config.predict_image)
    # predict("../model.pt", "images/cat_999.png")
