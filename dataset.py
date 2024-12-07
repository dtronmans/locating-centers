import os

import cv2
from torchvision import transforms
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def generate_gaussian_heatmap(image_size, points, sigma=5):
    """
    Generate a heatmap with Gaussian peaks at given points.

    Args:
        image_size (tuple): Size of the output heatmap (height, width).
        points (Tensor): Tensor of shape (N, 2), where N is the number of points.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        heatmap (Tensor): Heatmap of shape (1, H, W).
    """
    height, width = image_size
    heatmap = np.zeros((height, width), dtype=np.float32)

    for point in points:
        x, y = int(point[0]), int(point[1])

        # Create a meshgrid centered around the point (x, y)
        y_grid, x_grid = np.meshgrid(np.arange(0, height), np.arange(0, width), indexing='ij')

        # Compute the Gaussian centered at (x, y)
        gaussian = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))

        # Add the Gaussian to the heatmap
        heatmap += gaussian

    # Normalize heatmap to be between 0 and 1
    heatmap = np.clip(heatmap, 0, 1)
    return torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)


class ImagePointDataset(Dataset):
    def __init__(self, images_folder, transform=None, sigma=5, show_points=False):
        """
        Args:
            images_folder (str): Path to the folder containing the images and corresponding .txt files.
            transform (callable, optional): Optional transform to be applied on an image.
            sigma (float): Standard deviation for the Gaussian peaks in the heatmap.
        """
        self.images_folder = images_folder
        self.show_points = show_points
        self.image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.sigma = sigma

        self.default_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.images_folder, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        height, width = image.size

        # Load corresponding annotation file
        annotation_file = img_name.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
        points = []
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as file:
                for line in file:
                    x, y = map(float, line.strip().split(','))
                    points.append([int(x * width), int(y * height)])

        points = torch.tensor(points)

        if self.show_points:
            # Convert PIL image to OpenCV format (BGR)
            image_cv = np.array(image)[:, :, ::-1].copy()

            # Draw points on the image
            for point in points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(image_cv, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red points

            # Display the image with points (temporary, for debugging)
            cv2.imshow('Image with Points', image_cv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Generate heatmap
        heatmap = generate_gaussian_heatmap(image.size[::-1], points, sigma=self.sigma)

        if self.transform:
            image = self.transform(image)

        image = self.default_transform(image)

        return image, heatmap

    @staticmethod
    def show_heatmap(image, heatmap):
        return


if __name__ == "__main__":
    dataset = ImagePointDataset("images", show_points=True)
    image, heatmap = dataset[0]
    image_np = (image.squeeze(0).numpy() * 255).astype(np.uint8)
    image_np = np.transpose(image_np, (1, 2, 0))
    heatmap_np = heatmap.squeeze(0).numpy()
    heatmap_np = (heatmap_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
    added_image = cv2.addWeighted(heatmap_color, 0.4, image_np, 0.4, 0)
    cv2.imshow("heatmap", added_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
