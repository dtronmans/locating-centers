import cv2
import matplotlib.pyplot as plt
import os

from src.config import Config


class Annotator:

    def __init__(self, images_path):
        self.images_path = images_path
        self.points = []

    def annotate(self):
        for image_name in os.listdir(self.images_path):
            if image_name.endswith(('.jpg', '.png')):
                full_image_path = os.path.join(self.images_path, image_name)
                self.annotate_image(full_image_path)
                self.points = []

    def onclick(self, event):
        if event.xdata is not None and event.ydata is not None:
            normalized_x = event.xdata / self.image_width
            normalized_y = event.ydata / self.image_height
            self.points.append((normalized_x, normalized_y))
            plt.plot(event.xdata, event.ydata, 'ro')
            plt.draw()

    def annotate_image(self, image_file):
        image = cv2.imread(image_file)
        self.image_height, self.image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(f"Annotating {image_file}. Click to add points")
        cid = plt.gcf().canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

        self.save_annotations(image_file)

    def save_annotations(self, image_file):
        output_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        with open(output_file, 'w') as f:
            for point in self.points:
                f.write(f"{point[0]},{point[1]}\n")
        print(f"Annotations saved to {output_file}")


if __name__ == "__main__":
    config = Config("config.json")
    annotator = Annotator(images_path=config.images_path)
    annotator.annotate()
