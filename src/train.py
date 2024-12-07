import torch
from torch.utils.data import DataLoader, random_split

from src.losses import mse_loss
from src.unet import UNet
from src.dataset import ImagePointDataset
from src.thresholding import basic_threshold

from src.config import Config

if __name__ == "__main__":
    config = Config("config.json")

    dataset = ImagePointDataset("images")
    train_size = int((1 - config.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)

    model = UNet(n_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(config.num_epochs):
        model.train()
        running_train_loss = 0.0

        for i, (images, points) in enumerate(train_loader):
            images = images.to(device)
            heatmaps = points.to(device)

            predicted_heatmaps = model(images)
            predicted_points = basic_threshold(predicted_heatmaps)

            loss = mse_loss(predicted_heatmaps, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for images, points in val_loader:
                images = images.to(device)
                heatmaps = points.to(device)

                predicted_heatmaps = model(images)
                predicted_points = basic_threshold(predicted_heatmaps)

                loss = mse_loss(predicted_heatmaps, heatmaps)

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

    print("Training complete.")

    torch.save(model.state_dict(), "../model.pt")
