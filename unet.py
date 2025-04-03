import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Define constants
SEED = 909
BATCH_SIZE_TRAIN = 8  # Reduced batch size for 3D data
BATCH_SIZE_TEST = 8

# Input image dimensions
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
DEPTH = 155
CHANNELS = 4  # Number of input channels

# Data directories
data_dir = 'data/input_data_split/'
train_image_dir = os.path.join(data_dir, 'train/images')
train_mask_dir = os.path.join(data_dir, 'train/masks')
val_image_dir = os.path.join(data_dir, 'val/images')
val_mask_dir = os.path.join(data_dir, 'val/masks')

NUM_OF_EPOCHS = 100

# Set seeds for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)

# Custom Dataset for 3D Medical Segmentation
class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Get all image file names
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.npy')]
        self.image_files.sort()  # Ensure consistent ordering

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get file name
        image_name = self.image_files[idx]
        mask_name = image_name.replace('image', 'mask')

        # Load numpy files
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.load(image_path)  # Shape: (H, W, D, 4)
        mask = np.load(mask_path)  # Shape: (H, W, D, 4)

        # For this example, we'll work with 2D slices from the middle of the volume
        # In a full implementation, you might want to process the full 3D volume or use a sliding window approach
        mid_slice = DEPTH // 2

        # Extract middle slice and transpose to (C, H, W) format for PyTorch
        image_slice = image[:, :, mid_slice, :].transpose(2, 0, 1)  # Shape: (4, H, W)
        mask_slice = mask[:, :, mid_slice, :].transpose(2, 0, 1)  # Shape: (4, H, W)

        # Convert to float32 and normalize
        image_slice = image_slice.astype(np.float32)
        mask_slice = mask_slice.astype(np.float32)

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image_slice)
        mask_tensor = torch.from_numpy(mask_slice)

        return image_tensor, mask_tensor

    def get_full_volume(self, idx):
        """Function to get the full 3D volume for visualization or inference"""
        image_name = self.image_files[idx]
        mask_name = image_name.replace('image', 'mask')

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.load(image_path)  # Shape: (H, W, D, 4)
        mask = np.load(mask_path)  # Shape: (H, W, D, 4)

        return image, mask


# Create data loaders
def create_dataloaders(image_dir, mask_dir, batch_size, shuffle=False):
    dataset = MedicalSegmentationDataset(image_dir, mask_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    return dataloader, dataset


# Create dataloaders for training and validation
train_loader, train_dataset = create_dataloaders(train_image_dir, train_mask_dir, BATCH_SIZE_TRAIN, shuffle=True)
val_loader, val_dataset = create_dataloaders(val_image_dir, val_mask_dir, BATCH_SIZE_TEST)


# Display function for 2D slices
def display_slice(display_list, titles=None):
    if titles is None:
        titles = ['Input Image', 'True Mask', 'Predicted Mask']

    num_images = len(display_list)
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        if i < len(titles):
            plt.title(titles[i])

        # Display appropriate channel (for visualization, we'll use the first channel)
        if display_list[i].shape[0] == 4:  # If it has 4 channels
            plt.imshow(display_list[i][0], cmap='gray')
        else:
            plt.imshow(display_list[i], cmap='gray')

        plt.axis('off')
    plt.show()


# Show dataset samples
def show_dataset_samples(dataset, num=1):
    for i in range(min(num, len(dataset))):
        image, mask = dataset[i]
        display_slice([image.numpy(), mask.numpy()],
                      ['Input Channel 0', 'Mask Channel 0'])


# Display some training examples
# print("Displaying training examples:")
# show_dataset_samples(train_dataset, 5)


# Define U-Net model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_levels=4, initial_features=32, in_channels=4, out_channels=4):
        super(UNet, self).__init__()

        # Store the downsampling layers
        self.downs = nn.ModuleList()
        # Store the upsampling layers
        self.ups = nn.ModuleList()
        # Max pooling
        self.pool = nn.MaxPool2d(2)

        # Downstream path
        for level in range(n_levels):
            in_features = in_channels if level == 0 else initial_features * 2 ** (level - 1)
            out_features = initial_features * 2 ** level

            down_block = DoubleConv(in_features, out_features)
            self.downs.append(down_block)

        # Upstream path
        for level in range(n_levels - 1, 0, -1):
            in_features = initial_features * 2 ** level
            out_features = initial_features * 2 ** (level - 1)

            # Transposed convolution for upsampling
            self.ups.append(
                nn.ConvTranspose2d(
                    in_features, out_features,
                    kernel_size=2,
                    stride=2
                )
            )

            # Convolutions after concatenation
            self.ups.append(DoubleConv(in_features, out_features))

        # Final one-by-one convolution
        self.final_conv = nn.Conv2d(initial_features, out_channels, kernel_size=1)

        # For multi-class segmentation, we use Softmax; for binary, we'd use Sigmoid
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Store skip connections
        skip_connections = []

        # Downstream
        for i, down in enumerate(self.downs[:-1]):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottom layer
        x = self.downs[-1](x)

        # Upstream
        skip_connections = skip_connections[::-1]  # Reverse to use from bottleneck up

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip_connection = skip_connections[idx // 2]

            # Handle if shapes don't match perfectly (can happen due to different paddings)
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:],
                    mode="bilinear", align_corners=False
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)  # Double convolution

        # Final convolution
        x = self.final_conv(x)
        x = self.softmax(x)  # Apply softmax for multi-class segmentation

        return x


# Initialize the model
model = UNet(n_levels=4, initial_features=32, in_channels=CHANNELS, out_channels=CHANNELS).to(device)
print(model)

# Define loss function and optimizer
# For multi-class segmentation with one-hot encoded masks, we use CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Dice coefficient as a metric
def dice_coefficient(pred, target):
    smooth = 1e-5

    # Flatten the predictions and targets
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()

    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    dice_scores = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Print progress
            if (i + 1) % 5 == 0:
                print(f"  Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        dice_score = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()

                # Calculate Dice coefficient for each class and average
                pred = outputs.argmax(dim=1)
                target = masks.argmax(dim=1)

                batch_dice = 0
                for c in range(CHANNELS):
                    pred_c = (pred == c).float()
                    target_c = (target == c).float()
                    batch_dice += dice_coefficient(pred_c, target_c)

                dice_score += batch_dice / CHANNELS

        avg_val_loss = val_loss / len(val_loader)
        avg_dice_score = dice_score / len(val_loader)

        val_losses.append(avg_val_loss)
        dice_scores.append(avg_dice_score.item())

        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Dice Score: {avg_dice_score:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print("  Model saved!")

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(dice_scores, label='Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model


# Main training loop (uncomment to train)
# model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_OF_EPOCHS)

# Load the trained model
model.load_state_dict(torch.load('best_unet_model.pth'))
model.eval()


def visualize_predictions(model, dataset, num_samples=5, device=device):
    """
    Visualize model predictions compared to ground truth
    """
    model.eval()
    fig = plt.figure(figsize=(15, num_samples * 4))

    for i in range(num_samples):
        # Get a random sample
        idx = np.random.randint(0, len(dataset))
        image, mask = dataset[i]

        # Add batch dimension and send to device
        image = image.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            pred = model(image)
            pred = pred.squeeze(0).cpu().numpy()

        # Convert to class indices
        pred_class = np.argmax(pred, axis=0)
        mask_class = np.argmax(mask.numpy(), axis=0)

        # Plot original image (first channel)
        ax = fig.add_subplot(num_samples, 3, i * 3 + 1)
        ax.imshow(image.squeeze(0).cpu().numpy()[0], cmap='gray')
        ax.set_title(f'Input Image (Sample {idx})')
        ax.axis('off')

        # Plot ground truth mask
        ax = fig.add_subplot(num_samples, 3, i * 3 + 2)
        ax.imshow(mask_class, cmap='nipy_spectral')
        ax.set_title('Ground Truth')
        ax.axis('off')

        # Plot predicted mask
        ax = fig.add_subplot(num_samples, 3, i * 3 + 3)
        ax.imshow(pred_class, cmap='nipy_spectral')
        ax.set_title('Prediction')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_model(model, dataloader, device=device):
    """
    Evaluate model performance on the provided dataloader
    Returns metrics for each class
    """
    model.eval()

    # Initialize metrics
    dice_scores = np.zeros(CHANNELS)
    all_preds = []
    all_targets = []

    # Process all batches
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)

            # Get predicted class indices
            preds = outputs.argmax(dim=1).cpu().numpy()  # (B, H, W)
            targets = masks.argmax(dim=1).cpu().numpy()  # (B, H, W)

            # Add to collections for overall metrics
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())

            # Calculate Dice coefficient for each class
            for c in range(CHANNELS):
                pred_c = (preds == c).astype(float)
                target_c = (targets == c).astype(float)

                # Calculate Dice for this class
                intersection = (pred_c * target_c).sum()
                smooth = 1e-5
                dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
                dice_scores[c] += dice

    # Average Dice scores across all batches
    dice_scores /= len(dataloader)

    # Calculate additional metrics using sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=range(CHANNELS), average=None
    )

    # Convert to pandas DataFrame for better display
    metrics_df = pd.DataFrame({
        'Class': [f'Class {i}' for i in range(CHANNELS)],
        'Dice Score': dice_scores,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(CHANNELS))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return metrics_df, cm, cm_normalized


def plot_metrics(metrics_df, cm_normalized):
    """
    Plot metrics visualization
    """
    # Bar chart for metrics by class
    plt.figure(figsize=(12, 6))
    metrics_df.set_index('Class').plot(kind='bar', figsize=(12, 6))
    plt.title('Segmentation Metrics by Class')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Heatmap for confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(CHANNELS)],
                yticklabels=[f'Class {i}' for i in range(CHANNELS)])
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def visualize_class_activation(model, dataset, num_samples=3, device=device):
    """
    Visualize activation maps for each class
    """
    model.eval()

    for i in range(num_samples):
        # Get sample
        image, mask = dataset[i]
        image_tensor = image.unsqueeze(0).to(device)

        # Get model output
        with torch.no_grad():
            output = model(image_tensor)
            output = output.squeeze(0).cpu().numpy()  # (C, H, W)

        # Get ground truth mask
        mask = mask.numpy()  # (C, H, W)

        # Plot
        fig, axes = plt.subplots(2, CHANNELS + 1, figsize=(15, 6))

        # Display input image in first column
        axes[0, 0].imshow(image.numpy()[0], cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        axes[1, 0].imshow(np.argmax(mask, axis=0), cmap='nipy_spectral')
        axes[1, 0].set_title('Ground Truth')
        axes[1, 0].axis('off')

        # Display activation maps for each class
        for c in range(CHANNELS):
            # Model prediction probability for class c
            axes[0, c + 1].imshow(output[c], cmap='hot')
            axes[0, c + 1].set_title(f'Class {c} Activation')
            axes[0, c + 1].axis('off')

            # Ground truth for class c
            axes[1, c + 1].imshow(mask[c], cmap='gray')
            axes[1, c + 1].set_title(f'Class {c} GT')
            axes[1, c + 1].axis('off')

        plt.tight_layout()
        plt.show()


def plot_volume_slices(dataset, idx=0, slices=None):
    """
    Plot multiple slices from a 3D volume
    """
    # Get full volume
    image_vol, mask_vol = dataset.get_full_volume(idx)

    # Default slices if not specified
    if slices is None:
        # Get 5 evenly spaced slices
        slices = np.linspace(0, DEPTH - 1, 5, dtype=int)

    # Plot slices
    fig, axes = plt.subplots(2, len(slices), figsize=(15, 6))

    for i, slice_idx in enumerate(slices):
        # Plot image slice (first channel)
        axes[0, i].imshow(image_vol[:, :, slice_idx, 0], cmap='gray')
        axes[0, i].set_title(f'Slice {slice_idx}')
        axes[0, i].axis('off')

        # Plot mask slice (using argmax across channels)
        mask_slice = np.argmax(mask_vol[:, :, slice_idx, :], axis=2)
        axes[1, i].imshow(mask_slice, cmap='nipy_spectral')
        axes[1, i].set_title(f'Mask {slice_idx}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def run_3d_evaluation(model, dataset, device=device):
    """
    Evaluate model on full 3D volumes by processing slice-by-slice
    """
    model.eval()
    vol_dice_scores = []

    for idx in tqdm(range(min(5, len(dataset))), desc="Evaluating 3D volumes"):
        # Get full volume
        image_vol, mask_vol = dataset.get_full_volume(idx)

        # Initialize prediction volume
        pred_vol = np.zeros_like(mask_vol)

        # Process each slice
        for z in range(DEPTH):
            # Extract slice and convert to PyTorch tensor
            image_slice = image_vol[:, :, z, :].transpose(2, 0, 1)  # (4, H, W)
            image_slice = torch.from_numpy(image_slice).float().unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                pred_slice = model(image_slice)
                pred_slice = pred_slice.squeeze(0).cpu().numpy()  # (4, H, W)

            # Store prediction in volume
            pred_vol[:, :, z, :] = pred_slice.transpose(1, 2, 0)

        # Calculate 3D Dice score for each class
        dice_3d = np.zeros(CHANNELS)
        for c in range(CHANNELS):
            pred_c = pred_vol[:, :, :, c] > 0.5
            target_c = mask_vol[:, :, :, c] > 0.5

            intersection = np.logical_and(pred_c, target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union > 0:
                dice_3d[c] = 2. * intersection / union

        vol_dice_scores.append(dice_3d)

    # Average Dice scores across volumes
    avg_vol_dice = np.mean(vol_dice_scores, axis=0)

    # Plot 3D Dice scores
    plt.figure(figsize=(10, 6))
    plt.bar(range(CHANNELS), avg_vol_dice)
    plt.xlabel('Class')
    plt.ylabel('3D Dice Score')
    plt.title('Average 3D Dice Score by Class')
    plt.xticks(range(CHANNELS), [f'Class {i}' for i in range(CHANNELS)])
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return avg_vol_dice


# Run the evaluation
print("Visualizing model predictions...")
visualize_predictions(model, val_dataset, num_samples=3)

print("\nCalculating metrics on validation set...")
metrics_df, cm, cm_normalized = evaluate_model(model, val_loader)
print("\nMetrics by class:")
print(metrics_df)

print("\nPlotting metrics...")
plot_metrics(metrics_df, cm_normalized)

print("\nVisualizing class activation maps...")
visualize_class_activation(model, val_dataset, num_samples=2)

print("\nVisualizing 3D volume slices...")
plot_volume_slices(val_dataset, idx=0)

print("\nRunning 3D volume evaluation...")
avg_vol_dice = run_3d_evaluation(model, val_dataset)
print("\nAverage 3D Dice scores by class:")
for i, score in enumerate(avg_vol_dice):
    print(f"Class {i}: {score:.4f}")
