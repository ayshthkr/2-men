import os
import shutil
import sys
from glob import glob

from colorama import Fore, Style
from tqdm import tqdm
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import splitfolders

from utils import to_categorical

# Helper functions for file processing and preprocessing
def get_file_lists(data_dir: str | os.PathLike[str] = "data/training_data/"):
    """ Get sorted lists of file paths for different modalities and masks. """
    t1c_list = sorted(glob(f"{data_dir}/**/*t1c.nii.gz"))
    t1n_list = sorted(glob(f"{data_dir}/**/*t1n.nii.gz"))
    t2w_list = sorted(glob(f"{data_dir}/**/*t2w.nii.gz"))
    t2f_list = sorted(glob(f"{data_dir}/**/*t2f.nii.gz"))
    mask_list = sorted(glob(f"{data_dir}/**/*seg.nii.gz"))

    assert len(t1c_list) == len(t1n_list) == len(t2w_list) == len(t2f_list) == len(mask_list), (
        f"{Fore.RED}❌ Length mismatch{Style.RESET_ALL}: All modality lists and mask lists should have the same number of items."
    )

    print(f"✔ Found {Fore.GREEN}{len(t1c_list)}{Style.RESET_ALL} samples.")
    return t1c_list, t1n_list, t2w_list, t2f_list, mask_list

def load_and_preprocess_image(image_path: str | os.PathLike[str]):
    """ Load a NIfTI image and apply MinMax scaling. """
    image = nib.load(image_path).get_fdata()
    scaler = MinMaxScaler()
    return scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

def load_and_process_mask(mask_path: str | os.PathLike[str]):
    """ Load a NIfTI mask, convert to uint8, and adjust class labels(not for now). """
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    # mask[mask == 4] = 3  # Convert class 4 to class 3
    return mask

def process_and_save_images(t1c_list, t1n_list, t2w_list, t2f_list, mask_list,
                            output_dir: str | os.PathLike[str] = "data/input_data_total",
                            allImages: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    for img_idx in tqdm(range(len(t2w_list) if allImages else 50), desc="Processing images", colour="green", leave=False):
        print(f"⚙️ Now processing {Fore.BLUE}image and mask number: {img_idx}{Style.RESET_ALL}")

        # Load and preprocess modalities
        t1c_img = load_and_preprocess_image(t1c_list[img_idx])
        t1n_img = load_and_preprocess_image(t1n_list[img_idx])
        t2w_img = load_and_preprocess_image(t2w_list[img_idx])
        t2f_img = load_and_preprocess_image(t2f_list[img_idx])

        # Load and process mask
        mask_img = load_and_process_mask(mask_list[img_idx])

        # Crop region of interest
        combined_img = np.stack([t1c_img, t1n_img, t2w_img, t2f_img], axis=3)

        print("SHAPE: ", combined_img.shape)

        val, counts = np.unique(mask_img, return_counts=True)
        # print("VAL: ", val)
        # print("COUNTS: ", counts)
        tumor_ratio = 1 - (counts[0] / counts.sum())
        # print("tumor ratio: ", tumor_ratio)

        print(f"{Fore.GREEN}✔ Saving{Style.RESET_ALL}")
        mask_img = to_categorical(mask_img, num_classes=4)
        np.save(f"{output_dir}/images/image_{img_idx}.npy", combined_img)
        np.save(f"{output_dir}/masks/mask_{img_idx}.npy", mask_img)

        # sys.exit(0)

        # Calculate class distribution in mask
        # val, counts = np.unique(mask_img, return_counts=True)
        # tumor_ratio = 1 - (counts[0] / counts.sum())
        #
        # if tumor_ratio > 0.01:  # At least 1% useful volume with labels that are not 0
        #     print(f"{Fore.GREEN}✔ Saving{Style.RESET_ALL}: Significant tumor region found.")
        #     mask_img = to_categorical(mask_img, num_classes=4)
        #     np.save(f"{output_dir}/images/image_{img_idx}.npy", combined_img)
        #     np.save(f"{output_dir}/masks/mask_{img_idx}.npy", mask_img)
        # else:
        #     print(f"{Fore.RED}❌ Skipping{Style.RESET_ALL}: Insufficient tumor presence.")

def main():
    if not os.path.exists("data/training_data"):
        try:
            shutil.copytree("data/BraTS-MEN-Train", "data/training_data/")
            print(f"{Fore.CYAN}Dataset copied to data/training_data{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Warning:{Style.RESET_ALL} 'data/training_data' directory already exists! Delete it if you want to copy the dataset again.")

    t1c_list, t1n_list, t2w_list, t2f_list, mask_list = get_file_lists()
    process_and_save_images(t1c_list, t1n_list, t2w_list, t2f_list, mask_list)

    # Split data with a ratio into train and test.
    input_path = "data/input_data_total"
    output_path = "data/input_data_split"

    splitfolders.ratio(input_path, output_path, seed=42, ratio=(0.75, 0.25), group_prefix=None)

    # Counting the number of images in the train and val directories
    print(
        f"There are {Fore.BLUE}{len(os.listdir(os.path.join(output_path, 'train/images')))}{Style.RESET_ALL} images in train.")
    print(
        f"There are {Fore.BLUE}{len(os.listdir(os.path.join(output_path, 'val/images')))}{Style.RESET_ALL} images in val.")
    print(
        f"Total images with usable tumor information: {Fore.CYAN}{len(os.listdir(os.path.join(output_path, 'train/images'))) + len(os.listdir(os.path.join(output_path, 'val/images')))}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
