import os
import shutil
from sklearn.model_selection import train_test_split

def combine_image_folders(source_folders, destination_folder):
    """
    Combines images from multiple folders into a single destination folder.

    Parameters:
    - source_folders: List of paths to the source folders containing images.
    - destination_folder: Path to the destination folder where all images will be copied.
    """
    if os.path.exists(destination_folder):
        for filename in os.listdir(destination_folder):
            file_path = os.path.join(destination_folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"Content of {destination_folder} deleted.")
    else:
        os.makedirs(destination_folder)
        print(f"Folder created: {destination_folder}")

    for folder in source_folders:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.jpg')):
                    source_path = os.path.join(folder, filename)
                    destination_path = os.path.join(destination_folder, f"{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}")
                    shutil.copy2(source_path, destination_path)
                    print(f"Image copied: {destination_path}")

        else:
            print(f"The folder {folder} does not exist.")

def train_test(source_folder, train_folder, test_folder, test_size=0.3, random_state=42):
    """
    Split files from the source folder into training and testing folders.
    
    Args:
        source_folder (str): Path to the folder with the original files.
        train_folder (str): Path to the folder where training files will be copied.
        test_folder (str): Path to the folder where test files will be copied.
        test_size (float): Proportion of the dataset to be used for testing (default is 0.3).
        random_state (int): Seed for random number generator to ensure reproducibility.
    """
    # Create train and test directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Split the files into train and test sets
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)

    # Function to move files
    def move_files(file_list, destination_folder):
        for file_name in file_list:
            source_path = os.path.join(source_folder, file_name)
            dest_path = os.path.join(destination_folder, file_name)
            shutil.copy2(source_path, dest_path)  # Use shutil.copy2 to preserve file metadata
            print(f'Copied {file_name} to {destination_folder}')

    # Move the files
    move_files(train_files, train_folder)
    move_files(test_files, test_folder)

    print(f"Data split complete: {len(train_files)} training files, {len(test_files)} testing files.")