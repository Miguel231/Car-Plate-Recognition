import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split

def combine_image_folders(source_folders, destination_folder):
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

        else:
            print(f"The folder {folder} does not exist.")

def train_test(source_folder, train_folder, test_folder, val_folder, random_state=42):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    n_files = len(files)
    np.random.seed(random_state)
    indices = np.random.permutation(n_files)
    train_size = int(0.6 * n_files)
    test_size = int(0.3 * n_files)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    val_indices = indices[train_size + test_size:]
    def move_files(indices, destination_folder):
        for idx in indices:
            file_name = files[idx]
            source_path = os.path.join(source_folder, file_name)
            dest_path = os.path.join(destination_folder, file_name)
            shutil.copy2(source_path, dest_path)  # Use shutil.copy2 to preserve file metadata

    # Move the files to their respective folders
    move_files(train_indices, train_folder)
    move_files(test_indices, test_folder)
    move_files(val_indices, val_folder)

    print(f"Data split complete: {len(train_indices)} training files, {len(test_indices)} testing files, {len(val_indices)} validation files.")

def erase_double_images(folder_path):
    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file has '(1)' in its name
        if '(1)' in filename:
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f'Deleted: {file_path}')
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')
        elif '(2)' in filename:
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f'Deleted: {file_path}')
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')