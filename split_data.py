import os
import random
import shutil

# Define the directory paths
IMAGES_PATH = os.path.join('data', 'images')
TRAIN_IMAGES_PATH = os.path.join('data', 'train', 'images')
TEST_IMAGES_PATH = os.path.join('data', 'test', 'images')
VAL_IMAGES_PATH = os.path.join('data', 'val', 'images')

# Create train, test, val directories if they don't exist
os.makedirs(TRAIN_IMAGES_PATH, exist_ok=True)
os.makedirs(TEST_IMAGES_PATH, exist_ok=True)
os.makedirs(VAL_IMAGES_PATH, exist_ok=True)

# Define the percentage split
train_percent = 0.7
test_percent = 0.15
val_percent = 0.15

# Get the list of image file paths
image_files = os.listdir(IMAGES_PATH)
random.shuffle(image_files)

# Calculate the number of images for each split
total_images = len(image_files)
train_count = int(total_images * train_percent)
test_count = int(total_images * test_percent)
val_count = total_images - train_count - test_count

# Split the image files into train, test, and val sets
train_files = image_files[:train_count]
test_files = image_files[train_count:train_count + test_count]
val_files = image_files[train_count + test_count:]


# Function to move files to destination directory
def move_files(files, destination):
    for file in files:
        src = os.path.join(IMAGES_PATH, file)
        dst = os.path.join(destination, file)
        try:
            shutil.move(src, dst)
            print(f"Moved image from {src} to {destination} successfully.")
        except:
            print(f"Some error occurred in moving image from {src} to {destination}.")


# Move files to respective directories
move_files(train_files, TRAIN_IMAGES_PATH)
move_files(test_files, TEST_IMAGES_PATH)
move_files(val_files, VAL_IMAGES_PATH)

LABELS_PATH = os.path.join('data', 'labels')
TRAIN_LABELS_PATH = os.path.join('data', 'train', 'labels')
TEST_LABELS_PATH = os.path.join('data', 'test', 'labels')
VAL_LABELS_PATH = os.path.join('data', 'val', 'labels')

os.makedirs(TRAIN_LABELS_PATH, exist_ok=True)
os.makedirs(TEST_LABELS_PATH, exist_ok=True)
os.makedirs(VAL_LABELS_PATH, exist_ok=True)

# Move labels to respective directories
for folder in ['train', 'test', 'val']:
    for file in os.listdir(os.path.join('data', folder, 'images')):
        filename = file.split('.')[0] + '.json'
        existing_filepath = os.path.join('data', 'labels', filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join('data', folder, 'labels', filename)
            try:
                os.replace(existing_filepath, new_filepath)
                print(f"Moved label file (for image at {file}) from {existing_filepath} to {new_filepath} successfully.")
            except:
                print(f"Some error occurred in moving label file (for image at {file}) from {existing_filepath} to {new_filepath}.")
