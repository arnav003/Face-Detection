import os
import time
import uuid
import cv2

# Define the directory path to save images
IMAGES_PATH = os.path.join('data', 'images')

# Define the number of images to capture
number_images = 30

# Open the webcam capture device
cap = cv2.VideoCapture(0)

# Iterate over the specified number of images
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))

    # Read a frame from the webcam
    ret, frame = cap.read()

    # Generate a unique image name using UUID and save it to the specified path
    imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)

    # Display the captured frame
    cv2.imshow('frame', frame)

    # Wait for a short duration to give time for the user to adjust if needed
    time.sleep(0.5)

    # Check if the user pressed the 'q' key to quit capturing images
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam capture device
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
