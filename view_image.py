import os
import cv2
import matplotlib.pyplot as plt

# Check what's in the current folder
print("Current working directory:", os.getcwd())
print("Files and folders here:", os.listdir('.'))

data_dir = '.'
categories = ['cat', 'dog']

for category in categories:
    path = os.path.join(data_dir, category)
    for img_name in os.listdir(path)[:2]:  # show 2 images per category
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(category)
        plt.axis('off')
        plt.show()
