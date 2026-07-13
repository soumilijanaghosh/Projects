Tea Looper Detection & Classification

An object detection and classification pipeline to identify tea loopers (insect pests) in tea plantation images, built with YOLOv8.

Overview

This project uses YOLOv8 for two tasks:


Detection: Locating tea loopers in field images using a YOLOv8x detection model.
Classification: Classifying insect crops into species/categories using a YOLOv8x-cls model.


Tech Stack


Python, OpenCV, NumPy
Ultralytics YOLOv8 (detection + classification)
PyTorch, Torchvision
Google Colab (T4 GPU)


Workflow


Setup — Install dependencies (ultralytics, opencv-python-headless), mount Google Drive for dataset access.
Detection

Load pretrained YOLOv8x model.
Train on custom-labeled tea looper dataset (data.yaml).
Validate and run inference on test images.



Classification

Shuffle and split dataset into train/val/test (70/20/10).
Train YOLOv8x-cls classifier on cropped insect images.
Validate and run inference on new images.



Custom Inference Function — predict_image() loads a single image, preprocesses it, and outputs the predicted insect class with confidence score, along with a visualization.


How to Run


Open the notebook in Google Colab.
Mount your Google Drive containing the dataset.
Update dataset paths (DATASET_SRC, data.yaml path) to match your Drive structure.
Run cells sequentially to train detection/classification models and perform inference.


Notes


GPU (T4) recommended for training.
Dataset paths are configured for the author's personal Google Drive — update paths before running on your own data.
