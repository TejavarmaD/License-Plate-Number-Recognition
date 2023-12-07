# Comparative Analysis of YOLO and Deep Learning Models for License Plate Recognition

This repository contains the implementation of two different approaches for license plate recognition: one using the YOLO (You Only Look Once) model and another using a traditional deep learning model. The goal is to compare these methods in terms of accuracy, speed, and practical applicability.

## Prerequisites

Before running the application, ensure you have the following installed:

- **Python 3.x**: The programming language used to build the application. [Download Python](https://www.python.org/downloads/)
- **PyTorch**: An open-source machine learning library used for applications such as computer vision and natural language processing. [Install PyTorch](https://pytorch.org/get-started/locally/)
- **OpenCV (Open Source Computer Vision Library)**: A library of programming functions mainly aimed at real-time computer vision. [Install OpenCV](https://pypi.org/project/opencv-python/)
- **YOLOv5**: The latest version of YOLO, a family of object detection architectures and models pretrained on the COCO dataset. [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- **LabelImg**: A graphical image annotation tool necessary for labeling objects for training. [LabelImg GitHub](https://github.com/tzutalin/labelImg)
- **Flask**: A micro web framework written in Python, used for running the web application. [Install Flask](https://flask.palletsprojects.com/en/2.0.x/installation/)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-github-username/license-plate-recognition.git
cd license-plate-recognition
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

Make sure you ran the notebooks you will be getting hte weight which you have to put it under static/models:

Running the Application:
 Once all the prerequisites are installed,you can run the flask app:
```python app.py```

After running the command, navigate to http://127.0.0.1:5000/ in your web browser to access the application.

## Usage

Upload an image through the web interface.
Choose between the YOLO model and the deep learning model for license plate detection.
View the results, including the detected license plate and its text.


