# 🏙️ Smart City Object Detection
Garbage & Pothole Detection using YOLOv8 and Gradio

An AI-powered computer vision application that detects garbage dumps and road potholes from images to support smart city infrastructure monitoring.

The system uses a custom-trained YOLOv8 model and provides an interactive Gradio web interface for real-time inference and summarized insights.

# 📌 Project Status

🚧 In Progress — Production-Ready MVP

End-to-end ML pipeline completed (training → inference → UI)

Gradio-based inference application fully functional

Current model performance: ~60% mAP

Actively improving accuracy through data-centric optimizations

This project is intentionally published as an MVP to demonstrate real-world ML system design and iterative improvement.

#  Project Highlights

Custom-trained YOLOv8m object detection model

Dataset annotated & augmented using Roboflow

Model training performed on Google Colab (GPU)

Interactive Gradio UI for image-based detection

Adjustable Confidence and IoU thresholds

Automatic detection summary (count-based)

Clean, industry-aligned project structure

# 🧠 Problem Statement

Urban infrastructure issues such as potholes and garbage accumulation negatively affect safety, cleanliness, and quality of life.
Manual reporting systems are often slow, inconsistent, and reactive.

This project automates the detection of these issues using deep learning-based object detection, enabling:

Faster identification of infrastructure issues

Data-driven decision making

Scalable smart city monitoring solutions

# 🛠️ Tech Stack
Category	Technology
Language	Python 3.9+
Model	YOLOv8 (Ultralytics)
Dataset	Roboflow
UI	Gradio
Image Processing	Pillow
Training	Google Colab (GPU)

# 📂 Project Structure
Smart-City-Object-Detection/
│
├── .gradio/                    # Gradio runtime cache (ignored in Git)
├── .venv/                      # Virtual environment (local, ignored)
│
├── data/                       # Dataset (optional / reference)
│
├── app.py                      # Gradio application (deployment)
├── yolov8m.pt                  # Custom-trained YOLOv8 model
│
├── model-training.ipynb        # Model training notebook (Colab)
├── notebook.ipynb              # Evaluation & inference testing
│
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation

# 📊 Dataset & Annotation

Platform: Roboflow

Classes:

pothole

garbage

Annotation & preprocessing techniques:

Bounding box annotation

Data augmentation (flip, rotate, brightness, blur)

Train / validation / test split

Dataset size: ~2,000+ images

# 🧪 Model Training

Architecture: YOLOv8m

Framework: Ultralytics YOLO

Environment: Google Colab (GPU)

Training Notebook: model-training.ipynb

# ⚠️ Note:
Training notebooks are designed for GPU-based execution and are not intended for direct local execution without GPU support.

# 📊 Model Evaluation (Current)

Precision: ~0.62

Recall: ~0.58

mAP@0.5: ~0.60

Metrics are expected to improve with further dataset expansion, augmentation, and hyperparameter tuning.

# 🖥️ Application Workflow

User uploads an image via the Gradio UI

YOLOv8 model performs object detection

Bounding boxes are drawn on the image

Detected objects are counted per class

A natural-language summary is generated

# 🎯 Gradio Interface Features

Image upload (JPG / PNG)

Confidence threshold slider

IoU threshold slider

Detection visualization

Automatic textual summary

# ⚙️ Installation & Usage
Install Dependencies
pip install -r requirements.txt

Run the Application
python app.py after the all notebook execute

Gradio will launch a local web application and provide a shareable link.

# 📈 Output Example

Detected: 2 potholes and 1 garbage dump

Visual bounding boxes on the image

Text summary:

“Total of 3 issues detected: 2 potholes and 1 garbage detected.”

# 🧩 Challenges & Learnings

Challenges faced:

High variability in pothole shapes and lighting conditions

Class imbalance between garbage and pothole samples

Limited annotated data affecting generalization

CPU-based inference latency during testing

Key learnings:

Importance of data-centric AI development

Evaluating models beyond raw accuracy

Practical deployment tradeoffs in real-world ML systems

# 📈 Accuracy Improvement Plan

Expand dataset to 5,000+ images

Add hard-negative samples

Perform YOLOv8 hyperparameter tuning

Train for additional epochs with early stopping

Experiment with YOLOv8l architecture

# 🔐 Limitations

Image-based inference only

No real-time video or CCTV stream support yet

Performance depends on image quality

CPU inference is slower than GPU

# 🚧 Future Enhancements

Real-time video & CCTV stream detection

GPS-based issue mapping

Database integration for issue tracking

Web dashboard for municipal authorities

Mobile application integration
