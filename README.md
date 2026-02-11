🏙️ Smart City Issue Detection System

Garbage & Pothole Detection using YOLOv8 + Flask

An AI-powered computer vision web application that detects garbage dumps and road potholes from images to support smart city infrastructure monitoring.

The system uses a custom-trained YOLOv8 model and provides a Flask-based web dashboard for image upload, live camera capture, issue severity analysis, report history, and analytics.

🚧 In Progress — Production-Ready MVP 
📌 Project Status

🚀 Production-Ready MVP

✔ End-to-end ML pipeline completed (training → inference → web UI)
✔ Flask-based web application with database integration
✔ Report history, analytics dashboard, and severity scoring implemented
✔ Git-clean project structure following industry standards

Current model performance: ~60% mAP
Ongoing improvements through data-centric optimizations

This project is published as a real-world MVP demonstrating ML + Backend + UI integration.

✨ Key Features
🔍 Detection

Custom-trained YOLOv8m object detection model

Detects:

🕳️ Potholes

🗑️ Garbage

Bounding box visualization

Confidence-based filtering

🖥️ Web Application (Flask)

Image upload for issue reporting

Live camera capture support

Automatic detection summary

Severity classification (Low / Medium / High)

Responsive UI using Bootstrap 5

📊 Analytics & Reports

Automatic report saving

SQLite database integration

Report history page

Overall analytics dashboard:

Total reports

Total potholes detected

Total garbage detected

No-issue reports

Pie chart visualization using Chart.js

Delete individual reports or all reports

🧠 Problem Statement

Urban infrastructure issues such as potholes and garbage accumulation negatively impact safety, cleanliness, and quality of life.

Traditional manual reporting systems are:

Slow

Inconsistent

Reactive

This project automates detection using deep learning–based object detection, enabling:

Faster identification of infrastructure issues

Centralized issue tracking

Data-driven decision-making

Scalable smart city monitoring solutions

🛠️ Tech Stack
Category	Technology
Language	Python 3.9+
Model	YOLOv8 (Ultralytics)
Backend	Flask
Frontend	HTML, CSS, Bootstrap 5
Charts	Chart.js
Database	SQLite
Dataset	Roboflow
Image Processing	Pillow
Training	Google Colab (GPU)

📂 Project Structure
AI-Smart-City/
│
├── app.py                     # Flask application (production inference)
├── yolov8m.pt                 # Custom-trained YOLOv8 model
├── reports.db                 # SQLite database (report history)
│
├── templates/
│   ├── index.html             # Main detection UI
│   └── history.html           # Report history & analytics
│
├── static/
│   ├── style.css              # UI styling
│   ├── script.js              # Frontend logic (upload, camera, charts)
│   └── reports/               # Saved report images (ignored in Git)
│
├── model-training.ipynb        # Training notebook (Google Colab)
├── notebook.ipynb             # Evaluation & inference testing
│
├── requirements.txt           # Dependencies
├── .gitignore                 # Ignored runtime & generated files
└── README.md                  # Project documentation

📊 Dataset & Annotation

Platform: Roboflow

Classes:

pothole

garbage

Annotation & preprocessing:

Bounding box annotation

Data augmentation (flip, rotate, brightness, blur)

Train / validation / test split

Dataset size: ~2,000+ images

🧪 Model Training

Architecture: YOLOv8m

Framework: Ultralytics YOLO

Environment: Google Colab (GPU)

Training Notebook: model-training.ipynb

⚠️ Note: Training notebooks are GPU-oriented and not intended for local execution without GPU support.

📊 Model Evaluation (Current)
Metric	Value
Precision	~0.62
Recall	~0.58
mAP@0.5	~0.60

Metrics are expected to improve with dataset expansion and tuning.

🖥️ Application Workflow

User uploads an image or captures via camera

YOLOv8 model performs object detection

Bounding boxes are drawn on the image

Issues are counted per class

Severity is calculated automatically

Report is saved to database

Analytics dashboard updates in real-time

⚙️ Installation & Usage
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Application
python app.py

3️⃣ Open in Browser
http://127.0.0.1:5000

📈 Example Output

Detected: 2 potholes and 1 garbage

Severity: Medium

Visual bounding boxes on image

Summary:

“Total of 3 issues detected: 2 potholes and 1 garbage.”

🧩 Challenges & Learnings
Challenges

High variability in pothole shapes and lighting

Class imbalance between pothole and garbage

Limited annotated data

CPU-based inference latency

Key Learnings

Importance of data-centric AI development

Backend + ML integration challenges

Designing scalable ML web applications

Managing inference pipelines in production

📈 Accuracy Improvement Plan

Expand dataset to 5,000+ images

Add hard-negative samples

YOLOv8 hyperparameter tuning

Train with early stopping

Experiment with YOLOv8l architecture

🔐 Limitations

Image-based inference only

No real-time video stream yet

Performance depends on image quality

CPU inference slower than GPU

🚧 Future Enhancements

Real-time video & CCTV stream detection

GPS-based issue mapping

Role-based authentication (admin / user)

REST API for mobile app integration

Cloud deployment (AWS / GCP)

👤 Author

Shreeyash Paraj
Data Science Intern | AI & Backend Development
Project built to demonstrate real-world ML system design & deployment