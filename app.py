"""
Smart City Issue Detection - Flask Web App
-----------------------------------------
Author: Shree
Model: YOLOv8 (Custom-trained)

Description:
- Upload or capture images
- Detect garbage & potholes using YOLOv8
- Store reports in SQLite
- Visualize history, stats & heatmaps
"""

# =====================================================
# STANDARD LIBRARIES
# =====================================================
import os
import sys
import sqlite3
import json
import io
import base64
import csv
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# =====================================================
# THIRD-PARTY LIBRARIES
# =====================================================
from flask import Flask, redirect, render_template, request, jsonify, Response, send_file
from ultralytics import YOLO
from PIL import Image

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "yolov8m.pt"
DB_PATH = BASE_DIR / "reports.db"

CONF_THRESHOLD = 0.25
MAX_DET = 5

# =====================================================
# FLASK APP INITIALIZATION
# =====================================================

app = Flask(__name__)

# =====================================================
# MODEL LOADING
# =====================================================

if not MODEL_PATH.exists():
    sys.exit("❌ Model file not found")

print("📦 Loading YOLOv8 model...")
model = YOLO(str(MODEL_PATH))
print("✅ Model loaded")

# =====================================================
# DATABASE INITIALIZATION
# =====================================================

def init_db():
    """
    Create reports table if it does not exist.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            summary TEXT,
            severity TEXT,
            latitude REAL,
            longitude REAL,
            created_at TEXT,
            feedback TEXT DEFAULT NULL,
            type TEXT DEFAULT 'image'
        )
    """)
    
    # Check if 'type' column exists (migration for existing DB)
    cur.execute("PRAGMA table_info(reports)")
    columns = [info[1] for info in cur.fetchall()]
    if 'type' not in columns:
        print("⚠️ Migrating database: Adding 'type' column...")
        cur.execute("ALTER TABLE reports ADD COLUMN type TEXT DEFAULT 'image'")

    if 'feedback' not in columns:
        print("⚠️ Migrating database: Adding 'feedback' column...")
        cur.execute("ALTER TABLE reports ADD COLUMN feedback TEXT DEFAULT NULL")

    if 'department' not in columns:
        print("⚠️ Migrating database: Adding 'department' column...")
        cur.execute("ALTER TABLE reports ADD COLUMN department TEXT DEFAULT 'General'")

    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# =====================================================
# HOME PAGE STATS
# =====================================================

def get_home_stats():
    """
    Fetch aggregated statistics for homepage dashboard.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Total reports
    cur.execute("SELECT COUNT(*) FROM reports")
    total_reports = cur.fetchone()[0]

    # Aggregate detected issues
    cur.execute("SELECT summary FROM reports")
    rows = cur.fetchall()

    total_potholes = 0
    total_garbage = 0

    for (summary,) in rows:
        if summary:
            try:
                data = json.loads(summary)
                for key, value in data.items():
                    if "pothole" in key.lower():
                        total_potholes += value
                    elif "garbage" in key.lower():
                        total_garbage += value
            except json.JSONDecodeError:
                continue

    # Calculate Dynamic Accuracy based on user feedback
    # cur.execute("SELECT COUNT(*) FROM reports WHERE feedback IS NOT NULL")
    # total_feedback = cur.fetchone()[0]

    # cur.execute("SELECT COUNT(*) FROM reports WHERE feedback = 'correct'")
    # correct_feedback = cur.fetchone()[0]

    # if total_feedback > 0:
    #     accuracy = int((correct_feedback / total_feedback) * 100)
    # else:
    #     accuracy = 98 # Default/Fallback accuracy
    
    # User requested static accuracy between 70-80%
    accuracy = 78

    conn.close()

    return {
        "total_reports": total_reports,
        "total_potholes": total_potholes,
        "total_garbage": total_garbage,
        "avg_inference": "120ms",
        "model_accuracy": accuracy
    }

# =====================================================
# INFERENCE LOGIC
# =====================================================

def run_inference(image: Image.Image):
    """
    Run YOLOv8 inference on input image and
    return annotated image + detection summary.
    """
    results = model.predict(
        image,
        conf=CONF_THRESHOLD,
        max_det=MAX_DET
    )

    result = results[0]

    # Build class summary
    summary: Dict[str, int] = {}

    if result.boxes is not None:
        for cls in result.boxes.cls.tolist():
            class_name = model.names[int(cls)]
            summary[class_name] = summary.get(class_name, 0) + 1

    # Render annotated image
    output = result.plot()
    output_image = Image.fromarray(output[..., ::-1])

    # Convert image to base64
    buffer = io.BytesIO()
    output_image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return img_base64, summary

def process_video_frames(video_path: str) -> Tuple[Dict[str, int], List[str]]:
    """
    Process video frames:
    - Skip frames (process 1 per second)
    - Detect issues
    - Save key frames (frames with detections)
    - Return aggregate summary + list of keyframe paths
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_interval = fps  # Process 1 frame per second
    
    total_summary = {}
    key_frame_paths = []
    
    frame_count = 0
    saved_frames_count = 0
    max_saved_frames = 10 # Limit number of saved frames per video to save space
    
    reports_dir = BASE_DIR / "static" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Run inference on frame
            # Convert BGR (OpenCV) to RGB (PIL)
            # But YOLO can take numpy array (BGR) directly? Yes.
            results = model.predict(frame, conf=CONF_THRESHOLD, max_det=MAX_DET, verbose=False)
            result = results[0]
            
            has_detection = False
            local_summary = {}
            
            if result.boxes is not None:
                for cls in result.boxes.cls.tolist():
                    class_name = model.names[int(cls)]
                    local_summary[class_name] = local_summary.get(class_name, 0) + 1
                    total_summary[class_name] = total_summary.get(class_name, 0) + 1
                    has_detection = True
            
            if has_detection and saved_frames_count < max_saved_frames:
                # Save this frame as a "highlight"
                annotated_frame = result.plot()
                
                # Save to disk
                filename = f"video_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{saved_frames_count}.jpg"
                save_path = reports_dir / filename
                cv2.imwrite(str(save_path), annotated_frame)
                
                key_frame_paths.append(f"static/reports/{filename}")
                saved_frames_count += 1
                
        frame_count += 1
        
    cap.release()
    return total_summary, key_frame_paths

# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def index():
    """
    Homepage with stats preview.
    """
    stats = get_home_stats()
    return render_template("index.html", stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle image upload / camera capture,
    run inference, store report, return result.
    """
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    image = Image.open(file.stream).convert("RGB")

    img_base64, summary = run_inference(image)

    # Parse location data
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")

    try:
        latitude = float(latitude) if latitude else None
        longitude = float(longitude) if longitude else None
    except ValueError:
        latitude = longitude = None

    # Determine severity
    total_issues = sum(summary.values())
    if total_issues <= 1:
        severity = "Low"
    elif total_issues <= 3:
        severity = "Medium"
    else:
        severity = "High"

    # Save annotated image
    reports_dir = BASE_DIR / "static" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image_path = reports_dir / filename
    image.save(image_path)

    # Auto-Dispatch Logic and Save report to database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Determine department based on detected issues
    # Determine department based on detected issues
    department = "General"
    for class_name in summary.keys():
        if "pothole" in class_name.lower():
            department = "Roads Department"
            break 
        elif "garbage" in class_name.lower():
            department = "Department of Environment"
            break

    cur.execute("""
        INSERT INTO reports
        (image_path, summary, severity, latitude, longitude, created_at, type, department)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        f"static/reports/{filename}",
        json.dumps(summary),
        severity,
        latitude,
        longitude,
        datetime.now().isoformat(),
        'image',
        department
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "image": img_base64,
        "summary": summary,
        "severity": severity,
        "report_id": cur.lastrowid,
        "department": department
    })

@app.route("/predict-video", methods=["POST"])
def predict_video():
    """
    Handle video upload
    """
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video provided"}), 400
        
    # Save temp video
    temp_dir = BASE_DIR / "static" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    file.save(temp_path)
    
    # Process
    summary, key_frames = process_video_frames(temp_path)
    
    # Cleanup temp video
    if temp_path.exists():
        os.remove(temp_path)
        
    # Determine overall severity
    total_issues = sum(summary.values())
    if total_issues <= 5: severity = "Low"
    elif total_issues <= 15: severity = "Medium"
    else: severity = "High"
    
    # Check if we should save a "Video Report" to DB
    # For now, let's just save one entry representing the video analysis with the first keyframe as the thumb
    report_id = None
    if key_frames:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO reports
            (image_path, summary, severity, latitude, longitude, created_at, type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            key_frames[0], # Use first detected frame as thumbnail
            json.dumps(summary),
            severity,
            None, None, # No location for video uploads yet
            datetime.now().isoformat(),
            'video'
        ))
        
        report_id = cur.lastrowid
        conn.commit()
        conn.close()
        
    return render_template("video_result.html", 
        summary=summary, 
        key_frames=key_frames, 
        severity=severity,
        report_id=report_id
    )

@app.route("/export-csv")
def export_csv():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM reports")
    rows = cur.fetchall()
    
    # Get column names
    column_names = [description[0] for description in cur.description]
    
    conn.close()
    
    # Generate CSV in memory
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(column_names)
    cw.writerows(rows)
    
    output = io.BytesIO()
    output.write(si.getvalue().encode('utf-8'))
    output.seek(0)
    
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"reports_export_{datetime.now().strftime('%Y%m%d')}.csv"
    )

# =====================================================
# FeedBack ROUTE
# =====================================================

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()

    report_id = data.get("report_id")
    feedback_value = data.get("feedback")

    if not report_id or feedback_value not in ("correct", "incorrect"):
        return jsonify({"error": "Invalid feedback"}), 400

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "UPDATE reports SET feedback = ? WHERE id = ?",
        (feedback_value, report_id)
    )

    conn.commit()
    conn.close()

    return jsonify({"status": "feedback saved"})



# =====================================================
# HISTORY & ANALYTICS
# =====================================================

@app.route("/history")
def history():
    """
    Display report history with stats, maps, and heatmap.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, image_path, summary, severity, latitude, longitude, created_at, type, department
        FROM reports
        ORDER BY id DESC
    """)
    rows = cur.fetchall()

    conn.close()

    reports = []
    heatmap_points = []

    total_reports = len(rows)
    total_garbage = 0
    total_pothole = 0
    no_issue_reports = 0

    for row in rows:
        # Handle new 'department' column safely
        try:
             (report_id, image_path, summary, severity, latitude, longitude, created_at, r_type, department) = row
        except ValueError:
             # Fallback for old DB structure
             try:
                (report_id, image_path, summary, severity, latitude, longitude, created_at, r_type) = row
                department = "General"
             except ValueError:
                (report_id, image_path, summary, severity, latitude, longitude, created_at) = row
                r_type = 'image'
                department = "General"

        summary_dict = json.loads(summary) if summary else {}

        if not summary_dict:
            no_issue_reports += 1
        else:
            for key, value in summary_dict.items():
                if "garbage" in key.lower():
                    total_garbage += value
                elif "pothole" in key.lower():
                    total_pothole += value

        reports.append({
            "id": report_id,
            "image_path": image_path,
            "summary": summary_dict,
            "severity": severity,
            "latitude": latitude,
            "longitude": longitude,
            "created_at": created_at,
            "type": r_type,
            "department": department if department else "General"
        })

        if latitude and longitude:
            weight = 0.5 if severity == "Low" else 1.0 if severity == "Medium" else 2.0
            heatmap_points.append([float(latitude), float(longitude), weight])

    summary_stats = {
        "total_reports": total_reports,
        "total_garbage": total_garbage,
        "total_pothole": total_pothole,
        "no_issue_reports": no_issue_reports
    }

    return render_template(
        "history.html",
        reports=reports,
        stats=summary_stats,
        heatmap_points=heatmap_points
    )

# =====================================================
# DELETE ROUTES
# =====================================================

@app.route("/delete-report/<int:report_id>", methods=["POST"])
def delete_report(report_id):
    """
    Delete a single report and its image.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT image_path FROM reports WHERE id = ?", (report_id,))
    row = cur.fetchone()

    if row:
        image_path = BASE_DIR / row[0]
        if image_path.exists():
            os.remove(image_path)

        cur.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        conn.commit()

    conn.close()
    return redirect("/history")


@app.route("/delete_all", methods=["POST"])
def delete_all_reports():
    """
    Delete all reports and images.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT image_path FROM reports")
    rows = cur.fetchall()

    for (path,) in rows:
        img = BASE_DIR / path
        if img.exists():
            os.remove(img)

    cur.execute("DELETE FROM reports")
    conn.commit()
    conn.close()

    return redirect("/history")

@app.route("/fix-departments", methods=["GET"])
def fix_departments():
    """Helper to migrate old department names to new ones"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("SELECT id, summary FROM reports")
    rows = cur.fetchall()
    
    count = 0
    for r in rows:
        rid, summary_str = r
        if not summary_str: continue
        
        try:
            summary = json.loads(summary_str)
            new_dept = "unassigned"
            
            # Check keys
            for key in summary.keys():
                if "pothole" in key.lower():
                    new_dept = "Roads Department"
                    break
                elif "garbage" in key.lower():
                    new_dept = "Department of Environment"
                    break
            
            if new_dept != "unassigned":
                cur.execute("UPDATE reports SET department = ? WHERE id = ?", (new_dept, rid))
                count += 1
                
        except:
            continue
            
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "updated_count": count})

# =====================================================
# APPLICATION ENTRY POINT
# =====================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",   # explicit localhost
        port=8000,          # avoids blocked port 5000
        debug=True
    )
