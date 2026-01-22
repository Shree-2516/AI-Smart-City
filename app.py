"""
Smart City Issue Detection - Flask Web App
-----------------------------------------
Author: Shree
Model: YOLOv8 (Pre-trained)
"""

import os
import sys
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from flask import Flask, redirect, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import base64

# ===============================
# CONFIG
# ===============================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolov8m.pt"

CONF_THRESHOLD = 0.25
MAX_DET = 5

# ===============================
# INIT APP
# ===============================

app = Flask(__name__)

if not MODEL_PATH.exists():
    sys.exit("❌ Model file not found")

print("📦 Loading YOLOv8 model...")
model = YOLO(str(MODEL_PATH))
print("✅ Model loaded")

# ===============================
# DATABASE SETUP
# ===============================

DB_PATH = BASE_DIR / "reports.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            summary TEXT,
            severity TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ===============================
# INFERENCE LOGIC
# ===============================

def run_inference(image: Image.Image):
    results = model.predict(image, conf=CONF_THRESHOLD, max_det=MAX_DET)
    r = results[0]

    summary: Dict[str, int] = {}
    if r.boxes is not None:
        for cls in r.boxes.cls.tolist():
            name = model.names[int(cls)]
            summary[name] = summary.get(name, 0) + 1

    output = r.plot()
    output_image = Image.fromarray(output[..., ::-1])

    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode()

    return img_base64, summary

# ===============================
# ROUTES
# ===============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    image = Image.open(file.stream).convert("RGB")
    img_base64, summary = run_inference(image)

    # ---- SEVERITY ----
    total = sum(summary.values())
    if total <= 1:
        severity = "Low"
    elif total <= 3:
        severity = "Medium"
    else:
        severity = "High"

    # ---- SAVE IMAGE ----
    reports_dir = BASE_DIR / "static" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image_path = reports_dir / filename
    image.save(image_path)

    # ---- SAVE DB ----
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reports (image_path, summary, severity, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        f"static/reports/{filename}",
        json.dumps(summary),
        severity,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    return jsonify({
        "image": img_base64,
        "summary": summary,
        "severity": severity
    })

@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, image_path, summary, severity, created_at
        FROM reports
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    conn.close()

    reports = []

    total_reports = len(rows)
    total_garbage = 0
    total_pothole = 0
    no_issue_reports = 0

    for report_id, image_path, summary, severity, created_at in rows:
        summary_dict = json.loads(summary)

        if not summary_dict:
            no_issue_reports += 1
        else:
            for k, v in summary_dict.items():
                if "garbage" in k.lower():
                    total_garbage += v
                elif "pothole" in k.lower():
                    total_pothole += v

        reports.append({
            "id": report_id,
            "image_path": image_path,
            "summary": summary_dict,
            "severity": severity,
            "created_at": created_at
        })

    summary_stats = {
        "total_reports": total_reports,
        "total_garbage": total_garbage,
        "total_pothole": total_pothole,
        "no_issue_reports": no_issue_reports
    }

    return render_template(
        "history.html",
        reports=reports,
        stats=summary_stats
    )


@app.route("/delete/<int:report_id>", methods=["POST"])
def delete_report(report_id):
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

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    app.run(debug=True)
