import io
import json
import os
import sqlite3
from datetime import datetime
from typing import List

import cv2
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Table, TableStyle
from ultralytics import YOLO

app = FastAPI()

# Setup CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Database setup
DB_NAME = "detection_history.db"


def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='detections'"
        )
        table_exists = cursor.fetchone()

        if table_exists:
            # Check if columns exist and add them if missing
            cursor.execute("PRAGMA table_info(detections)")
            columns = [column[1] for column in cursor.fetchall()]

            if "detected_objects" not in columns:
                cursor.execute(
                    "ALTER TABLE detections ADD COLUMN detected_objects TEXT"
                )

            if "image_path" not in columns:
                cursor.execute("ALTER TABLE detections ADD COLUMN image_path TEXT")
        else:
            # Create new table with all columns
            cursor.execute(
                """
            CREATE TABLE detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                detected_objects TEXT NOT NULL,
                image_path TEXT
            )
            """
            )
        conn.commit()


init_db()


def save_detection(
    filename: str, detected_objects: List[dict], image_bytes: bytes = None
):
    timestamp = datetime.now().isoformat()
    objects_json = json.dumps(detected_objects)
    image_path = None

    if image_bytes:
        os.makedirs("detection_images", exist_ok=True)
        image_path = f"detection_images/{timestamp}_{filename}"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO detections (timestamp, filename, detected_objects, image_path) VALUES (?, ?, ?, ?)",
            (timestamp, filename, objects_json, image_path),
        )
        conn.commit()


def get_detection_history():
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC")
        return cursor.fetchall()


def generate_pdf_report(detections):
    filename = "detection_report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom style for title
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=20,
        alignment=1,  # Center aligned
    )

    elements = []

    # Add title
    elements.append(Paragraph("Detection History Report", title_style))

    # Prepare data for table
    data = [["Timestamp", "Filename", "Detected Objects", "Count"]]

    for detection in detections:
        objects = json.loads(detection["detected_objects"])
        object_counts = {}
        for obj in objects:
            object_counts[obj["class"]] = object_counts.get(obj["class"], 0) + 1

        objects_str = ", ".join([f"{k} ({v})" for k, v in object_counts.items()])
        data.append(
            [
                detection["timestamp"],
                detection["filename"],
                objects_str,
                str(len(objects)),
            ]
        )

    # Create table
    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    elements.append(table)
    doc.build(elements)
    return filename


def generate_excel_report(detections):
    filename = "detection_report.xlsx"

    data = []
    for detection in detections:
        objects = json.loads(detection["detected_objects"])
        object_counts = {}
        for obj in objects:
            object_counts[obj["class"]] = object_counts.get(obj["class"], 0) + 1

        data.append(
            {
                "Timestamp": detection["timestamp"],
                "Filename": detection["filename"],
                "Detected Objects": ", ".join(
                    [f"{k} ({v})" for k, v in object_counts.items()]
                ),
                "Total Detections": len(objects),
            }
        )

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    return filename


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run YOLO inference
    results = model(img)
    detected_objects = []

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding boxes
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Save detection info
            detected_objects.append(
                {"class": label, "confidence": conf, "bbox": xyxy.tolist()}
            )

    # Save to history
    _, png_img = cv2.imencode(".png", img)
    image_bytes = png_img.tobytes()
    save_detection(file.filename, detected_objects, image_bytes)

    # Return processed image
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@app.get("/history")
async def get_history():
    return get_detection_history()


@app.get("/report/pdf")
async def generate_pdf():
    detections = get_detection_history()
    if not detections:
        raise HTTPException(status_code=404, detail="No detection history found")

    filename = generate_pdf_report(detections)
    return FileResponse(filename, media_type="application/pdf", filename=filename)


@app.get("/report/excel")
async def generate_excel():
    detections = get_detection_history()
    if not detections:
        raise HTTPException(status_code=404, detail="No detection history found")

    filename = generate_excel_report(detections)
    return FileResponse(
        filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
