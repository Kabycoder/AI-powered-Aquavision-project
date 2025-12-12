from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import cv2
import numpy as np
import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import io
import rasterio
from fpdf import FPDF
import os
from datetime import datetime
import sqlite3
import json

app = FastAPI(title="AquaVision API", version="1.0.0")

# Load pretrained DeepLab model for segmentation
model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

# COCO class for water is 41
WATER_CLASS = 41

# Transform for input image
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Database setup (simple SQLite for prototype)
def init_db():
    conn = sqlite3.connect('aquavision.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS water_analysis (
        id INTEGER PRIMARY KEY,
        original_image TEXT,
        segmented_image TEXT,
        area REAL,
        volume REAL,
        location TEXT,
        date TEXT,
        alert TEXT,
        pdf_file TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

@app.post("/detect-water")
async def detect_water(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    original_image = np.array(image)

    # Preprocess for model
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

    # Get water mask
    water_mask = (output_predictions == WATER_CLASS).astype(np.uint8) * 255

    # Calculate area (assume pixel size 10m x 10m = 100 sqm per pixel)
    pixel_area = 100  # sqm
    water_pixels = np.sum(water_mask > 0)
    area_sqm = water_pixels * pixel_area

    # Create overlay
    overlay = original_image.copy()
    overlay[water_mask > 0] = [0, 0, 255]  # blue for water
    overlay = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)

    # Save files (in memory for response)
    mask_pil = Image.fromarray(water_mask)
    overlay_pil = Image.fromarray(overlay)

    # For response, return base64 or paths, but for simplicity, return data
    # Actually, return JSON with area, and perhaps URLs, but since no storage, return data

    return {
        "area_sqm": area_sqm,
        "mask": mask_pil.tobytes(),  # or base64
        "overlay": overlay_pil.tobytes()
    }

@app.post("/estimate-volume")
async def estimate_volume(mask_file: UploadFile = File(...), dem_file: UploadFile = File(...), water_level: float = 100.0):
    # Read mask
    mask_data = await mask_file.read()
    mask = np.array(Image.open(io.BytesIO(mask_data)))

    # Read DEM
    dem_data = await dem_file.read()
    with rasterio.open(io.BytesIO(dem_data)) as dem:
        elevation = dem.read(1)

    # Assume pixel size from DEM
    pixel_area = dem.res[0] * dem.res[1]  # sqm

    # Calculate volume
    volume = 0.0
    mask_binary = mask > 0
    for i in range(elevation.shape[0]):
        for j in range(elevation.shape[1]):
            if mask_binary[i, j]:
                depth = max(0, water_level - elevation[i, j])
                volume += depth * pixel_area

    return {"volume_cubic_m": volume}

@app.post("/generate-report")
async def generate_report(area: float, volume: float, location: str = "Unknown"):
    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AquaVision Water Analysis Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Location: {location}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(200, 10, txt=f"Water Area: {area} sqm", ln=True)
    pdf.cell(200, 10, txt=f"Estimated Volume: {volume} cubic meters", ln=True)

    # Save PDF to file
    pdf_path = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_path)

    # Save to DB
    conn = sqlite3.connect('aquavision.db')
    c = conn.cursor()
    c.execute("INSERT INTO water_analysis (area, volume, location, date, pdf_file) VALUES (?, ?, ?, ?, ?)",
              (area, volume, location, datetime.now().isoformat(), pdf_path))
    conn.commit()
    conn.close()

    return FileResponse(pdf_path, media_type='application/pdf', filename=pdf_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)