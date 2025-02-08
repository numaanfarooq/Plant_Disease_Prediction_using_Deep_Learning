from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import json
import uvicorn
from PIL import Image
import io
import os

# Initialize FastAPI
app = FastAPI()

# Configure template and static file locations
templates = Jinja2Templates(directory="../templates")
app.mount("/static", StaticFiles(directory="../static"), name="static")

# Load model and class names
MODEL_PATH = os.path.join("model", "plant_disease_model.keras")
CLASS_NAMES_PATH = os.path.join("model", "class_names.json")

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

def preprocess_image(image):
    # Resize and scale image (same as training)
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return {
        "class": predicted_class,
        "confidence": confidence,
        "all_predictions": dict(zip(class_names, predictions[0].tolist()))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)