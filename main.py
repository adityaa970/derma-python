import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# Initialize the App
app = FastAPI(
    title="DermaSetu AI API",
    description="Backend for DermaSetu Rural Dermatology Triage",
    version="1.0"
)

# --- CONFIGURATION ---
# Allow Next.js frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the model and class names
model = None
CLASS_NAMES = []

# --- RISK MAPPING ---
# Specific dermatological risk assessment
RISK_MAPPING = {
    "Melanoma": "High",
    "Basal Cell Carcinoma": "High",
    "Actinic Keratosis": "High",
    "Melanocytic Nevi": "Low",
    "Benign Keratosis": "Low",
    "Dermatofibroma": "Low",
    "Vascular Lesions": "Low"
}

# --- DEMO MODE FLAG ---
DEMO_MODE = False  # Set to True when no model is available

# --- STARTUP EVENT ---
@app.on_event("startup")
async def load_ai_assets():
    global model, CLASS_NAMES, DEMO_MODE
    try:
        # Load the trained .h5 model
        model = load_model("dermasetu_model.h5")
        print("✅ DermaSetu AI Model loaded successfully!")
        
        # Load the class names
        with open("class_names.txt", "r") as f:
            CLASS_NAMES = [line.strip() for line in f.readlines()]
        print(f"✅ Loaded {len(CLASS_NAMES)} disease classes.")
        
    except Exception as e:
        print(f"⚠️ Model not found: {e}")
        print("🔄 Running in DEMO MODE with simulated predictions")
        DEMO_MODE = True
        # Set default class names for demo mode
        CLASS_NAMES = [
            "Actinic Keratosis",
            "Basal Cell Carcinoma", 
            "Benign Keratosis",
            "Dermatofibroma",
            "Melanocytic Nevi",
            "Melanoma",
            "Vascular Lesions"
        ]

# --- HELPER FUNCTIONS ---
def preprocess_image(img_bytes):
    """
    Converts uploaded bytes to a format compatible with MobileNetV2
    """
    try:
        # Open image and convert to RGB (handles PNGs with transparency)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Resize to 224x224 (Model Input Shape)
        img = img.resize((224, 224))
        
        # Convert to Array
        img_array = image.img_to_array(img)
        
        # Add Batch Dimension (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess (Scale pixel values)
        return preprocess_input(img_array)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None, "demo_mode": DEMO_MODE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    import random
    
    # Check if running in demo mode
    if DEMO_MODE:
        # Demo mode - return random realistic prediction
        diagnosis = random.choice(CLASS_NAMES)
        confidence = random.uniform(75.0, 98.0)
        risk_level = RISK_MAPPING.get(diagnosis, "Moderate")
        action_plan = "Immediate Referral to Specialist" if risk_level == "High" else "Monitor for 2 weeks. Apply standard care."
        
        print(f"📋 [DEMO] Simulated diagnosis: {diagnosis} ({confidence:.1f}%)")
        
        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "risk_level": risk_level,
            "action_plan": action_plan,
            "demo_mode": True
        }
    
    # Real model inference
    if model is None:
        raise HTTPException(status_code=503, detail="AI Model is not loaded")

    # 1. Read and Preprocess Image
    img_bytes = await file.read()
    processed_img = preprocess_image(img_bytes)

    # 2. Run Inference
    predictions = model.predict(processed_img)
    
    # 3. Process Results
    confidence = float(np.max(predictions))
    class_index = np.argmax(predictions)
    diagnosis = CLASS_NAMES[class_index]
    
    # 4. Determine Risk Level
    # Default to "Moderate" if class name doesn't exactly match mapping
    risk_level = RISK_MAPPING.get(diagnosis, "Moderate") 
    
    # 5. Determine Action Plan
    action_plan = "Immediate Referral to Specialist" if risk_level == "High" else "Monitor for 2 weeks. Apply standard care."

    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence * 100, 2), # Return as percentage
        "risk_level": risk_level,
        "action_plan": action_plan
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)