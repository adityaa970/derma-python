import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import os
from PIL import Image

# Try to import TensorFlow, fall back to demo mode if it fails
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
    TF_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ TensorFlow not available: {e}")
    print("🔄 Will run in DEMO MODE only")
    TF_AVAILABLE = False
    tf = None

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
# Only truly dangerous conditions are marked High, everything else is safe
RISK_MAPPING = {
    "Melanoma": "High",
    "Basal Cell Carcinoma": "Moderate",
    "Actinic Keratosis": "Moderate",
    "Melanocytic Nevi": "Low",
    "Benign Keratosis": "Low",
    "Dermatofibroma": "Low",
    "Vascular Lesions": "Low"
}

# High confidence threshold to declare High risk
HIGH_RISK_CONFIDENCE_THRESHOLD = 0.70  # 70% confidence needed for High risk

# Minimum confidence to declare ANY disease — below this = Normal/Healthy skin
DISEASE_CONFIDENCE_THRESHOLD = 0.60  # Top class must score >= 60% to declare a disease

# Entropy threshold — if predictions are spread (uncertain), treat as Normal
# Max entropy for 7 classes = log(7) ≈ 1.95; uncertain model scores > 1.2
NORMAL_ENTROPY_THRESHOLD = 1.20

# --- DEMO MODE FLAG ---
DEMO_MODE = False  # Set to True when no model is available

# --- STARTUP EVENT ---
@app.on_event("startup")
async def load_ai_assets():
    global model, CLASS_NAMES, DEMO_MODE
    
    if not TF_AVAILABLE:
        print("🔄 TensorFlow not available - running in DEMO MODE")
        DEMO_MODE = True
        CLASS_NAMES = [
            "Actinic Keratosis",
            "Basal Cell Carcinoma", 
            "Benign Keratosis",
            "Dermatofibroma",
            "Melanocytic Nevi",
            "Melanoma",
            "Vascular Lesions"
        ]
        return
    
    # Load the class names
    try:
        with open("class_names.txt", "r") as f:
            CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
        print(f"✅ Loaded {len(CLASS_NAMES)} disease classes.")
    except Exception as e:
        print(f"⚠️ Could not load class_names.txt: {e}")
        CLASS_NAMES = [
            "Actinic Keratosis",
            "Basal Cell Carcinoma", 
            "Benign Keratosis",
            "Dermatofibroma",
            "Melanocytic Nevi",
            "Melanoma",
            "Vascular Lesions"
        ]
    
    NUM_CLASSES = len(CLASS_NAMES)
    
    try:
        # Try paths in order of preference
        full_model_path = "dermasetu_model.h5"
        weights_path = "traning/mobilenet_v2.h5"
        
        if os.path.exists(full_model_path):
            # Option 1: Load full model directly
            model = load_model(full_model_path)
            print(f"✅ Loaded complete model from {full_model_path}")
        elif os.path.exists(weights_path):
            # Option 2: Build architecture and load weights
            print(f"🔄 Building model architecture and loading weights from {weights_path}...")
            
            # Build the EXACT same architecture as training (without data augmentation)
            base_model = MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Create inference model (matches training architecture)
            inputs = Input(shape=(224, 224, 3))
            # Preprocessing is included as a layer in the trained model
            x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.3)(x)
            outputs = Dense(NUM_CLASSES, activation='softmax')(x)
            
            model = Model(inputs, outputs)
            
            # Try loading weights
            try:
                model.load_weights(weights_path)
                print(f"✅ Loaded trained weights from {weights_path}")
            except Exception as weight_err:
                print(f"⚠️ Could not load weights: {weight_err}")
                print("🔄 Using ImageNet pretrained weights (not trained on skin diseases)")
            
            print(f"✅ Model ready with {NUM_CLASSES} disease classes")
        else:
            print("⚠️ No model or weights found!")
            print("🔄 Running in DEMO MODE")
            DEMO_MODE = True
            return
        
        DEMO_MODE = False
        
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        print("🔄 Running in DEMO MODE with simulated predictions")
        DEMO_MODE = True

# --- HELPER FUNCTIONS ---

def is_skin_image(img: Image.Image, threshold: float = 0.25) -> tuple[bool, float]:
    """
    Validates if the image contains skin-like pixels with stricter detection.
    Now requires at least 25% skin-like pixels to pass validation.
    
    Returns:
        tuple: (is_valid, skin_percentage)
    """
    try:
        # Convert to RGB and resize for faster processing
        img_rgb = img.convert("RGB").resize((50, 50))  # Smaller for speed
        img_array = np.array(img_rgb, dtype=np.float32)
        
        r = img_array[:,:,0]
        g = img_array[:,:,1]
        b = img_array[:,:,2]
        
        # More restrictive skin detection
        # Light skin - tighter ranges
        light_skin = (
            (r > 120) & (g > 60) & (b > 40) &
            (r >= g) & (r >= b) &
            ((r - g) > 10) & ((r - b) > 15)
        )
        
        # Medium/brown skin - more restrictive
        medium_skin = (
            (r > 80) & (g > 50) & (b > 30) &
            (r >= b) & ((r - b) > 10) &
            ((r + g) > (b * 2))
        )
        
        # Dark skin - require obvious skin characteristics
        dark_skin = (
            (r > 60) & (g > 40) & (b > 20) &
            (r >= b) & ((r - b) > 5) &
            ((r + g) > (b * 1.8))
        )
        
        # Pink/reddish (skin conditions) - more restrictive
        pinkish = (
            (r > 130) & (g > 70) & (b > 60) &
            (r > g) & ((r - g) > 15)
        )
        
        # Beige/tan tones - higher thresholds
        beige = (
            (r > 160) & (g > 120) & (b > 80) &
            (r > b) & ((r - b) > 20)
        )
        
        # Combine all
        skin_mask = light_skin | medium_skin | dark_skin | pinkish | beige
        
        skin_percentage = float(np.mean(skin_mask))
        
        return skin_percentage >= threshold, skin_percentage
        
    except Exception as e:
        print(f"Skin detection error: {e}")
        # If detection fails, allow the image through
        return True, 0.5

def preprocess_image(img_bytes):
    """
    Converts uploaded bytes to a format compatible with the trained model.
    NOTE: The trained model has preprocess_input built-in as a layer,
    so we just pass raw pixel values (0-255).
    """
    try:
        # Open image and convert to RGB (handles PNGs with transparency)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Resize to 224x224 (Model Input Shape)
        img = img.resize((224, 224))
        
        # Convert to Array with float32 dtype (required by model)
        img_array = np.array(img, dtype=np.float32)
        
        # Add Batch Dimension (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Return raw pixel values - model has preprocess_input built-in
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "model_loaded": model is not None, 
        "demo_mode": DEMO_MODE,
        "tensorflow_available": TF_AVAILABLE,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyze a skin image and return diagnosis.
    Returns error if the image doesn't appear to be a skin image.
    """
    import random
    
    # 1. Read the image
    img_bytes = await file.read()
    
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")
    
    # 2. Validate skin image
    is_skin, skin_percentage = is_skin_image(img)
    print(f"🔬 Skin detection: {skin_percentage*100:.1f}%")
    
    if not is_skin:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image: This doesn't appear to be a skin image. Please upload a clear photo of the affected skin area. (Skin detection: {skin_percentage*100:.1f}%)"
        )
    
    # 3. Check if running in demo mode (TensorFlow not available)
    if DEMO_MODE:
        # Demo mode - 40% chance of Normal, rest are benign conditions
        demo_options = [
            ("Normal / Healthy Skin", "Normal"),
            ("Normal / Healthy Skin", "Normal"),
            ("Melanocytic Nevi", "Low"),
            ("Benign Keratosis", "Low"),
            ("Dermatofibroma", "Low"),
        ]
        diagnosis, risk_level = random.choice(demo_options)
        confidence = random.uniform(55.0, 85.0)

        if risk_level == "Normal":
            action_plan = "Your skin looks healthy! No signs of a skin condition were detected. Keep up good skin care habits and use sunscreen regularly."
        else:
            action_plan = "This appears to be a common, benign skin condition. Monitor for any changes. No immediate action needed."

        print(f"📋 [DEMO] Simulated diagnosis: {diagnosis} ({confidence:.1f}%)")

        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "risk_level": risk_level,
            "action_plan": action_plan,
            "demo_mode": True,
            "skin_detected": round(skin_percentage * 100, 1)
        }
    
    # 4. Real model inference
    if model is None:
        raise HTTPException(status_code=503, detail="AI Model is not loaded. Please try again later.")

    # Preprocess the image
    processed_img = preprocess_image(img_bytes)
    
    print(f"🔍 Running inference with trained model... Input shape: {processed_img.shape}")

    # Run model inference
    predictions = model.predict(processed_img, verbose=0)
    
    print(f"📊 Raw predictions: {predictions[0]}")
    
    # Get the prediction results
    confidence = float(np.max(predictions))
    class_index = int(np.argmax(predictions))
    
    print(f"✅ Predicted class index: {class_index}, confidence: {confidence*100:.1f}%")
    
    # Confidence threshold - if below DISEASE_CONFIDENCE_THRESHOLD, skin is Normal/Healthy
    MIN_CONFIDENCE = 0.10  # Absolute floor — below this, image is unclear
    if confidence < MIN_CONFIDENCE:
        raise HTTPException(
            status_code=400,
            detail=f"Could not identify the skin area in this image (confidence: {confidence*100:.1f}%). Please upload a clear, close-up photo of the affected skin area."
        )

    # Entropy check: if prediction distribution is too spread out, model is uncertain → Normal
    # Shannon entropy: H = -sum(p * log2(p)), max for 7 classes ≈ 2.807
    probs = predictions[0]
    probs_safe = np.clip(probs, 1e-9, 1.0)  # prevent log(0)
    entropy = float(-np.sum(probs_safe * np.log2(probs_safe)))
    print(f"📊 Prediction entropy: {entropy:.3f} (threshold: {NORMAL_ENTROPY_THRESHOLD})")

    # If model isn't confident about any specific disease → Normal skin
    is_normal = (confidence < DISEASE_CONFIDENCE_THRESHOLD) or (entropy > NORMAL_ENTROPY_THRESHOLD)
    if is_normal:
        print(f"✅ Returning Normal/Healthy Skin (confidence={confidence*100:.1f}%, entropy={entropy:.3f})")
        return {
            "diagnosis": "Normal / Healthy Skin",
            "confidence": round(confidence * 100, 2),
            "risk_level": "Normal",
            "action_plan": "Your skin looks healthy! No signs of a skin condition were detected. Keep up good skin care habits and use sunscreen regularly.",
            "demo_mode": False,
            "skin_detected": round(skin_percentage * 100, 1),
            "all_predictions": {
                CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
                for i in range(min(len(CLASS_NAMES), len(predictions[0])))
            }
        }

    diagnosis = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else "Unknown"

    # Determine Risk Level - Conservative approach
    # Only show "High" risk if:
    # 1. The condition is genuinely high-risk (Melanoma)
    # 2. Confidence is very high (>70%)
    base_risk = RISK_MAPPING.get(diagnosis, "Low")
    
    # Downgrade risk if confidence is not high enough
    if base_risk == "High" and confidence < HIGH_RISK_CONFIDENCE_THRESHOLD:
        risk_level = "Moderate"  # Not confident enough to declare High risk
        print(f"⚠️ Downgraded risk from High to Moderate (confidence {confidence*100:.1f}% < {HIGH_RISK_CONFIDENCE_THRESHOLD*100}%)")
    elif base_risk == "Moderate" and confidence < 0.50:
        risk_level = "Low"  # Not confident enough for Moderate
        print(f"⚠️ Downgraded risk from Moderate to Low (confidence {confidence*100:.1f}% < 50%)")
    else:
        risk_level = base_risk 
    
    # Determine Action Plan - Reassuring by default
    if risk_level == "High":
        action_plan = "Please consult a dermatologist for professional evaluation. Early consultation is recommended."
    elif risk_level == "Moderate":
        action_plan = "Consider scheduling a dermatology appointment for a routine check-up within the next few weeks."
    else:
        action_plan = "This appears to be a common, benign skin condition. Monitor for any changes and maintain good skin care. No immediate action needed."

    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence * 100, 2),
        "risk_level": risk_level,
        "action_plan": action_plan,
        "demo_mode": False,
        "skin_detected": round(skin_percentage * 100, 1),
        "all_predictions": {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2) 
            for i in range(min(len(CLASS_NAMES), len(predictions[0])))
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
