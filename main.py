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
    Validates if the image contains skin-like pixels using strict color analysis.
    Uses RGB color rules to detect actual skin tones.
    
    Returns:
        tuple: (is_valid, skin_percentage)
    """
    try:
        # Convert to RGB and resize for faster processing
        img_rgb = img.convert("RGB").resize((100, 100))
        img_array = np.array(img_rgb, dtype=np.float32)
        
        r = img_array[:,:,0]
        g = img_array[:,:,1]
        b = img_array[:,:,2]
        
        # Strict skin detection rules based on dermatology research
        # Rule 1: Skin has more red than green, more green than blue typically
        # Rule 2: Skin falls within specific RGB ranges
        # Rule 3: R-G difference should be in a specific range for skin
        
        # Light/medium skin tones (strict)
        light_skin = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (g > b) &  # R > G > B typical for skin
            ((r - g) > 15) &  # Clear red dominance
            ((r - g) < 100) &  # But not too extreme (like red objects)
            (np.abs(r - g) <= np.abs(g - b) * 3)  # Balanced color distribution
        )
        
        # Brown/dark skin tones (strict)
        dark_skin = (
            (r > 50) & (g > 30) & (b > 15) &
            (r > b) &  # Red channel dominant over blue
            (r >= g) &  # Red >= green
            ((r - b) > 10) &  # Clear warm undertone
            (r < 200)  # Not too bright
        )
        
        # Pink/reddish skin (common in skin conditions like rashes)
        pinkish_skin = (
            (r > 150) & (g > 80) & (b > 80) &
            (r > g) & (r > b) &
            ((r - g) > 20) & ((r - b) > 20) &
            ((g - b) < 30)  # Green and blue relatively close (pinkish)
        )
        
        # Combine skin detections
        skin_mask = light_skin | dark_skin | pinkish_skin
        
        skin_percentage = float(np.mean(skin_mask))
        
        return skin_percentage >= threshold, skin_percentage
        
    except Exception as e:
        print(f"Skin detection error: {e}")
        return True, 0.0

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
        # Demo mode - return simulated prediction with warning
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
    
    # Confidence threshold - reject if model is too uncertain
    MIN_CONFIDENCE = 0.40  # 40% minimum
    if confidence < MIN_CONFIDENCE:
        raise HTTPException(
            status_code=400,
            detail=f"Could not identify a skin condition in this image (confidence: {confidence*100:.1f}%). Please upload a clear, close-up photo of the affected skin area."
        )
    
    diagnosis = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else "Unknown"
    
    # Determine Risk Level
    risk_level = RISK_MAPPING.get(diagnosis, "Moderate") 
    
    # Determine Action Plan
    if risk_level == "High":
        action_plan = "Immediate Referral to Specialist - This condition requires urgent medical attention."
    elif risk_level == "Moderate":
        action_plan = "Schedule a consultation with a dermatologist within 2 weeks."
    else:
        action_plan = "Monitor for 2 weeks. Apply standard care. Consult if symptoms worsen."

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
