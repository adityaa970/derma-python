import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import os
from PIL import Image

# -- TensorFlow (optional -- falls back to demo mode) -------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow not available: {e}")
    TF_AVAILABLE = False
    tf = None

# -- App ----------------------------------------------------------------------
app = FastAPI(title="DermaSetu AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Globals ------------------------------------------------------------------
model = None
CLASS_NAMES = []
DEMO_MODE = False

# -- Constants ----------------------------------------------------------------
WEIGHTS_PATH = "traning/mobilenet_v2.h5"
CLASS_FILE   = "class_names.txt"

# If the top softmax probability is below this, return Normal/Healthy.
# HAM10000-trained MobileNetV2 peaks at 50-70% on real images, so 0.40 is safe.
NORMAL_THRESHOLD = 0.40

RISK_MAP = {
    "Melanoma":             "High",
    "Basal Cell Carcinoma": "Moderate",
    "Actinic Keratosis":    "Moderate",
    "Melanocytic Nevi":     "Low",
    "Benign Keratosis":     "Low",
    "Dermatofibroma":       "Low",
    "Vascular Lesions":     "Low",
}

ACTION_MAP = {
    "High":    "Please consult a dermatologist promptly. Early evaluation is important.",
    "Moderate":"Schedule a dermatology appointment for a routine check-up soon.",
    "Low":     "Appears to be a common benign condition. Monitor for changes; no immediate action needed.",
    "Normal":  "Your skin looks healthy! Keep up good skin-care habits and use sunscreen daily.",
}

# -- Startup: load model ------------------------------------------------------
@app.on_event("startup")
async def load_assets():
    global model, CLASS_NAMES, DEMO_MODE

    # Load class names
    try:
        with open(CLASS_FILE) as f:
            CLASS_NAMES = [l.strip() for l in f if l.strip()]
        print(f"Loaded {len(CLASS_NAMES)} classes.")
    except Exception:
        CLASS_NAMES = [
            "Actinic Keratosis", "Basal Cell Carcinoma", "Benign Keratosis",
            "Dermatofibroma", "Melanocytic Nevi", "Melanoma", "Vascular Lesions",
        ]
        print("class_names.txt not found -- using default list.")

    if not TF_AVAILABLE:
        print("TensorFlow unavailable -- DEMO MODE.")
        DEMO_MODE = True
        return

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Weights not found at {WEIGHTS_PATH} -- DEMO MODE.")
        DEMO_MODE = True
        return

    try:
        num_classes = len(CLASS_NAMES)

        # Raw pixels (0-255) -> preprocess_input (scales to [-1,1]) -> MobileNetV2 -> softmax
        inputs  = Input(shape=(224, 224, 3))
        x       = preprocess_input(inputs)
        base    = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
        x       = base(x, training=False)
        x       = GlobalAveragePooling2D()(x)
        x       = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation="softmax")(x)
        model   = Model(inputs, outputs)

        model.load_weights(WEIGHTS_PATH)
        DEMO_MODE = False
        print(f"Model loaded from {WEIGHTS_PATH}")
    except Exception as e:
        print(f"Model load failed: {e} -- DEMO MODE.")
        DEMO_MODE = True

# -- Skin validation ----------------------------------------------------------
def is_skin(img: Image.Image, min_frac: float = 0.12) -> tuple[bool, float]:
    """
    Returns (is_valid, skin_pixel_fraction).
    Checks light/medium/dark/pink/beige tone bands.
    Requires at least 12% of pixels to match any band.
    """
    try:
        arr = np.array(img.convert("RGB").resize((64, 64)), dtype=np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        light  = (r>120)&(g>60)&(b>40)&(r>=g)&(r>=b)&((r-g)>8)&((r-b)>12)
        medium = (r>80)&(g>50)&(b>30)&(r>=b)&((r-b)>8)&((r+g)>(b*2))
        dark   = (r>50)&(g>35)&(b>20)&(r>=b)&((r-b)>5)&((r+g)>(b*1.7))
        pink   = (r>130)&(g>70)&(b>60)&(r>g)&((r-g)>12)
        beige  = (r>155)&(g>115)&(b>75)&(r>b)&((r-b)>18)

        frac = float(np.mean(light | medium | dark | pink | beige))
        return frac >= min_frac, frac
    except Exception:
        return True, 0.5   # fail-open so a detection error never blocks a real scan

# -- Image preprocessing ------------------------------------------------------
def prepare(img_bytes: bytes) -> np.ndarray:
    """Resize to 224x224, return raw uint8->float32 array with batch dim."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)   # 0-255; model applies preprocess_input internally
    return np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)

# -- Endpoints ----------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "demo_mode": DEMO_MODE,
        "classes": CLASS_NAMES,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    import random

    img_bytes = await file.read()

    # 1. Parse image
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    # 2. Skin validation
    valid, skin_frac = is_skin(img)
    print(f"Skin fraction: {skin_frac*100:.1f}%")
    if not valid:
        raise HTTPException(
            400,
            f"This doesn't look like a skin image ({skin_frac*100:.1f}% skin pixels). "
            "Please upload a clear, close-up photo of the affected skin area."
        )

    # 3. Demo mode (TensorFlow or weights unavailable)
    if DEMO_MODE:
        opts = [
            ("Normal / Healthy Skin", "Normal"),
            ("Normal / Healthy Skin", "Normal"),
            ("Melanocytic Nevi",       "Low"),
            ("Benign Keratosis",       "Low"),
            ("Dermatofibroma",         "Low"),
        ]
        diagnosis, risk = random.choice(opts)
        conf = round(random.uniform(52.0, 88.0), 2)
        return {
            "diagnosis":    diagnosis,
            "confidence":   conf,
            "risk_level":   risk,
            "action_plan":  ACTION_MAP[risk],
            "demo_mode":    True,
            "skin_detected": round(skin_frac * 100, 1),
        }

    # 4. Real inference
    processed = prepare(img_bytes)
    preds     = model.predict(processed, verbose=0)[0]   # (num_classes,)

    top_idx = int(np.argmax(preds))
    conf    = float(preds[top_idx])
    print(f"Top: {CLASS_NAMES[top_idx]} @ {conf*100:.1f}%")
    print(f"All: { {CLASS_NAMES[i]: f'{preds[i]*100:.1f}%' for i in range(len(CLASS_NAMES))} }")

    # 5. Normal/Healthy -- only when model cannot confidently pick any disease
    if conf < NORMAL_THRESHOLD:
        print(f"Below {NORMAL_THRESHOLD*100}% threshold -> Normal/Healthy")
        return {
            "diagnosis":    "Normal / Healthy Skin",
            "confidence":   round(conf * 100, 2),
            "risk_level":   "Normal",
            "action_plan":  ACTION_MAP["Normal"],
            "demo_mode":    False,
            "skin_detected": round(skin_frac * 100, 1),
            "all_predictions": {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASS_NAMES))},
        }

    # 6. Disease result
    diagnosis = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else "Unknown"
    base_risk = RISK_MAP.get(diagnosis, "Low")

    if base_risk == "High" and conf < 0.70:
        risk = "Moderate"   # not confident enough to alarm the user
    elif base_risk == "Moderate" and conf < 0.50:
        risk = "Low"
    else:
        risk = base_risk

    print(f"Diagnosis: {diagnosis} | Risk: {risk} | Conf: {conf*100:.1f}%")

    return {
        "diagnosis":    diagnosis,
        "confidence":   round(conf * 100, 2),
        "risk_level":   risk,
        "action_plan":  ACTION_MAP[risk],
        "demo_mode":    False,
        "skin_detected": round(skin_frac * 100, 1),
        "all_predictions": {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASS_NAMES))},
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
