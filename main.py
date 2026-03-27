import os
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import gdown
import requests

app = Flask(__name__)
CORS(app)

# ================= GOOGLE DRIVE FILE IDs =================
MODELS_CONFIG = {
    "mobilenet_v3_large": {
        "file_id": "1kPeScw6q9ltu5p7U3F41OOrxJbyg12qN",
        "model_file": "best_mobilenet_v3_large.pth",
        "labels_file": "mobilenet_v3_large_levels.json",
        "labels_id": "1y7MJ_b3-JMSaR8xmEAKamjfP4U8r1CPJ",
        "builder": "mobilenet_v3_large"
    },
    "resnet18": {
        "file_id": "1hAHP7K6cuhS9S_nB1D_a9UeeRaMa3w5R",
        "model_file": "resnet18_best.pth",
        "labels_file": "resnet18_labels.json",
        "labels_id": "1UaaSCCsu4NsEvt9nWRW3DTGRahLdWykY",
        "builder": "resnet18"
    },
    "resnet50": {
        "file_id": "1LZy62YZo2qh6TgobJC1IaPkkm9rmWgcm",
        "model_file": "best_resnet50.pth",
        "labels_file": "resnet50_levels.json",
        "labels_id": "1J3zxBgKZ7x0VyN43a65xlKVYHem3Od7Z",
        "builder": "resnet50"
    },
    "resnet101": {
        "file_id": "1Q7gB2ST5haEeS_EH0g3Inz7FbyNftR7D",
        "model_file": "best_resnet101.pth",
        "labels_file": "resnet101_levels.json",
        "labels_id": "1SrWzGaHbIIXN0JtjaGbXzQsy7NO-jesa",
        "builder": "resnet101"
    },
    "efficientnet_v2_s": {
        "file_id": "1E5UhW_cdavKFG4PV7BIWlTWNaySOK9Tq",
        "model_file": "best_efficientnet_v2_s.pth",
        "labels_file": "efficientnet_v2_s_levels.json",
        "labels_id": "1lao8ofmEwBzrKcYLuiL3A-n_IjxRmlVW",
        "builder": "efficientnet_v2_s"
    },
    "efficientnet_v2_m": {
        "file_id": "14jo2nmcKFgfYi1kYw9krK7wa6QZzwGgE",
        "model_file": "best_efficientnet_v2_m.pth",
        "labels_file": "efficientnet_v2_m_levels.json",
        "labels_id": "1WYPqrP8zM_EbE4j8X2DB-TvslDeTnN6v",
        "builder": "efficientnet_v2_m"
    }
}

DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ================= MODEL BUILDERS =================
def build_model(model_type, num_classes):
    if model_type == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_type == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "resnet101":
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == "efficientnet_v2_m":
        model = models.efficientnet_v2_m(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

# ================= IMAGE PREPROCESSING =================
inference_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= MODEL MANAGER =================
class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.class_names = {}
    
    def download_file(self, file_id, output_path):
        if os.path.exists(output_path):
            print(f"✅ {output_path} already exists")
            return True
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
            print(f"✅ Downloaded {output_path}")
            return True
        except Exception as e:
            print(f"❌ Download error: {e}")
            try:
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                response = requests.get(download_url, stream=True)
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                print(f"✅ Downloaded {output_path} (alt method)")
                return True
            except Exception as e2:
                print(f"❌ Alternative failed: {e2}")
                return False
    
    def load_model(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.class_names[model_name]
        
        if model_name not in MODELS_CONFIG:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = MODELS_CONFIG[model_name]
        
        if not self.download_file(config["file_id"], config["model_file"]):
            raise Exception(f"Failed to download model: {model_name}")
        if not self.download_file(config["labels_id"], config["labels_file"]):
            raise Exception(f"Failed to download labels: {model_name}")
        
        with open(config["labels_file"], 'r') as f:
            classes = json.load(f)
        self.class_names[model_name] = classes
        
        num_classes = len(classes)
        model = build_model(config["builder"], num_classes)
        
        state_dict = torch.load(config["model_file"], map_location=DEVICE)
        if "model_state" in state_dict:
            model.load_state_dict(state_dict["model_state"])
        else:
            model.load_state_dict(state_dict)
        
        model.eval()
        model.to(DEVICE)
        
        self.loaded_models[model_name] = model
        print(f"✅ Loaded {model_name}: {num_classes} classes")
        
        return model, classes
    
    def predict(self, image_bytes, model_name):
        model, classes = self.load_model(model_name)
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
            
            all_probs = {
                classes[i]: round(probabilities[i].item(), 4)
                for i in range(len(classes))
            }
        
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "prediction": classes[predicted_idx],
            "class_index": predicted_idx,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "all_probabilities": sorted_probs,
            "model_used": model_name
        }

model_manager = ModelManager()

# ================= API ENDPOINTS =================

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "available_models": list(MODELS_CONFIG.keys()),
        "loaded_models": list(model_manager.loaded_models.keys())
    })

@app.route('/models')
def list_models():
    model_info = {
        "mobilenet_v3_large": {
            "name": "MobileNet V3 Large",
            "description": "Fastest - Best for Samsung M10/Redmi 11 Prime",
            "size_mb": 16.4,
            "speed": "Very Fast",
            "accuracy": "Good"
        },
        "resnet18": {
            "name": "ResNet 18",
            "description": "Lightweight - Good balance",
            "size_mb": 42.8,
            "speed": "Fast",
            "accuracy": "Good"
        },
        "resnet50": {
            "name": "ResNet 50",
            "description": "Standard - Higher accuracy",
            "size_mb": 90.2,
            "speed": "Medium",
            "accuracy": "Better"
        },
        "resnet101": {
            "name": "ResNet 101",
            "description": "Deep - Highest accuracy",
            "size_mb": 162.9,
            "speed": "Slow",
            "accuracy": "Best"
        },
        "efficientnet_v2_s": {
            "name": "EfficientNet V2 S",
            "description": "Efficient - Modern architecture",
            "size_mb": 78.0,
            "speed": "Medium",
            "accuracy": "Better"
        },
        "efficientnet_v2_m": {
            "name": "EfficientNet V2 M",
            "description": "Most accurate - Large model",
            "size_mb": 203.3,
            "speed": "Slow",
            "accuracy": "Best"
        }
    }
    return jsonify({
        "models": model_info,
        "default": "mobilenet_v3_large"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.form.get('model', 'mobilenet_v3_large')
        
        if model_name not in MODELS_CONFIG:
            return jsonify({
                "success": False,
                "error": f"Unknown model. Available: {list(MODELS_CONFIG.keys())}"
            }), 400
        
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400
        
        image_bytes = file.read()
        result = model_manager.predict(image_bytes, model_name)
        
        return jsonify({
            "success": True,
            **result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "available_models": list(MODELS_CONFIG.keys()),
        "loaded": list(model_manager.loaded_models.keys())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
