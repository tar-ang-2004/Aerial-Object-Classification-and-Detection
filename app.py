from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import json
from pathlib import Path
import time
import traceback
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'webm'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Configure basic logging
log_path = os.path.join(app.config['RESULTS_FOLDER'], 'app.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Custom CNN architecture
class CustomCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        
        # Conv Block 1
        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(0.25)
        )
        
        # Conv Block 2
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(0.25)
        )
        
        # Conv Block 3
        self.conv_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(0.25)
        )
        
        # Conv Block 4
        self.conv_block4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(0.25)
        )
        
        # Calculate the size of flattened features
        # After 4 max pooling operations: 224 -> 112 -> 56 -> 28 -> 14
        self.feature_size = 256 * 14 * 14
        
        # Classifier (matching notebook architecture exactly)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.feature_size, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Model loader
class ModelLoader:
    def __init__(self):
        self.models = {}
        self.class_names = ['Bird', 'Drone']
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.load_models()
    
    def load_models(self):
        """Load all available trained models"""
        model_configs = {
            'custom_cnn': (CustomCNN(num_classes=2), 'best_custom_cnn.pth'),
            'resnet50': (models.resnet50(weights=None), 'best_resnet50.pth'),
            'vgg16': (models.vgg16(weights=None), 'best_vgg16.pth')
        }
        
        # Modify ResNet50 output layer to match notebook architecture
        num_features = model_configs['resnet50'][0].fc.in_features
        model_configs['resnet50'][0].fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 2)
        )
        
        # Modify VGG16 output layer
        model_configs['vgg16'][0].classifier[6] = torch.nn.Linear(4096, 2)
        
        for model_name, (model, weight_path) in model_configs.items():
            if os.path.exists(weight_path):
                try:
                    model.load_state_dict(torch.load(weight_path, map_location=device))
                    model.eval()
                    model = model.to(device)
                    self.models[model_name] = model
                    print(f"✅ Loaded {model_name}")
                except Exception as e:
                    print(f"⚠️ Failed to load {model_name}: {e}")
            else:
                print(f"⚠️ Model weights not found: {weight_path}")
        
        if not self.models:
            print("❌ No models loaded! Please ensure model weights are available.")
    
    def predict_image(self, image_path):
        """Predict on a single image"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(device)
        
        predictions = {}
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predictions[model_name] = {
                    'class': self.class_names[predicted.item()],
                    'confidence': float(confidence.item()),
                    'probabilities': {
                        self.class_names[i]: float(prob)
                        for i, prob in enumerate(probabilities[0].cpu().numpy())
                    }
                }
        
        # Ensemble prediction
        if predictions:
            pred_classes = [pred['class'] for pred in predictions.values()]
            ensemble_class = max(set(pred_classes), key=pred_classes.count)
            ensemble_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
            
            return {
                'models': predictions,
                'ensemble': {
                    'class': ensemble_class,
                    'confidence': float(ensemble_confidence),
                    'agreement': pred_classes.count(ensemble_class) / len(pred_classes)
                }
            }
        
        return None

# Initialize model loader
model_loader = ModelLoader()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_gif(gif_path):
    """Process GIF by analyzing each frame"""
    gif = Image.open(gif_path)
    frame_predictions = []
    
    try:
        frame_count = 0
        while True:
            gif.seek(frame_count)
            frame = gif.convert('RGB')
            
            # Save frame temporarily
            temp_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_frame_{frame_count}.jpg')
            frame.save(temp_frame_path)
            
            # Predict on frame
            prediction = model_loader.predict_image(temp_frame_path)
            if prediction:
                frame_predictions.append(prediction)
            
            os.remove(temp_frame_path)
            frame_count += 1
            
            if frame_count >= 30:  # Limit to 30 frames for performance
                break
    except EOFError:
        pass
    
    if frame_predictions:
        # Aggregate predictions
        bird_count = sum(1 for pred in frame_predictions if pred['ensemble']['class'] == 'Bird')
        drone_count = len(frame_predictions) - bird_count
        
        final_class = 'Bird' if bird_count > drone_count else 'Drone'
        confidence = max(bird_count, drone_count) / len(frame_predictions)
        
        # Aggregate model predictions from all frames
        model_aggregates = {}
        for pred in frame_predictions:
            for model_name, model_pred in pred['models'].items():
                if model_name not in model_aggregates:
                    model_aggregates[model_name] = {'Bird': 0, 'Drone': 0}
                model_aggregates[model_name][model_pred['class']] += 1
        
        # Convert counts to final predictions
        models_result = {}
        for model_name, counts in model_aggregates.items():
            pred_class = 'Bird' if counts['Bird'] > counts['Drone'] else 'Drone'
            conf = max(counts['Bird'], counts['Drone']) / len(frame_predictions)
            models_result[model_name] = {
                'class': pred_class,
                'confidence': conf
            }
        
        return {
            'ensemble': {
                'class': final_class,
                'confidence': confidence,
                'frames_analyzed': len(frame_predictions),
                'bird_frames': bird_count,
                'drone_frames': drone_count
            },
            'models': models_result
        }
    
    return None

def process_video(video_path):
    """Process video by analyzing frames"""
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    
    frame_count = 0
    sample_rate = 10  # Analyze every 10th frame
    
    while cap.isOpened() and frame_count < 100:  # Limit to 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Save frame temporarily
            temp_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_video_frame_{frame_count}.jpg')
            frame_pil.save(temp_frame_path)
            
            # Predict on frame
            prediction = model_loader.predict_image(temp_frame_path)
            if prediction:
                frame_predictions.append(prediction)
            
            os.remove(temp_frame_path)
        
        frame_count += 1
    
    cap.release()
    
    if frame_predictions:
        # Aggregate predictions
        bird_count = sum(1 for pred in frame_predictions if pred['ensemble']['class'] == 'Bird')
        drone_count = len(frame_predictions) - bird_count
        
        final_class = 'Bird' if bird_count > drone_count else 'Drone'
        confidence = max(bird_count, drone_count) / len(frame_predictions)
        
        # Aggregate model predictions from all frames
        model_aggregates = {}
        for pred in frame_predictions:
            for model_name, model_pred in pred['models'].items():
                if model_name not in model_aggregates:
                    model_aggregates[model_name] = {'Bird': 0, 'Drone': 0}
                model_aggregates[model_name][model_pred['class']] += 1
        
        # Convert counts to final predictions
        models_result = {}
        for model_name, counts in model_aggregates.items():
            pred_class = 'Bird' if counts['Bird'] > counts['Drone'] else 'Drone'
            conf = max(counts['Bird'], counts['Drone']) / len(frame_predictions)
            models_result[model_name] = {
                'class': pred_class,
                'confidence': conf
            }
        
        return {
            'ensemble': {
                'class': final_class,
                'confidence': confidence,
                'frames_analyzed': len(frame_predictions),
                'bird_frames': bird_count,
                'drone_frames': drone_count
            },
            'models': models_result
        }
    
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            file_ext = filename.rsplit('.', 1)[1].lower()
            logging.info(f"Received file {filename} (ext={file_ext})")

            if file_ext == 'gif':
                result = process_gif(filepath)
            elif file_ext in ['mp4', 'avi', 'mov', 'webm']:
                result = process_video(filepath)
            else:
                result = model_loader.predict_image(filepath)

            if result:
                logging.info(f"Prediction successful for {filename}")
                return jsonify({
                    'success': True,
                    'prediction': result,
                    'filename': filename,
                    'file_type': file_ext
                })
            else:
                logging.error(f"Prediction returned no result for {filename}")
                return jsonify({'error': 'Prediction failed'}), 500

        except Exception as e:
            # Log full traceback to file and console
            tb = traceback.format_exc()
            logging.error(f"Error during prediction for {filename}: {e}\n{tb}")
            return jsonify({'error': 'Internal server error', 'details': str(e)}), 500
        finally:
            # Attempt to remove uploaded file to save disk space
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                logging.warning(f"Failed to remove temp file: {filepath}")
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(model_loader.models),
        'device': str(device)
    })

if __name__ == '__main__':
    print(f"🚀 Starting Flask App on {device}")
    print(f"📊 Models loaded: {len(model_loader.models)}")
    app.run(debug=True, host='0.0.0.0', port=5000)
