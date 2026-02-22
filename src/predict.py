"""
Inference script for disease prediction on new images.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import json
from model import get_model
from data import HistopathologyDataset
from torchvision import transforms


def load_checkpoint(model_path, config_path, device):
    """Load trained model and config."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    num_classes = config['num_classes']
    architecture = config['architecture']
    class_names = config['class_names']
    
    model = get_model(num_classes, architecture=architecture, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, config


def predict_image(image_path, model, class_names, device, image_size=224, threshold=None):
    """
    Predict disease class for a single image.
    
    Args:
        image_path: Path to image file
        model: Trained model
        class_names: List of class names
        device: Device (cpu or cuda)
        image_size: Image size for resizing
        threshold: Optional confidence threshold for prediction
    
    Returns:
        predicted_class, confidence, all_probabilities
    """
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = HistopathologyDataset._stain_normalize(image)
    image = cv2.resize(image, (image_size, image_size))
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probs.max(1)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence = confidence.item()
    all_probs = {
        class_names[i]: probs[0, i].item()
        for i in range(len(class_names))
    }
    
    if threshold and confidence < threshold:
        predicted_class = "UNCERTAIN"
    
    return predicted_class, confidence, all_probs


def batch_predict(image_dir, model, class_names, device, image_size=224):
    """
    Predict on all images in a directory.
    
    Args:
        image_dir: Directory containing images
        model: Trained model
        class_names: List of class names
        device: Device
        image_size: Image size
    
    Returns:
        List of (image_path, prediction, confidence)
    """
    image_dir = Path(image_dir)
    results = []
    
    for image_path in image_dir.glob('*'):
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        try:
            predicted_class, confidence, _ = predict_image(
                image_path, model, class_names, device, image_size
            )
            results.append({
                'image': str(image_path),
                'prediction': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict disease class for images")
    parser.add_argument("image_path", type=str, help="Path to image or directory")
    parser.add_argument("--model", type=str, default="models/resnet50_best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="models/config.json",
                        help="Path to config file")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu, cuda, or auto)")
    parser.add_argument("--batch", action="store_true",
                        help="Process directory of images")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_checkpoint(args.model, args.config, device)
    class_names = config['class_names']
    
    # Predict
    if args.batch:
        results = batch_predict(args.image_path, model, class_names, device)
        for result in results:
            print(f"{result['image']}: {result['prediction']} ({result['confidence']:.2%})")
    else:
        predicted_class, confidence, all_probs = predict_image(
            args.image_path, model, class_names, device
        )
        print(f"Predicted: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print("\nAll probabilities:")
        for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {prob:.4f}")
