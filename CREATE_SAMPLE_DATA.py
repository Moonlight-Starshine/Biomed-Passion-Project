"""
Create dummy sample data for testing the pipeline.
This generates synthetic images to verify the setup works.
"""
import cv2
import numpy as np
from pathlib import Path

def create_synthetic_image(disease_type, image_id, size=224):
    """Create a synthetic histopathology image."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 240
    
    if disease_type == "malaria":
        # Red blood cells with malaria parasites
        for i in range(15):
            y = np.random.randint(20, size-20)
            x = np.random.randint(20, size-20)
            cv2.circle(img, (x, y), 8, (200, 50, 50), -1)  # RBC
            cv2.circle(img, (x-3, y-3), 3, (50, 50, 100), -1)  # Parasite
    
    elif disease_type == "healthy":
        # Normal red blood cells
        for i in range(20):
            y = np.random.randint(20, size-20)
            x = np.random.randint(20, size-20)
            cv2.circle(img, (x, y), 8, (200, 100, 100), -1)  # RBC
    
    elif disease_type == "tuberculosis":
        # Acid-fast bacilli (AFB)
        for i in range(10):
            y = np.random.randint(20, size-20)
            x = np.random.randint(20, size-20)
            cv2.line(img, (x, y), (x+15, y+5), (100, 50, 200), 2)  # Bacilli
    
    # Add some noise
    noise = np.random.randint(-20, 20, img.shape)
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    
    return img

if __name__ == "__main__":
    # Create directories and sample images
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    diseases = ["malaria", "healthy", "tuberculosis"]
    images_per_disease = 10
    
    for disease in diseases:
        disease_dir = data_dir / disease
        disease_dir.mkdir(exist_ok=True)
        
        for i in range(images_per_disease):
            img = create_synthetic_image(disease, i)
            output_path = disease_dir / f"{disease}_sample_{i:03d}.jpg"
            cv2.imwrite(str(output_path), img)
            print(f"Created: {output_path}")
    
    print(f"\n✓ Created {sum(len(list((data_dir / d).glob('*.jpg'))) for d in diseases)} sample images")
    print(f"Ready to train! Run:")
    print(f"  python src/train.py --data_dir data/raw --num_epochs 10")
