import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os

def extract_lbp_features(gray_patch, num_points=8, radius=1):
    """
    Extract LBP features from a grayscale patch.
    """
    lbp = local_binary_pattern(gray_patch, P=num_points, R=radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), density=True)
    return lbp_hist

def extract_patch_features(patch):
    """
    Extract mean HSV, mean Lab, and LBP features from a patch.
    """
    # Convert to HSV and Lab color spaces
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)
    
    # Compute mean HSV and Lab values
    hsv_mean = np.mean(hsv_patch, axis=(0, 1))
    lab_mean = np.mean(lab_patch, axis=(0, 1))
    
    # Convert to grayscale and compute LBP histogram
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    lbp_hist = extract_lbp_features(gray_patch)
    
    # Concatenate all features
    return np.concatenate((hsv_mean, lab_mean, lbp_hist))

def divide_image_into_patches(image, patch_size=(16, 8), stride=(8, 4)):
    """
    Divide an image into overlapping patches.
    """
    height, width = image.shape[:2]
    patches = []
    for y in range(0, height - patch_size[0] + 1, stride[0]):
        for x in range(0, width - patch_size[1] + 1, stride[1]):
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            patches.append(patch)
    return patches

def extract_global_features(image, patch_size=(16, 8), stride=(8, 4)):
    """
    Extract global features for an image by combining features from all patches.
    """
    patches = divide_image_into_patches(image, patch_size, stride)
    global_features = [extract_patch_features(patch) for patch in patches]
    return np.concatenate(global_features)

if __name__ == "__main__":
    # Load an example image
    image_path = "images"  # Adjust path if necessary
    for filename in os.listdir(image_path):
        new_path = os.path.join(image_path, filename)
        if not os.path.exists(new_path):
            print(f"Error: Image not found at {new_path}")
            exit(1)
        
        image = cv2.imread(new_path)
        if image is None:
            print(f"Error: Unable to load image at {new_path}")
            exit(1)

        # Extract global features
        global_features = extract_global_features(image)
        print("Global feature vector shape:", global_features.shape)

        # Save features to file
        output_dir = "output/feature_vectors/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "example_features.npy")
        np.save(output_path, global_features)
        print(f"Feature vector saved to {output_path}")
