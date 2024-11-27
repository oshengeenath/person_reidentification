import cv2
import numpy as np
import os
from itertools import combinations
import torch


def load_images_and_extract_features(image_dir):
    """
    Load images from a directory and extract feature vectors.
    :param image_dir: Path to the directory containing images.
    :return: A list of extracted feature vectors and the list of corresponding image files.
    """
    from feature_extraction import extract_global_features  # Ensure feature_extraction.py is in the same directory
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg')]
    feature_vectors = []
    
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is not None:
            features = extract_global_features(image)
            feature_vectors.append(features)
        else:
            print(f"Warning: Could not load {image_path}")
    
    # Convert to NumPy array, padding or trimming feature vectors to ensure uniform shape
    max_length = max(len(f) for f in feature_vectors)
    padded_features = np.array([np.pad(f, (0, max_length - len(f))) for f in feature_vectors])
    
    return padded_features, image_files



def generate_pairs(num_images, split=6):
    """
    Generate "same" and "different" pairs based on a split index.
    :param num_images: Total number of images.
    :param split: Index where the "same" group ends, and "different" starts.
    :return: Two lists of tuples (same_pairs, diff_pairs).
    """
    same_pairs = list(combinations(range(split), 2))  # Combinations within the first group
    diff_pairs = [(i, j) for i in range(split) for j in range(split, num_images)]  # Across groups
    return same_pairs, diff_pairs

def compute_pairwise_differences(features, pairs):
    """
    Compute pairwise differences for given pairs of feature vectors.
    """
    return [features[i] - features[j] for i, j in pairs]

def compute_covariance_matrix(differences):
    """
    Compute the covariance matrix for a set of difference vectors.
    """
    differences = np.array(differences)
    return np.cov(differences, rowvar=False)

def learn_metric(features, same_pairs, diff_pairs):
    """
    Learn the Mahalanobis metric M.
    """
    same_differences = compute_pairwise_differences(features, same_pairs)
    diff_differences = compute_pairwise_differences(features, diff_pairs)
    Sigma_S = compute_covariance_matrix(same_differences)
    Sigma_D = compute_covariance_matrix(diff_differences)
    delta_sigma = Sigma_S - Sigma_D
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    delta_sigma_torch = torch.tensor(delta_sigma, device=device)  # Move to GPU
    eigenvalues, eigenvectors = torch.linalg.eigh(delta_sigma_torch)
    # Transfer tensor to CPU before using with NumPy
    eigenvalues = eigenvalues.cpu().numpy()
    eigenvectors = eigenvectors.cpu().numpy()
    M = eigenvectors @ np.diag(np.maximum(eigenvalues, 0)) @ eigenvectors.T
    return M

def mahalanobis_distance(x_i, x_j, M):
    """
    Compute the Mahalanobis distance between two feature vectors.
    """
    diff = x_i - x_j
    return np.sqrt(diff.T @ M @ diff)

def classify_test_image(test_feature, gallery_features, M):
    """
    Classify a test image by ranking gallery images based on Mahalanobis distance.
    :param test_feature: Feature vector of the test image.
    :param gallery_features: Feature vectors of gallery images.
    :param M: Learned Mahalanobis metric matrix.
    :return: Sorted list of distances and corresponding gallery indices.
    """
    distances = []
    for idx, gallery_feature in enumerate(gallery_features):
        dist = mahalanobis_distance(test_feature, gallery_feature, M)
        distances.append((dist, idx))
    
    # Sort distances in ascending order
    distances.sort(key=lambda x: x[0])
    return distances


if __name__ == "__main__":
    # Paths
    image_dir = "images"  # Directory containing the .jpeg images
    
    # Load images and extract features
    features, image_files = load_images_and_extract_features(image_dir)
    print(f"Loaded {len(features)} feature vectors.")
    
    # Generate pairs
    same_pairs, diff_pairs = generate_pairs(num_images=len(features), split=6)
    print(f"Generated {len(same_pairs)} same pairs and {len(diff_pairs)} different pairs.")
    
    # Learn metric
    M = learn_metric(features, same_pairs, diff_pairs)
    print("Learned metric matrix M shape:", M.shape)
    
    # Classify multiple test images
    num_test_images = 5  # Specify how many test images to evaluate
    for test_idx in range(num_test_images):
        # Select a test image and treat the rest as the gallery
        test_feature = features[test_idx]
        gallery_features = np.delete(features, test_idx, axis=0)  # Exclude the test image from the gallery
        gallery_image_files = [img for idx, img in enumerate(image_files) if idx != test_idx]
        
        # Classify the test image
        ranked_results = classify_test_image(test_feature, gallery_features, M)
        
        # Print results for the current test image
        print(f"\nTest image: {image_files[test_idx]}")
        print("Ranked gallery matches (distance, index):")
        for rank, (dist, idx) in enumerate(ranked_results[:5], start=1):  # Top 5 matches
            print(f"Rank {rank}: Distance = {dist:.4f}, Gallery Image = {gallery_image_files[idx]}")
