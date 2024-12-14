import onnxruntime
import numpy as np
from PIL import Image


onnx_session = onnxruntime.InferenceSession("siamese_model_path")

def resize_image(image, size):
    """
    Resize the image using NumPy.
    
    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        size (tuple): Desired output size (height, width).
        
    Returns:
        numpy.ndarray: Resized image as a NumPy array.
    """
    pil_image = Image.fromarray(image)
    resized_pil_image = pil_image.resize(size[::-1], resample=Image.BILINEAR)
    resized_image = np.array(resized_pil_image)
    return resized_image

def to_tensor(image):
    """
    Convert the image to a PyTorch like tensor.
    
    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        
    Returns:
        numpy.ndarray: Image tensor.
    """
    return np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0

def get_image_similarity_onnx(image_path1, image_path2):
    """
    Compute the similarity between two images using the ONNX Siamese network model.
    
    Args:
        image_path1 (str): File path of the first image.
        image_path2 (str): File path of the second image.
        
    Returns:
        float: Similarity score between the two images (1.0 for identical, 0.0 for completely different).
    """
    # Load and preprocess the images
    # img1 = np.array(Image.open(image_path1))
    img1 = image_path1
    img1 = resize_image(img1, (100, 100))
    img1 = to_tensor(img1)
    img1 = np.expand_dims(img1, axis=0)

    
    # img2 = np.array(Image.open(image_path2))
    img2 = image_path2
    img2 = resize_image(img2, (100, 100))
    img2 = to_tensor(img2)
    img2 = np.expand_dims(img2, axis=0)
    
    
    outputs = onnx_session.run(None, {'input1': img1, 'input2': img2})
    output1, output2 = outputs
 
    euclidean_distance = np.linalg.norm(output1 - output2)
    
    similarity_score = 1.0 / (1.0 + np.exp(euclidean_distance))
    
    return euclidean_distance




"""
Test: siamease network 
"""

# image1_path = r'/app/test_images/15.png'
# image2_path = r'/app/test_images/787.png'

# similarity = get_image_similarity_onnx(image1_path, image2_path)
# similarity = similarity*100
# print(f'Similarity percentage: {similarity}%')