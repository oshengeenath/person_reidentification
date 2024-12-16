# Person Re-Identification Implementations

This repository contains two implementations for Person Re-Identification in video streams using machine learning techniques. The first is an older method based on image-based similarity comparison, while the second is a more recent approach leveraging YOLOv8 and Siamese networks for multi-frame analysis.

## Implementations

### 1. Image-Based Re-Identification (Old Implementation)
In this implementation, we extract features from pairs of images and compare them to determine if they belong to the same person. The process involves:
- Pairwise comparison using a simple metric learning method (Mahalanobis Distance).

**Results:** <br />
The results for this implementation are stored as two example images showcasing person re-identification:
<br />
Test Images <br />
![Test Images](https://github.com/Thathsarani-Sandarekha/person_reidentification/blob/main/old_implementation/test_results/Test%20Images.jpg) <br />
<br />
Similarity Ranks <br />
![](https://github.com/Thathsarani-Sandarekha/person_reidentification/blob/main/old_implementation/test_results/Results.jpg)  

### 2. YOLOv8 + Siamese Network (New Implementation)
This newer method integrates object detection with deep learning for more robust re-identification:
- **YOLOv8** is used to detect and track people in video frames.
- A **Siamese network** is employed to compare feature vectors and compute similarity scores for re-identification.
- Multi-frame analysis and a dynamic reference management system handle occlusions and re-entries of individuals.

The results for this implementation are demonstrated in a video, which can be found at the following link:
[Watch the result video here](https://drive.google.com/drive/folders/1u0FZhDmirIm2bBIXIFwsSGikUFX6BHYI?usp=sharing)

## Requirements
To run the code for both implementations, install the following dependencies:

```bash
pip install -r requirements.txt
```

- PyTorch
- OpenCV
- YOLOv8

## Running the Code

### 1. Old Implementation (Image-Based)
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the `metric_learning.py` script:
   ```bash
   python metric_learning.py
   ```

### 2. New Implementation (YOLOv8 + Siamese Network)
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Request the model weights by contacting the repository maintainer.
3. Once you have the weights, open and run `inference.ipynb` to perform person re-identification on video streams.

## Conclusion
The new implementation using YOLOv8 and Siamese networks provides more robust and accurate results for person re-identification in video streams, particularly in dynamic and crowded environments. The video-based system also handles occlusions and re-entries, making it a significant improvement over the old image-based approach.

For detailed results, refer to the example images and the linked video.
