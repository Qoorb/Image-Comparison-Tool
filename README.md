# Image Comparison Tool

This project provides an image comparison tool that utilizes various methods to analyze the similarity between two images. The tool supports deep learning-based methods, feature matching techniques (SIFT and ORB), histogram comparison, and perceptual hashing.

# Features

- Deep Learning: Uses a Siamese Network to compute similarity.
- Feature Matching: Implements SIFT and ORB algorithms for keypoint matching.
- Histogram Comparison: Compares images based on their color histograms.
- Perceptual Hashing: Uses different hashing techniques to measure image similarity.

# Usage

To run the program, use the following command in your terminal:
```python 
python -m main --image1 <path_to_image1> --image2 <path_to_image2> [options]
```

## Arguments

- `--image1`: Path to the first image (required).
- `--image2`: Path to the second image (required).
- `--method`: Comparison method to use. Options include:
    - `deep`: Use deep learning for similarity comparison.
    - `hash`: Use perceptual hashing methods.
    - `sift`: Use SIFT feature matching.
    - `orb`: Use ORB feature matching.
    - `hist`: Use histogram comparison.
- `--model_path`: Path to the pre-trained model (for deep learning method). Optional.
- `--hash_size`: Hash size for perceptual hashing methods. Default is 8. Optional.
- `--no_visualization`: Disable results visualization. Optional.

# Example Commands

1. To compare two images using deep learning:
```python
   python -m main --image1 images/salmon.jpg --image2 images/new_year_salmon.jpg --method deep
```
2. To compare two images using SIFT:
```python
   python -m main --image1 images/salmon.jpg --image2 images/new_year_salmon.jpg --method sift
```
3. To compare two images using histogram comparison:
```python
   python -m main --image1 images/salmon.jpg --image2 images/new_year_salmon.jpg --method hist
```
4. To compare two images using perceptual hashing with a custom hash size:
```python
   python -m main --image1 images/salmon.jpg --image2 images/new_year_salmon.jpg --method hash --hash_size 16
```

# Output

The program will print the similarity scores for each method used. If visualization is enabled, it will display the results graphically, showing the images and their similarity scores.

# Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

# License

This project is licensed under the MIT License. See the LICENSE file for more details.
