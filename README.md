Overview

This repository contains implementations for Q1 and Q2 of the image segmentation project:

Q1: Standard object detection and segmentation using pre-trained models.

Q2: Text-driven image segmentation leveraging SAM 2 (Segment Anything Model 2) with bounding box proposals from a grounding model.

Both notebooks are designed for experimentation, reproducibility, and clear visualization of segmentation outputs.

Environment Setup

The notebooks are designed to run in Google Colab or a local Python 3.12+ environment with GPU support. Recommended dependencies:
pip install torch torchvision torchaudio
pip install opencv-python-headless pillow matplotlib
pip install transformers timm einops ftfy
pip install git+https://github.com/facebookresearch/segment-anything.git

Tip: Ensure your GPU is available (cuda) for faster inference.

Q1 – Object Detection & Segmentation

Purpose: Detect and segment objects in a given image using pre-trained models.

Workflow:

Load the input image.

Use a pre-trained segmentation model (e.g., SAM or similar) for mask generation.

Visualize the segmented regions with bounding boxes or overlay masks.

Inputs:

Single image file (JPEG/PNG).

Optional: class labels for filtering.

Outputs:

Image with detected object masks.

Bounding box coordinates for detected objects.

Usage Example:

from q1 import segment_image

image_path = "images/sample.jpg"
masks, boxes = segment_image(image_path)
segment_image.display(masks, boxes)

Q2 – Text-Driven Image Segmentation (SAM 2)

Purpose: Segment objects in an image based on a text prompt, combining natural language understanding with image segmentation.

Workflow:

1. Load the input image.

2. Accept a text prompt describing the target object (e.g., "dog").

3. Use a grounding model (e.g., GroundingDINO or GLIP) to generate bounding box proposals.

4. Feed proposals into SAM 2 to generate precise masks.

5. Display final segmentation results over the original image.

Inputs:

Image file (JPEG/PNG).

Text prompt describing the object of interest.

Outputs:

Segmentation mask(s) corresponding to the text prompt.

Visual overlay on the original image.

Usage Example:

from q2 import text_segmentation

image_path = "images/sample.jpg"
prompt = "dog"
masks = text_segmentation.segment_with_text(image_path, prompt)
text_segmentation.display_masks(image_path, masks)

Notes & Best Practices

Ensure images are clearly visible and well-lit for accurate segmentation.

The text-driven approach in Q2 relies heavily on prompt quality; be specific.

GPU acceleration is highly recommended for SAM 2 inference.

Both notebooks support saving outputs to disk for further evaluation.



References:


https://github.com/facebookresearch/segment-anything


https://github.com/IDEA-Research/GroundingDINO
