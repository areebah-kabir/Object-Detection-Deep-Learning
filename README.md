# Object Detection Using Deep Learning with OpenCV and Python

[![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project demonstrates object detection using the OpenCV `dnn` module, which supports pre-trained deep learning models from frameworks like Caffe, Torch, and TensorFlow. It focuses on YOLO (You Only Look Once) for efficient real-time detection. Popular frameworks for object detection include YOLO, SSD, and Faster R-CNN. Recent updates to OpenCV's DNN module include native support for YOLO/DarkNet models.

This implementation processes input images or videos to detect and annotate objects, making it suitable for applications in computer vision, surveillance, or automation.

## Key Features

- Real-time object detection with YOLO v3.
- Support for image and video inputs.
- Bounding box visualization with class labels and confidence scores.
- Easy integration with OpenCV for custom pipelines.
- Extensible to other models (SSD and Faster R-CNN examples forthcoming).

## Tech Stack

| Tool     | Purpose                     | Version |
|----------|-----------------------------|---------|
| Python  | Core scripting              | 3.7+   |
| OpenCV  | DNN inference and rendering | 4.0+   |
| NumPy   | Array operations            | 1.18+  |

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/TahaMazhar01/Object-Detection-YOLO-OpenCV.git
   cd Object-Detection-YOLO-OpenCV
   ```

2. Install dependencies:
   ```
   pip install numpy opencv-python
   ```

**Note:** Compatibility with Python 2.x is not officially tested. Use Python 3.7 or later.

## Usage

### YOLO v3 Setup

Download the pre-trained YOLO v3 weights file:
```
wget https://pjreddie.com/media/files/yolov3.weights
```

Place the following files in the current directory:
- `yolov3.weights` (weights)
- `yolov3.cfg` (configuration)
- `yolov3.txt` (class names)

Run detection on an input image (e.g., `dog.jpg`):
```
python yolo_opencv.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
```

**General Command Format:**
```
python yolo_opencv.py --image /path/to/input/image --config /path/to/config/file --weights /path/to/weights/file --classes /path/to/classes/file
```

For video input, replace `--image` with `--video /path/to/input/video.mp4`.

For more details, refer to the [original blog post](http://www.arunponnusamy.com/yolo-object-detection-opencv-python.html).

### Sample Output

The output displays the input image with bounding boxes, class labels (e.g., "dog"), and confidence scores overlaid.

![Sample Detection Output](object-detection.jpg)

## How It Works

1. Load the DNN model using OpenCV's `cv2.dnn.readNetFromDarknet` for YOLO.
2. Prepare the input image: Resize to network dimensions and convert to blob.
3. Run forward inference to get bounding boxes, confidences, and class IDs.
4. Apply non-maximum suppression to filter overlapping detections.
5. Draw annotations on the output image using OpenCV drawing functions.

Example snippet from `yolo_opencv.py`:
```python
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)
# Process outputs for boxes, confidences, and classes
```

## Requirements

- Python 3.7+
- Listed dependencies:
  ```
  numpy>=1.18.0
  opencv-python>=4.0.0
  ```

Full `requirements.txt` available in the repository.

## Roadmap

- Add SSD (Single Shot Detector) implementation.
- Include Faster R-CNN examples.
- Support for video streaming (e.g., webcam or RTSP).
- Integration with additional libraries like cvlib for simplified detection: `detect_common_objects()`.

Feature requests can be submitted via issues.

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m "Description of changes"`.
4. Push the branch: `git push origin feature-name`.
5. Open a pull request.

Follow PEP 8 style guidelines. Include tests where applicable.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
Author: Areeba Kabir
GitHub: AreebaKabir
Email: areebakabir196@gmail.com

For questions or feedback, open an issue or email the author.
