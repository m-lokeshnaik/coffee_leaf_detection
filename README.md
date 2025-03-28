# Coffee Leaf Disease Detection

A deep learning-based web application that detects diseases in coffee plant leaves using computer vision and machine learning techniques.

## Overview

This project implements a disease detection system for coffee plant leaves using a VGG16-based deep learning model. The system can classify coffee leaf images into four categories:
- Healthy (No Disease)
- Phoma
- Rust
- Leaf Miner

## Features

- Real-time disease detection from uploaded images
- User-friendly web interface built with Streamlit
- High-accuracy predictions using VGG16 architecture
- Support for common image formats (PNG, JPG)
- Confidence score for predictions

## Requirements

- Python 3.6+
- TensorFlow 2.x
- Streamlit
- PIL (Python Imaging Library)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lokeshdangii/coffee_leaf_detection.git
cd coffee_leaf_detection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the displayed local URL (typically http://localhost:8501)

3. Upload a coffee leaf image using the file uploader

4. View the prediction results and confidence scores

## Project Structure

- `app.py`: Main Streamlit web application for handling user interface and image processing
- `utils.py`: Helper functions for image preprocessing, model predictions, and result formatting
- `requirements.txt`: Comprehensive list of Python dependencies with specific versions
- `model_20.h5`: Pre-trained VGG16-based model weights for disease detection (download separately)

## Model Architecture

The system uses a VGG16-based model with the following architecture:
- Pre-trained VGG16 base (trained on ImageNet) with input shape (256, 256, 3)
- Global Average Pooling layer for feature extraction
- Dense output layer (4 units) with softmax activation for multi-class classification:
  - Class 0: Healthy leaves
  - Class 1: Phoma infection
  - Class 2: Coffee rust disease
  - Class 3: Leaf miner infestation

## Input Requirements

- Supported image formats: PNG, JPG
- Images are automatically resized to 512x512 pixels
- RGB images (3 channels)

## License

MIT License

Copyright (c) 2024 Coffee Leaf Disease Detection

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contributors

This project is maintained by Lokesh Dangi. We welcome contributions from the community! If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

Please ensure your code follows these standards:
- Use consistent code formatting (PEP 8 for Python)
- Add appropriate comments and documentation
- Include error handling for robustness
- Write unit tests for new features
- Update requirements.txt if adding new dependencies

## Acknowledgments

- VGG16 architecture from TensorFlow/Keras for transfer learning
- Streamlit framework for creating the interactive web interface
- The coffee farming community for providing domain expertise and test data
- All contributors who have helped improve this project's accuracy and usability

## Contact

For questions, suggestions, or collaboration opportunities:

- GitHub Issues: Please use the [issue tracker](https://github.com/m-lokeshnaik/coffee_leaf_detection/issues)
- Email: lokeshdangi.ece20@gmail.com
- Project Website: [Coffee Leaf Disease Detection](https://github.com/m-lokeshnaik/coffee_leaf_detection)

For bug reports, please include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details:
  - Python version
  - TensorFlow version
  - Operating system
  - Input image details (format, size)
