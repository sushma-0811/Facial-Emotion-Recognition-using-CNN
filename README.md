
# Facial Emotion Recognition using Convolutional Neural Network (CNN)



## Overview

This project is a facial emotion recognition system that uses a Convolutional Neural Network (CNN) to classify facial expressions into seven different emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. It is built using TensorFlow and Keras.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python (>=3.6)
- TensorFlow (>=2.0)
- Keras
- PIL (Pillow)
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook (optional for experimenting)

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/facial-emotion-recognition.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd facial-emotion-recognition
   ```

3. **Install the required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **[Download the FER2013 dataset](https://www.kaggle.com/deadskull7/fer2013) and place it in the `data` directory.**

## Dataset

The project uses the FER2013 dataset, which contains facial images labeled with seven different emotions. The dataset is divided into training and testing sets.

## Training the Model

To train the CNN model, run the following command:

```bash
python train.py
```

This will train the model using the training data and save the trained model weights.

## Making Predictions

To make predictions on a new image, use the following command:

```bash
python predict.py path/to/your/image.jpg
```

This will load the trained model and display the predicted emotion for the input image.

## Results

The model achieved an accuracy of XX% on the test dataset. You can find more details about the model's performance in the `results` directory.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the project.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push your changes to your fork.
5. Open a pull request.



