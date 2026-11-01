# SIGN_READER__Real-Time-Sign_Language-Recognition
SIGNREADER_Real-Time-Sign-Language-Recognition Real-Time Sign Language Recognition:  To remove communication barrier between def ,mute, people with others . its basically made for other people whom can listen and talk , to understand them .


## file structure to detect sentence from recognizing two hand gesture
SignReader/

│
├── README.md

├── requirements.txt

├── .gitignore

├── LICENSE
│

├── src/

│   ├── sign_reader_sentence.py

│   ├── test_common_words_2hand_webcam.py

│   ├── extract_word_landmarks_2hand.py

│   └── train_common_words_2hand.py

│

├── model/

│   ├── common_words_2hand_model.pkl

│   └── common_words_2hand_labels.pkl

│

├── dataset/

│   └── README.md

│

├── screenshots/

│   ├── webcam_detection.png
     <img width="925" height="462" alt="Screenshot 2026-10-31 220500" src="https://github.com/user-attachments/assets/1b8153e6-b0dd-4d8a-8cdc-2e4c658bd766" />



│   └── sentence_output.png

│
└── docs/

    └── project_report.pdf



## Overview

**SignReader** is a computer-vision and machine-learning project designed to recognize sign-language gestures using a webcam and convert recognized gestures into text.

The system uses **Python, OpenCV, and MediaPipe** to detect hands and extract hand-landmark coordinates. These features are processed by a trained machine-learning model to classify gestures into predefined word categories.

The project supports both single-hand recognition and two-hand gesture recognition. Its two-hand recognition system can combine consecutively recognized words into a text-based sentence, helping users communicate through recognized gestures.

SignReader aims to explore how computer vision and machine learning can contribute to more accessible communication for people who use sign language.


## Features

- **Real-Time Hand Detection:** Detects hands through a webcam using MediaPipe.
- **Hand Landmark Extraction:** Extracts 3D coordinates of hand landmarks to represent gestures numerically.
- **Sign Gesture Recognition:** Uses a trained machine-learning model to classify gestures into supported word categories.
- **Single-Hand Recognition:** Supports recognition of gestures performed using one hand.
- **Two-Hand Recognition:** Supports recognition of gestures that require both hands.
- **Sentence Construction:** Combines consecutively recognized words into a text-based sentence.
- **Confidence-Based Prediction:** Uses prediction confidence to help filter uncertain gesture classifications.
- **Live Visual Feedback:** Displays detected hands, predicted words, and confidence scores on the webcam feed.
- **Sentence Controls:** Allows users to clear the sentence or undo the last captured word using keyboard controls.


## Installation and Setup

Follow these steps to set up and run SignReader on your local machine.

### Prerequisites

- Python installed on your system
- A working webcam
- Git (optional, for cloning the repository)

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd SignReader
```

Replace `YOUR_REPOSITORY_URL` with your GitHub repository URL.

Alternatively, download the repository as a ZIP file and extract it.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Verify the Project Files

Ensure the required Python scripts and trained model files are present in their documented locations.

The two-hand recognition script requires:

- `model/common_words_2hand_model.pkl`
- `model/common_words_2hand_labels.pkl`

### 6. Run the Application

Run the appropriate Python script for the recognition mode you want to use.

**Two-hand recognition with sentence construction:**

```bash
python -u test_common_words_2hand_webcam.py
```

**Note:** Adjust the script path if you place your Python files inside a `src/` directory.

### Troubleshooting

- **Webcam does not open:** Check that the webcam is connected and not being used by another application.
- **Module not found:** Activate the virtual environment and install the dependencies.
- **Model file not found:** Check that the trained model files exist in the expected directory.
- **Incorrect predictions:** Ensure adequate lighting and perform gestures clearly within the webcam frame.


## Usage

SignReader uses a webcam to detect hand gestures, recognize supported words, and display the results as text.

### 1. Start the Application

Activate your virtual environment and run the two-hand recognition script:

```bash
python -u test_common_words_2hand_webcam.py
```

### 2. Perform Sign Gestures

- Position your hands clearly within the webcam frame.
- Perform a gesture supported by the trained model.
- Hold the gesture steady until the predicted word is captured.
- Change to another supported gesture to add another word to the sentence.

### 3. View the Results

The application displays:
- Number of detected hands
- Predicted word
- Prediction confidence
- Accumulated sentence

Captured words are also printed in the terminal.

### 4. Keyboard Controls

| Key | Function |
|---|---|
| `C` | Clear the current sentence |
| `U` | Undo the last captured word |
| `Q` | Quit the application |

### Notes

- The two-hand model recognizes only its trained word categories.
- Recognition performance depends on lighting, hand visibility, and gesture execution.
- Sentence construction combines recognized words in sequence; it does not automatically correct grammar or translate every sign-language sentence.


## Dataset and Model Information

### Dataset

SignReader uses hand-landmark data to train its gesture-recognition model.

The two-hand dataset contains numerical features extracted from hand landmarks. Each hand is represented by 21 landmarks, with three coordinates (X, Y, and Z) per landmark.

- **Landmarks per hand:** 21
- **Coordinates per landmark:** 3
- **Features per hand:** 63
- **Total features for two hands:** 126

When a hand is not detected, its feature vector is represented by zeros.

### Feature Extraction

MediaPipe Hands detects hand landmarks from images. The coordinates are converted into numerical feature vectors relative to the wrist position. The left- and right-hand feature vectors are then combined to create the input for the model.

### Model Training

The extracted features and their corresponding labels are used to train a machine-learning classifier.

The trained model predicts a word category from the detected hand landmarks. The webcam application uses prediction confidence to help filter uncertain results.

### Supported Vocabulary

The current two-hand model supports **19 word categories**.

Its recognition capabilities are limited to the categories represented in the trained model.

### Model Files

The trained model and label files are:

- `model/common_words_2hand_model.pkl`
- `model/common_words_2hand_labels.pkl`

These files allow the recognition application to load the trained model without training it again.

### Limitations

- Recognition is limited to the model's trained vocabulary.
- Prediction accuracy depends on the quality and consistency of the training data.
- A detected gesture may be classified incorrectly.
- Sentence construction combines recognized words in sequence; it does not independently interpret sign-language grammar.


![Uploading Screenshot 2026-10-31 220500.png…]()



<img width="835" height="392" alt="Screenshot 2026-11-01 114248" src="https://github.com/user-attachments/assets/52408106-92cb-4c0a-ab6c-b8b357dda260" />


<img width="287" height="452" alt="Screenshot 2026-11-01 114439" src="https://github.com/user-attachments/assets/77c2e372-7f71-4355-8c2b-7f002ee09b1d" />


## Results and Limitations

### Results

- detects hands in real time using a webcam.
- Extracts hand-landmark coordinates using MediaPipe.
- Classifies detected gestures into supported word categories using a trained machine-learning model.
- Displays predicted words and their confidence scores.
- Combines consecutively recognized words into a text-based sentence.
- Provides keyboard controls to clear the sentence, undo the last word, and exit the application.

### Limitations

- The current two-hand model supports only 19 trained word categories.
- Recognition performance depends on lighting, hand visibility, camera quality, and gesture execution.
- Some gestures may be classified incorrectly.
- Low-confidence predictions may not be captured.
- Sentence construction combines recognized words but does not automatically generate grammatically correct sentences.
- The system has not been demonstrated to recognize every sign in a complete sign language.

### Future Improvements

- Expand the supported vocabulary with additional training data.
- Improve recognition accuracy and stability.
- Add text-to-speech output for recognized words and sentences.
- Develop better sentence formation and grammar processing.
- Create a user-friendly graphical interface.
- Evaluate the model using a separate test dataset and report measured performance metrics.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Captures webcam video and displays real-time recognition results |
| MediaPipe Hands | Detects hands and extracts hand landmarks |
| NumPy | Processes landmark coordinates and creates feature arrays |
| scikit-learn | Provides machine-learning functionality for gesture classification |
| Joblib | Saves and loads the trained model and related files |

### Development Tools

- Visual Studio Code
- Python virtual environment (`venv`)
- Git and GitHub for version control and project hosting


## License

This project is intended for educational and portfolio purposes.

A formal license has not yet been selected. Until a license is added, others should not assume they have permission to reuse, modify, or redistribute the project's code.

## Acknowledgements

- **MediaPipe:** For hand landmark detection.
- **OpenCV:** For webcam access and real-time image processing.
- **NumPy:** For numerical operations and feature preparation.
- **scikit-learn:** For machine-learning model development and prediction.
- **Python:** For providing the programming environment used to develop SignReader.

