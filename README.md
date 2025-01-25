# Speech Emotion Recognition Using the TESS Dataset

![Speech_Emotion_Recognition_Poster](https://github.com/user-attachments/assets/a5ce24f0-53bd-45a3-a022-f1c04b792a0a)

This project is a Speech Emotion Recognition (SER) system that classifies emotions in speech audio files using the Toronto Emotional Speech Set (TESS) dataset. It employs audio preprocessing techniques, MFCC feature extraction, and machine learning algorithms to predict emotions such as Angry, Happy, Sad, and more.

---

## 🚀 Project Overview
Speech Emotion Recognition (SER) is a crucial task in understanding human emotions from audio signals, with applications in customer service, healthcare, and AI-driven assistants. This project builds a pipeline that:
1. Processes audio data.
2. Extracts meaningful features (MFCCs).
3. Trains machine learning models to classify emotions.

---

## 📂 Dataset
- **Dataset Name**: Toronto Emotional Speech Set (TESS)
- **Description**: ~2800 clean WAV files labeled with 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, and Sad.
- **Actors**: Two female speakers aged 26 and 64.
- **Format**: WAV files recorded at a sampling rate of 24,000 Hz.
- **Link**: [TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

---

## 🌳 Project Structure
      Speech_Emotion_Recognition/
      ├── Toronto Emotional Speech Set/            # Contains audio files for training and validation.
      ├── Unused_Data/                             # Contains audio files separated for testing only.
      ├── Speech_Emotion_Recognition.ipynb         # Main notebook
      └── README.md                                # Project documentation

---

## 🛠️ Project Workflow
1. **Preprocessing**:
   - Extract **MFCC (Mel-Frequency Cepstral Coefficients)** features.
   - Normalize and balance the dataset.
   - Split into training and testing sets.
2. **Model Training**:
   - Train multiple machine learning classifiers (e.g., Logistic Regression, SVM, KNN, Random Forest).
   - Perform hyperparameter tuning for optimal performance.
3. **Evaluation**:
   - Compare models using accuracy, precision, recall, and F1-score.
   - Identify the best-performing model.
4. **Generalization**:
   - Test the selected model on unseen data.

---

## ⚙️ Features Extracted
- **MFCC**: Captures frequency and time domain characteristics of speech.
- **Chroma Features**: Represent pitch class information.
- **Zero-Crossing Rate**: Measures frequency of signal sign changes.

---

## 📊 Model Comparison
The following machine learning classifiers were evaluated:
- Logistic Regression
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest

| Model                 | Accuracy | Precision (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) |
|-----------------------|----------|------------------------|---------------------|-----------------------|
| Logistic Regression   | 0.85     | 0.86                   | 0.84                | 0.85                  |
| **SVM**               | **0.93** | **0.92**               | **0.93**            | **0.92**              |
| KNN                   | 0.88     | 0.87                   | 0.89                | 0.88                  |
| Random Forest         | 0.90     | 0.89                   | 0.91                | 0.90                  |

---

## 🔧 Tools and Libraries
- **Python**
- **Librosa**: Audio feature extraction.
- **Scikit-learn**: Machine learning and evaluation.
- **Matplotlib**: Visualization.
- **Pandas & NumPy**: Data manipulation and analysis.

---

## 🔍 Key Findings
- SVM achieved the highest accuracy (93%) on the test set, making it the best-performing model.
- Emotions like Fear and Anger showed overlaps due to similar acoustic features.
- MFCCs proved to be highly effective features for emotion classification.

---

## 🎯 Output Metrics
1. Logistic Regression
   ![image](https://github.com/user-attachments/assets/0e850bfb-e841-494b-a8e7-af7dd5fb852d)
   
3. Support Vector Machine
   ![image](https://github.com/user-attachments/assets/bb5d01c3-d3c2-42f9-8105-22c78c1bed81)

4. K-Nearest Neighbor
   ![image](https://github.com/user-attachments/assets/a26145ce-4895-4575-a62d-324399b3292e)

5. Random Forest Classifier
   ![image](https://github.com/user-attachments/assets/aa7cb924-e589-4ebd-8a60-633d72df4ea1)

---

## 🏷️ Applications
- **Call Centers**: Analyze customer sentiment during calls.
- **Healthcare**: Monitor mental health through emotional analysis.
- **AI Assistants**: Enhance responses based on emotional states.
- **Education**: Measure student engagement via emotional feedback.

---

## 📌 Future Improvements
- Incorporate deep learning models (e.g., CNN, RNN) for improved accuracy.
- Use a more diverse dataset with varied speakers and accents.
- Test the system in real-world noisy environments.

---

## 📥 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/speech-emotion-recognition.git
   ```
2. Navigate to the project directory:
   ```bash
   cd speech-emotion-recognition
   ```
3. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```

## ▶️ Usage
1. Run the notebook:
   ```bash
    jupyter notebook Speech_Emotion_Recognition.ipynb
   
2. Follow the steps in the notebook to preprocess, train, and evaluate models.


## 🤝 Contributing
  Contributions are welcome! Please feel free to submit issues or pull requests.

## 📜 License
  This project is licensed under the MIT License. See the LICENSE file for details.
