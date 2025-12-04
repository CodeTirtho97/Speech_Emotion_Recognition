"""
Speech Emotion Recognition - ML Pipeline Module
Handles data preprocessing, feature extraction, model training, and saving
"""

import os
import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


class SpeechEmotionPipeline:
    """Complete pipeline for Speech Emotion Recognition"""

    def __init__(self, n_mfcc=13, n_components=10, target_sr=22050):
        """
        Initialize the pipeline with configuration parameters

        Args:
            n_mfcc: Number of MFCC features to extract
            n_components: Number of PCA components
            target_sr: Target sampling rate for audio
        """
        self.n_mfcc = n_mfcc
        self.n_components = n_components
        self.target_sr = target_sr

        # Pipeline components (will be fitted during training)
        self.pca = None
        self.scaler = None
        self.label_encoder = None
        self.best_model = None
        self.best_model_name = None
        self.models = {}

    def preprocess_audio(self, file_path):
        """
        Load and preprocess audio file

        Args:
            file_path: Path to audio file

        Returns:
            normalized_signal: Preprocessed audio signal
            sr: Sampling rate
        """
        signal, sr = librosa.load(file_path, sr=self.target_sr)
        normalized_signal = librosa.util.normalize(signal)
        return normalized_signal, sr

    def extract_features(self, file_path, n_fft=2048, hop_length=512):
        """
        Extract MFCC features from audio file

        Args:
            file_path: Path to audio file
            n_fft: FFT window size
            hop_length: Number of samples between successive frames

        Returns:
            mfccs_mean: Mean MFCC features
        """
        signal, sr = self.preprocess_audio(file_path)
        mfccs = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mfccs_mean = mfccs.mean(axis=1)
        return mfccs_mean

    def load_dataset(self, dataset_path):
        """
        Load dataset from directory structure

        Args:
            dataset_path: Path to dataset folder

        Returns:
            DataFrame with file paths and emotion labels
        """
        emotions = os.listdir(dataset_path)
        data = []
        for emotion in emotions:
            emotion_path = os.path.join(dataset_path, emotion)
            if not os.path.isdir(emotion_path):
                continue
            for file in os.listdir(emotion_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(emotion_path, file)
                    data.append({"path": file_path, "emotion": emotion})
        return pd.DataFrame(data)

    def extract_dataset_features(self, data):
        """
        Extract features for entire dataset

        Args:
            data: DataFrame with 'path' and 'emotion' columns

        Returns:
            X: Feature matrix
            y: Labels
        """
        features = []
        labels = []
        print(f"Extracting features from {len(data)} audio files...")

        for idx, row in data.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(data)} files...")

            mfccs = self.extract_features(row['path'])
            features.append(mfccs)
            labels.append(row['emotion'])

        print(f"Feature extraction complete!")
        return np.array(features), np.array(labels)

    def train(self, dataset_path, sample_frac=0.5, test_size=0.2, random_state=42):
        """
        Train the complete pipeline

        Args:
            dataset_path: Path to training dataset
            sample_frac: Fraction of data to use (for faster training)
            test_size: Test set size
            random_state: Random seed

        Returns:
            results_df: DataFrame with model comparison results
        """
        print("=" * 60)
        print("STARTING SPEECH EMOTION RECOGNITION TRAINING PIPELINE")
        print("=" * 60)

        # 1. Load dataset
        print("\n[1/7] Loading dataset...")
        data = self.load_dataset(dataset_path)
        print(f"Total files found: {len(data)}")

        # Sample data if needed
        if sample_frac < 1.0:
            data = data.sample(frac=sample_frac, random_state=random_state)
            print(f"Sampled {len(data)} files ({sample_frac*100}% of dataset)")

        # 2. Extract features
        print(f"\n[2/7] Extracting MFCC features (n_mfcc={self.n_mfcc})...")
        X, y = self.extract_dataset_features(data)
        print(f"Feature shape: {X.shape}")

        # 3. Split data
        print(f"\n[3/7] Splitting dataset (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # 4. Apply PCA
        print(f"\n[4/7] Applying PCA (n_components={self.n_components})...")
        self.pca = PCA(n_components=self.n_components)
        X_train_reduced = self.pca.fit_transform(X_train)
        X_test_reduced = self.pca.transform(X_test)
        print(f"Reduced feature shape: {X_train_reduced.shape}")
        print(f"Total variance retained: {sum(self.pca.explained_variance_ratio_):.3f}")

        # 5. Standardize features
        print("\n[5/7] Standardizing features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_reduced)
        X_test_scaled = self.scaler.transform(X_test_reduced)

        # 6. Balance dataset with SMOTE
        print("\n[6/7] Balancing dataset with SMOTE...")
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"Balanced training set: {len(X_train_balanced)} samples")

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train_balanced)
        y_test_encoded = self.label_encoder.transform(y_test)

        # 7. Train models
        print("\n[7/7] Training models...")
        print("-" * 60)

        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=random_state),
            "SVM": SVC(kernel='rbf', class_weight='balanced', random_state=random_state, probability=True),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=random_state, n_estimators=100)
        }

        results = []

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_balanced, y_train_encoded)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test_encoded, y_pred)
            report = classification_report(y_test_encoded, y_pred, output_dict=True, zero_division=1)

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": report["macro avg"]["precision"],
                "Recall": report["macro avg"]["recall"],
                "F1-Score": report["macro avg"]["f1-score"],
            })

            print(f"  Accuracy: {acc:.4f}")

        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="Accuracy", ascending=False)

        # Select best model
        best_row = results_df.iloc[0]
        self.best_model_name = best_row["Model"]
        self.best_model = self.models[self.best_model_name]

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print("\nModel Comparison:")
        print(results_df.to_string(index=False))
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Accuracy: {best_row['Accuracy']:.4f}")
        print("=" * 60)

        return results_df

    def save_pipeline(self, save_dir='models'):
        """
        Save all pipeline components to disk

        Args:
            save_dir: Directory to save models
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"\nSaving pipeline to '{save_dir}' directory...")

        # Save all components
        joblib.dump(self.best_model, os.path.join(save_dir, 'best_model.pkl'))
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(save_dir, 'pca.pkl'))
        joblib.dump(self.label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))

        # Save configuration
        config = {
            'n_mfcc': self.n_mfcc,
            'n_components': self.n_components,
            'target_sr': self.target_sr,
            'best_model_name': self.best_model_name,
            'emotion_classes': self.label_encoder.classes_.tolist()
        }
        joblib.dump(config, os.path.join(save_dir, 'config.pkl'))

        print("[OK] Saved: best_model.pkl")
        print("[OK] Saved: scaler.pkl")
        print("[OK] Saved: pca.pkl")
        print("[OK] Saved: label_encoder.pkl")
        print("[OK] Saved: config.pkl")
        print(f"\nPipeline saved successfully!")

    def load_pipeline(self, save_dir='models'):
        """
        Load pipeline components from disk

        Args:
            save_dir: Directory containing saved models
        """
        print(f"Loading pipeline from '{save_dir}' directory...")

        self.best_model = joblib.load(os.path.join(save_dir, 'best_model.pkl'))
        self.scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
        self.pca = joblib.load(os.path.join(save_dir, 'pca.pkl'))
        self.label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))

        config = joblib.load(os.path.join(save_dir, 'config.pkl'))
        self.n_mfcc = config['n_mfcc']
        self.n_components = config['n_components']
        self.target_sr = config['target_sr']
        self.best_model_name = config['best_model_name']

        print("[OK] Loaded: best_model.pkl")
        print("[OK] Loaded: scaler.pkl")
        print("[OK] Loaded: pca.pkl")
        print("[OK] Loaded: label_encoder.pkl")
        print("[OK] Loaded: config.pkl")
        print(f"\nPipeline loaded successfully!")
        print(f"Model: {self.best_model_name}")
        print(f"Emotion classes: {self.label_encoder.classes_.tolist()}")

    def predict(self, audio_file_path):
        """
        Predict emotion from audio file

        Args:
            audio_file_path: Path to audio file

        Returns:
            Dictionary with prediction results
        """
        if self.best_model is None:
            raise ValueError("Pipeline not trained or loaded. Call train() or load_pipeline() first.")

        # Extract features
        features = self.extract_features(audio_file_path)
        features = features.reshape(1, -1)  # Reshape for single sample

        # Apply PCA
        features_reduced = self.pca.transform(features)

        # Scale features
        features_scaled = self.scaler.transform(features_reduced)

        # Predict
        prediction_encoded = self.best_model.predict(features_scaled)[0]
        predicted_emotion = self.label_encoder.inverse_transform([prediction_encoded])[0]

        # Get probability scores if available
        probabilities = None
        if hasattr(self.best_model, 'predict_proba'):
            proba = self.best_model.predict_proba(features_scaled)[0]
            probabilities = {
                emotion: float(prob)
                for emotion, prob in zip(self.label_encoder.classes_, proba)
            }

        return {
            'predicted_emotion': predicted_emotion,
            'confidence_scores': probabilities,
            'model_used': self.best_model_name
        }


if __name__ == "__main__":
    # Example usage
    print("Speech Emotion Recognition Pipeline")
    print("This module provides the complete ML pipeline for emotion recognition.")
    print("\nUsage:")
    print("  from pipeline import SpeechEmotionPipeline")
    print("  pipeline = SpeechEmotionPipeline()")
    print("  pipeline.train('path/to/dataset')")
    print("  pipeline.save_pipeline('models')")
    print("  result = pipeline.predict('path/to/audio.wav')")
