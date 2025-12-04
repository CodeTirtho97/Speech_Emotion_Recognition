"""
Test script for the Speech Emotion Recognition Pipeline
Tests prediction on sample audio files
"""

from pipeline import SpeechEmotionPipeline
import os
import glob

def test_pipeline():
    """Test the trained pipeline with sample audio files"""

    print("\n" + "="*60)
    print("TESTING SPEECH EMOTION RECOGNITION PIPELINE")
    print("="*60)

    # Load the trained pipeline
    print("\n[1/3] Loading trained pipeline...")
    pipeline = SpeechEmotionPipeline()

    try:
        pipeline.load_pipeline(save_dir='models')
    except Exception as e:
        print(f"\n[ERROR] Could not load pipeline.")
        print(f"Error message: {e}")
        print("\n[INFO] Please train the model first by running:")
        print("   python train_model.py")
        return

    # Find test audio files
    print("\n[2/3] Finding test audio files...")

    # Try to find audio files from the dataset
    test_files = []

    # Check for unused dataset
    if os.path.exists('unused_dataset'):
        test_files = glob.glob('unused_dataset/*.wav')[:5]

    # If no unused dataset, use samples from main dataset
    if not test_files and os.path.exists('Toronto Emotional Speech Set'):
        for emotion_folder in os.listdir('Toronto Emotional Speech Set'):
            folder_path = os.path.join('Toronto Emotional Speech Set', emotion_folder)
            if os.path.isdir(folder_path):
                files = glob.glob(os.path.join(folder_path, '*.wav'))[:1]
                test_files.extend(files)
                if len(test_files) >= 5:
                    break

    if not test_files:
        print("\n[WARNING] No test audio files found!")
        print("Please provide audio files in 'unused_dataset' folder or check dataset path.")
        return

    print(f"Found {len(test_files)} test files")

    # Test predictions
    print("\n[3/3] Testing predictions...")
    print("-"*60)

    for i, audio_file in enumerate(test_files, 1):
        print(f"\n[Test {i}/{len(test_files)}]")
        print(f"File: {os.path.basename(audio_file)}")

        try:
            result = pipeline.predict(audio_file)

            print(f"[OK] Predicted Emotion: {result['predicted_emotion']}")
            print(f"  Model Used: {result['model_used']}")

            if result['confidence_scores']:
                print("\n  Confidence Scores:")
                sorted_scores = sorted(
                    result['confidence_scores'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for emotion, score in sorted_scores:
                    bar = '#' * int(score * 20)
                    print(f"    {emotion:20s} {score:6.2%} {bar}")

        except Exception as e:
            print(f"[ERROR] Error predicting: {e}")

        print("-"*60)

    print("\n[OK] Testing complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_pipeline()
