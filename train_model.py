"""
Training Script for Speech Emotion Recognition Model
Run this script to train the model and save pipeline artifacts
"""

from pipeline import SpeechEmotionPipeline
import os

def main():
    # Configuration
    DATASET_PATH = 'Toronto Emotional Speech Set'
    MODELS_DIR = 'models'
    SAMPLE_FRAC = 0.5  # Use 50% of data for faster training (set to 1.0 for full dataset)
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Initialize pipeline
    pipeline = SpeechEmotionPipeline(
        n_mfcc=13,
        n_components=10,
        target_sr=22050
    )

    # Train the pipeline
    print("\n>> Starting training process...\n")
    results_df = pipeline.train(
        dataset_path=DATASET_PATH,
        sample_frac=SAMPLE_FRAC,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Save the pipeline
    print("\n>> Saving trained models...")
    pipeline.save_pipeline(save_dir=MODELS_DIR)

    print("\n>> Training complete! Models are ready for deployment.")
    print(f">> Models saved in: {os.path.abspath(MODELS_DIR)}")


if __name__ == "__main__":
    main()
