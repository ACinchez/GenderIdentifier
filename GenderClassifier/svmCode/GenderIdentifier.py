import os
import pickle
import warnings
import numpy as np
import sounddevice as sd  # for capturing live audio
from FeaturesExtractor import FeaturesExtractor

warnings.filterwarnings("ignore")


class GenderIdentifier:
    def __init__(self, females_model_path, males_model_path):
        self.error = 0
        self.total_sample = 0
        self.features_extractor = FeaturesExtractor()
        # load models
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm = pickle.load(open(males_model_path, 'rb'))

    def process_audio(self, audio_path):
        self.total_sample += 1
        print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(audio_path)))

        vector = self.features_extractor.extract_features(audio_path)
        winner = self.identify_gender(vector)

        print("%10s %3s %1s" % ("+ IDENTIFICATION", ":", winner))
        print("----------------------------------------------------")

        return winner

    def process_live_audio(self):
        # Set the sample rate and duration for recording live audio
        sample_rate = 16000  # Sample rate (samples/second)
        duration = 5  # Duration of recording in seconds

        print("Recording live audio for {} seconds. Speak now...".format(duration))
        # Record audio
        recorded_audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished

        # Save recorded audio to a temporary file
        temp_audio_path = "recorded_audio.wav"
        sf.write(temp_audio_path, recorded_audio, sample_rate, subtype='PCM_16')

        # Process the recorded audio
        winner = self.process_audio(temp_audio_path)

        # Delete the temporary audio file
        os.remove(temp_audio_path)

        return winner

    def identify_gender(self, vector):
        # Female hypothesis scoring
        is_female_scores = np.array(self.females_gmm.score(vector))
        is_female_log_likelihood = is_female_scores.sum()
        # Male hypothesis scoring
        is_male_scores = np.array(self.males_gmm.score(vector))
        is_male_log_likelihood = is_male_scores.sum()

        if is_male_log_likelihood > is_female_log_likelihood:
            winner = "male"
        else:
            winner = "female"
        return winner


if __name__ == "__main__":
    females_model_path = "females.gmm"
    males_model_path = "males.gmm"

    # Initialize the GenderIdentifier with the trained models
    gender_identifier = GenderIdentifier(females_model_path, males_model_path)

    # Choose whether to process live audio or submit an audio file
    choice = input("Enter '1' to capture live audio, '2' to submit an audio file: ")

    if choice == '1':
        gender_identifier.process_live_audio()
    elif choice == '2':
        audio_path = input("Enter the path to the audio file: ")
        gender_identifier.process_audio(audio_path)
    else:
        print("Invalid choice. Please enter '1' or '2'.")
