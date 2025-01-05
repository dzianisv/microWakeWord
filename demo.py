from microwakeword import inference
import os
import urllib.request
import logging
import pyaudio
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

SAMPLE_RATE=16000
STEP_MS=30
sliding_window_average_size = 10
probability_cutoff = 0.5

def require_model():
    model_filename = "hey_jarvis.tflite"
    model_url = "https://github.com/esphome/micro-wake-word-models/raw/refs/heads/main/models/hey_jarvis.tflite"
    
    # Check if the model file exists in the current working directory
    if not os.path.exists(model_filename):
        logging.info(f"{model_filename} not found. Downloading...")
        # Download the model file
        urllib.request.urlretrieve(model_url, model_filename)
        logging.info(f"Downloaded {model_filename}.")
    
    # Return the path to the model file
    return os.path.abspath(model_filename)

model = inference.Model(tflite_model_path=require_model())

def capture_audio_and_predict():
    # Initialize PyAudio
    p = pyaudio.PyAudio()


    # Define audio stream parameters
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=SAMPLE_RATE)

    try:
        while True:
            # Read audio data from the stream
            read_buffer_size = sliding_window_average_size * SAMPLE_RATE * STEP_MS // 1000
            audio_data = stream.read(read_buffer_size)
            # Convert audio data to numpy array
            data = np.frombuffer(audio_data, dtype=np.int16)
            probabilities = model.predict_clip(data, step_ms=30)
            print(probabilities)
            avg = sum(probabilities) / len(probabilities)
            if avg >= probability_cutoff:
                print("detected")

    except KeyboardInterrupt:
        logging.info("Stopping audio capture.")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate PyAudio
        p.terminate()

# Call the function to start capturing audio and predicting
capture_audio_and_predict()