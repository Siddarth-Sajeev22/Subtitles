import collections
import threading
import whisper
import sounddevice as sd
import numpy as np
import queue

# Load the Whisper model (small model for faster transcription)
model = whisper.load_model("small")

# Queue to store audio chunks
audio_queue = queue.Queue()

# Parameters
SAMPLERATE = 16000  # Whisper expects 16kHz audio
BLOCKSIZE = 1024  # Size of each recorded audio block
DURATION = 1  # Duration of each chunk in seconds
DATA_TIME = 10

rolling_buffer = collections.deque(maxlen=((SAMPLERATE // BLOCKSIZE)*(DURATION)) * DATA_TIME * SAMPLERATE)
# Callback function to store audio in the queue
def callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    audio_queue.put(indata.copy())

def transcribe(): 
    try:
        while True:
            # Collect enough samples to match the chunk duration
            audio_chunk = []
            for _ in range(int((SAMPLERATE / BLOCKSIZE) * DURATION)):
                audio_chunk.append(audio_queue.get())

            # Convert to NumPy array
            rolling_buffer.extend(audio_chunk)
            audio_data = np.concatenate(rolling_buffer, axis=0)

            # Convert to float32 (Whisper expects float32 PCM)
            audio_data = audio_data.flatten().astype(np.float32)

            # Transcribe using Whisper
            result = model.transcribe(audio_data, fp16=False, language="english")

            # Print subtitles
            print("\n[Subtitle]:", result["text"])
    
    except KeyboardInterrupt:
        print("\nStopped by user.")

# Start audio streaming
with sd.InputStream(samplerate=SAMPLERATE, channels=1, callback=callback, blocksize=BLOCKSIZE):
    print("Listening... (Press Ctrl+C to stop)")
    event = threading.Thread(target=transcribe)
    event.start()
    try:
        threading.Event().wait()  # Wait until user interrupts
    except KeyboardInterrupt:
        print("\nStopped by user.")