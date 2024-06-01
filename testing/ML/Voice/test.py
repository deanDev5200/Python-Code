import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import sounddevice as sd

DATASET_PATH = 'data/speech_commands'
freq = 16000

duration = 1
print('start')
recording = sd.rec(int(duration * freq), 
                   samplerate=freq, channels=1)

sd.wait()

data_dir = pathlib.Path(DATASET_PATH)
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

model = tf.saved_model.load("saved")

x = tf.convert_to_tensor(recording)
x = tf.squeeze(x, axis=-1)
x = x[tf.newaxis, :]

prediction = model(x)
x_labels = ['_background_noise_', 'go', 'stop']
plt.bar(x_labels, tf.nn.softmax(prediction['predictions']).numpy()[0])
plt.title('No')
plt.show()
