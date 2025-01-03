import torch
import torchaudio
from torchsynth.synth import Voice

voice = Voice()
# Run on the GPU if it's available
if torch.cuda.is_available():
    voice = voice.to("cuda")

# Generate batch 312
# All audio batches are [128, 176400], i.e. 128 4-second sounds at 44100Hz
# Each sound is a monophonic 1D tensor.
# Param batches are [128, 72], which are the 72 latent Voice
# parameters that generated each sound.
# The training tensor is a [128] bool, indicating whether
# instances are designated as train or test, for reproducibility.
synth1B1_312_audio, synth1B1_312_params, synth1B1_312_is_train = voice(312)

# Select synth1B1-312-6
synth1B1_312_6 = synth1B1_312_audio[6]

# We add one channel at the beginning, for torchaudio
torchaudio.save("synth1B1-312-6.wav", synth1B1_312_6.unsqueeze(0).cpu(), voice.sample_rate)