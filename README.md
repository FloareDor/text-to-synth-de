# Neural Audio Synthesis Explorer

This project explores neural audio synthesis by combining CLAP (Contrastive Language-Audio Pretraining) embeddings with a modular synthesizer to create sounds from text descriptions or match existing audio samples.

## Overview

The project consists of two main components:

1. **Text-to-Synth**: Generate synthesizer parameters from text descriptions
2. **Inverse-Synth**: Reverse engineer synthesizer parameters to match target audio samples

Both approaches use CLAP embeddings and differential evolution to optimize synthesizer parameters, creating an AI-guided sound design system.

## Features

- Text-to-audio synthesis using natural language descriptions
- Audio matching through inverse synthesis
- Parallel processing for faster optimization
- CUDA support for GPU acceleration
- Parameter save/load functionality
- Modular synthesizer with:
  - Dual oscillators
  - Dual LFOs
  - Multiple ADSR envelopes
  - Modulation matrix
  - Noise generator

## Examples

### Text-to-Synth Example

Input text: "engine"

🔊 [engine.wav](examples/engine.wav)

### Inverse Synthesis Example

Original audio:

🔊 [chirpingbirds.wav](examples/chirpingbirds.wav)

Synthesized match:

🔊 [chirpingbirds_output.wav](examples/chirpingbirds_output.wav)

## Installation

```bash
Clone the repository
git clone https://github.com/floaredor/text-to-synth-de.git
```

Install dependencies
```bash
pip install torch torchaudio laion_clap torchsynth pymoo librosa numpy
```


## Usage

### Text-to-Synth

```bash
python text-to-synth.py
```

### Inverse-Synth

```bash
python inverse-synth.py
```


## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch
- LAION-CLAP
- TorchSynth
- Pymoo
- LibROSA
- NumPy

## License

MIT License

## Acknowledgments

- LAION-CLAP team for the audio-text embedding model
- TorchSynth team for the synthesizer implementation
- Pymoo team for the optimization framework


