# Usage: 

In a directory that is a parent of cli_app, run:

```
python cli_app\clean_speech.py mixed_audio.wav clean_audio.wav
```

...being sure to include the .wav and .py extensions. This will run the UNet model mixed_audio.wav to generate a new file in the parent directory called clean_audio.wav.

Required Python modules:
- tensorflow
- librosa
- soundfile
- numpy
