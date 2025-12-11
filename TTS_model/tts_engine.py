#!/usr/bin/env python3
import os
import torch
import soundfile as sf
import nemo.collections.tts as nemo_tts


FASTPITCH_NAME = "tts_en_fastpitch"
HIFIGAN_NAME = "tts_en_hifigan"
DEFAULT_MODELS_DIR = "models"
SAMPLE_RATE = 22050


class TTSEngine:
    def __init__(self, device: str | None = None, models_dir: str = DEFAULT_MODELS_DIR):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        self._fastpitch = None
        self._vocoder = None

    @property
    def fastpitch_path(self) -> str:
        return os.path.join(self.models_dir, FASTPITCH_NAME + ".nemo")

    @property
    def vocoder_path(self) -> str:
        return os.path.join(self.models_dir, HIFIGAN_NAME + ".nemo")

    def _load_fastpitch(self):
        if self._fastpitch is not None:
            return self._fastpitch

        if os.path.exists(self.fastpitch_path):
            model = nemo_tts.models.FastPitchModel.restore_from(
                self.fastpitch_path, map_location=self.device
            )
        else:
            model = nemo_tts.models.FastPitchModel.from_pretrained(
                model_name=FASTPITCH_NAME, map_location=self.device
            )
            model.save_to(self.fastpitch_path)

        model.eval()
        self._fastpitch = model
        return model

    def _load_vocoder(self):
        if self._vocoder is not None:
            return self._vocoder

        HifiGanModel = nemo_tts.models.HifiGanModel
        if os.path.exists(self.vocoder_path):
            vocoder = HifiGanModel.restore_from(
                self.vocoder_path, map_location=self.device
            )
        else:
            vocoder = HifiGanModel.from_pretrained(
                model_name=HIFIGAN_NAME, map_location=self.device
            )
            vocoder.save_to(self.vocoder_path)

        vocoder.eval()
        self._vocoder = vocoder
        return vocoder

    def synthesize(self, text: str, out_path: str = "output.wav") -> str:
        """Generate speech for the given text and save it to out_path."""
        fastpitch = self._load_fastpitch()
        vocoder = self._load_vocoder()

        with torch.no_grad():
            tokens = fastpitch.parse(text)
            spectrogram = fastpitch.generate_spectrogram(tokens=tokens)
            audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

        audio = audio.cpu().numpy()[0]
        sf.write(out_path, audio, SAMPLE_RATE)
        return out_path

