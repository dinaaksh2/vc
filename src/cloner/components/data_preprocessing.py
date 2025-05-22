TTS_PATH = "D:/office/vc/TTS"
import sys
import os
from scipy.stats import norm
import numpy as np
from TTS.tts.datasets.formatters import *
from TTS.utils.audio import AudioProcessor
from TTS.config.shared_configs import BaseAudioConfig
from cloner import logger 
import librosa
from pathlib import Path
from cloner.constants import *
from cloner.utils.common import read_yaml
from cloner.entity.config_entity import DataPreProcessConfig
CONFIG = BaseAudioConfig()

class DataPreprocessor:
    def __init__(self,config: DataPreProcessConfig):
        self.config=config
        self.params=read_yaml(PARAMS_FILE_PATH)
        self.audio_processor=AudioProcessor

    def get_audio_processor(self)-> AudioProcessor:
        tuned_config = CONFIG.copy()
        tuned_config.update(self.params.get("reset", {}))
        tuned_config.update(self.params.get("tune_params", {}))
        logger.info("Initialized AudioProcessor with tuned config.")
        self.audio_processor = AudioProcessor(**tuned_config)
        return self.audio_processor
    
    def melspectrogram(self, audio_path):
        logger.info(f"Generating mel-spectrogram for: {audio_path}")
        y, sr = librosa.load(audio_path, sr=self.params["tune_params"]["sample_rate"])
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.params["tune_params"]["num_mels"],
            n_fft=self.params["tune_params"]["fft_size"],
            hop_length=self.params["tune_params"].get("hop_length") or 256
        )
        return mel_spectrogram

  
    def process_audio(self):
        root_dir=Path(self.config.root_dir)
        processed_audio_dir=Path(self.config.processed_audio_dir)
        audio_path=Path(self.config.audio_path)
        logger.info(f"Starting audio processing...")
        logger.info(f"Looking for WAV files in: {audio_path}")

        os.makedirs(processed_audio_dir, exist_ok=True)

        audio_files = list(audio_path.glob("*.wav"))
        logger.info(f"Found {len(audio_files)} audio files.")

        for audio_file in audio_files:
            try:
                mel_spec = self.melspectrogram(audio_file)
                mel_spec_file = Path(processed_audio_dir) / f"{audio_file.stem}.npy"
                np.save(mel_spec_file, mel_spec)
                logger.info(f"Processed and saved mel-spectrogram for {mel_spec_file}")

            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
        logger.info(f"Completed processing {len(audio_files)} files.")