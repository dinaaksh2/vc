from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path

@dataclass(frozen=True)
class DataPreProcessConfig:
    root_dir:Path
    processed_audio_dir:Path
    audio_path:Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    output_dir: Path
    phoneme_cache_path: Path
    dataset_name: str
    dataset_path: Path
    metadata_path: Path
    restore_path: Path