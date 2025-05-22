from pathlib import Path
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from cloner.utils.common import read_yaml
from cloner.pipeline.stage_02_data_preprocessing import DataPreprocessor
from cloner.entity.config_entity import DataPreProcessConfig
from cloner.config.configuration import ModelTrainingConfig
from cloner.constants import *
from cloner.utils.common import read_yaml

import os

class ModelConfig:
    def __init__(self,config: ModelTrainingConfig):
        self.config=config
        self.params=read_yaml(PARAMS_FILE_PATH)

        self.audio_config=self.get_audio_config()
        self.dataset_config=self.get_dataset_config()
        self.vits_config=self.get_vits_config()

        self._audio_processor=None
        self._tokenizer=None
        self._model=None
        self._trainer_instance=None
        self._train_samples=None
        self._eval_samples=None

    def get_audio_config(self):
        return self.params["audio_config"]

    def get_dataset_config(self):
        return BaseDatasetConfig(
            formatter=self.config.dataset_name,
            meta_file_train=self.config.metadata_path,
            path=self.config.dataset_path
        )

    def get_vits_config(self):
        config=self.config
        params=self.params["model_config"]
        audio_config=self.audio_config
        dataset_config=self.dataset_config
        return VitsConfig(
            audio=audio_config,
            run_name=params["run_name"],
            batch_size=params["batch_size"],
            eval_batch_size=params["eval_batch_size"],
            batch_group_size=params["batch_group_size"],
            num_loader_workers=params["num_loader_workers"],
            num_eval_loader_workers=params["num_eval_loader_workers"],
            run_eval=params["run_eval"],
            test_delay_epochs=params["test_delay_epochs"],
            epochs=params["epochs"],
            text_cleaner=params["text_cleaner"],
            use_phonemes=params["use_phonemes"],
            phoneme_language=params["phoneme_language"],
            phoneme_cache_path=os.path.join(params["output_path"], "phoneme_cache"),
            compute_input_seq_cache=params["compute_input_seq_cache"],
            print_step=params["print_step"],
            print_eval=params["print_eval"],
            mixed_precision=params["mixed_precision"],
            output_path=params["output_path"],
            datasets=[dataset_config],
            cudnn_benchmark=params["cudnn_benchmark"],
        )

    def get_audio_processor(self):
        if self._audio_processor is None:
            data_preprocess_config=DataPreProcessConfig(
                root_dir=self.config.root_dir,
                processed_audio_dir="",  
                audio_path=""         
            )
            processor=DataPreprocessor(config=data_preprocess_config)
            self._audio_processor=processor.get_audio_processor()
        return self._audio_processor

    def get_tokenizer(self):
            if self._tokenizer is None:
                vits_config=self.vits_config
                tokenizer, config=TTSTokenizer.init_from_config(vits_config)
                self._tokenizer=tokenizer
            return self._tokenizer
    
    def get_data_split(self):
        if self._train_samples is None or self._eval_samples is None: 
            self._train_samples,self._eval_samples=load_tts_samples(
                self.dataset_config,
                eval_split=True,
                eval_split_max_size=self.vits_config.eval_split_max_size,
                eval_split_size=self.vits_config.eval_split_size,
            )
        return self._train_samples,self._eval_samples
    
    def get_model(self):
        if self._model is None:
            config=self.vits_config
            ap=self.get_audio_processor()
            tokenizer=self.get_tokenizer()
            self._model=Vits(config,ap,tokenizer,speaker_manager=None)
            return self._model
    
    def get_trainer(self):
        if self._trainer_instance is None:
            train_samples,eval_samples=self.get_data_split()
            self._trainer_instance=Trainer(
                TrainerArgs(),
                config=self.vits_config,
                output_path=self.config.output_dir,
                model=self.get_model(),
                train_samples=train_samples,
                eval_samples=eval_samples,
                parse_command_line_args=False
            )
        return self._trainer_instance

    def get_fit(self):
        trainer=self.get_trainer()
        if trainer is None:
            raise ValueError("Trainer instance is None. Cannot start training.")
        trainer.fit()