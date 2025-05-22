from cloner.constants import *
from cloner.utils.common import read_yaml, create_directories
from cloner.entity.config_entity import DataIngestionConfig, DataPreProcessConfig, ModelTrainingConfig
class ConfigurationManager:
    def __init__(
            self,
            config_filepath= CONFIG_FILE_PATH,
            params_filepath= PARAMS_FILE_PATH):
            
            self.config=read_yaml(config_filepath)
            self.params=read_yaml(params_filepath)

            create_directories([self.config.artifacts_root])  #using box return type we can call the key-value pair with ".", here we are creating directories mentioned in artifacts

    def get_data_ingestion_config(self)->DataIngestionConfig:
          config=self.config.data_ingestion #storing configuration  inside config variable
          create_directories([config.root_dir]) #creating artifact directory

          data_ingestion_config=DataIngestionConfig( #defining return type
                root_dir=config.root_dir,
                source_URL=config.source_URL,
                local_data_file=config.local_data_file,
                unzip_dir=config.unzip_dir
          ) 

          return data_ingestion_config 
    
    def get_data_preprocess_config(self)-> DataPreProcessConfig:
          config=self.config.data_preprocessing 
          create_directories([config.root_dir]) 

          data_preprocess_config=DataPreProcessConfig(
                root_dir=config.root_dir,
                processed_audio_dir=config.processed_audio_dir,
                audio_path=config.audio_path
          ) 

          return data_preprocess_config
    
    def get_model_training_config(self)-> ModelTrainingConfig:
          config=self.config.model_training
          create_directories([config.root_dir]) 

          model_training_config=ModelTrainingConfig( 
                root_dir= config.root_dir,
                output_dir= config.output_dir,
                phoneme_cache_path= config.phoneme_cache_path,
                dataset_name= config.dataset_name,
                dataset_path= config.dataset_path,
                metadata_path= config.metadata_path,
                restore_path= config.restore_path
                ) 

          return model_training_config