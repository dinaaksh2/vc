from cloner.constants import *
from cloner.utils.common import read_yaml, create_directories
from cloner.entity.config_entity import DataIngestionConfig
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