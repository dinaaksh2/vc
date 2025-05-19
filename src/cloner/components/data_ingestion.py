import os
import urllib.request as request
import zipfile
from cloner import logger
from cloner.utils.common import get_size
from cloner.entity.config_entity import DataIngestionConfig
from pathlib import Path
import gdown

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config=config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            file_id = self.config.source_URL.split('/d/')[1].split('/')[0]
            gdrive_url = f"https://drive.google.com/uc?id={file_id}"
        
            gdown.download(url=gdrive_url, output=self.config.local_data_file, quiet=False)
            logger.info(f"Downloaded file to {self.config.local_data_file}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        unzip_path=self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)