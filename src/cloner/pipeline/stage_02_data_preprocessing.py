from cloner.config.configuration import ConfigurationManager
from cloner.components.data_preprocessing import DataPreprocessor
from cloner import logger

STAGE_NAME="Data Preprocessing"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
            config=ConfigurationManager()
            data_preprocess_config=config.get_data_preprocess_config()
            data_preprocess=DataPreprocessor(config=data_preprocess_config)
            data_preprocess.get_audio_processor()
            data_preprocess.process_audio()

if __name__=='__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj=DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e