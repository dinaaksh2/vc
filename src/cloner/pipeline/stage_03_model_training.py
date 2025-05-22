from cloner.config.configuration import ConfigurationManager
from cloner.components.model_training import ModelConfig
from cloner import logger

STAGE_NAME="Model Configuration"

class ModelConfiguration:
    def __init__(self):
        pass

    def main(self):
            config=ConfigurationManager()
            model_training_config=config.get_model_training_config()
            model_training=ModelConfig(config=model_training_config)
            model_training.get_audio_config()
            model_training.get_dataset_config()
            model_training.get_vits_config()
            model_training.get_fit()

if __name__=='__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj=ModelConfiguration()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
