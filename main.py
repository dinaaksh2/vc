from cloner import logger 
from cloner.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cloner.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline

STAGE_NAME="Data Ingestion Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion=DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Data Preprocessing Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_preprocessing=DataPreprocessingPipeline()
    data_preprocessing.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e