from omegaconf import OmegaConf
from zenml import pipeline

from src.config.settings import (
    EXTRACTED_DATASETS_PATH,
    MLFLOW_EXPERIMENT_PIPELINE_NAME,
)
from src.steps.data.data_extractor import dataset_extractor
from src.steps.data.datalake_initializers import (
    data_source_list_initializer,
    minio_client_initializer,
)
from src.steps.data.dataset_preparators import (
    dataset_creator,
    dataset_to_yolo_converter,
)
from src.steps.training.model_appraisers import model_appraiser
from src.steps.training.model_evaluators import model_evaluator
from src.steps.training.model_trainers import (
    get_pre_trained_weights_path,
    model_trainer,
)

from src.models.model_data_source import DataSourceList


@pipeline(name=MLFLOW_EXPERIMENT_PIPELINE_NAME)
def gitflow_experiment_pipeline(cfg: str) -> None:
    """
    Experiment a local training and evaluate if the model can be deployed.

    Args:
        cfg: The Hydra configuration.
    """
    pipeline_config = OmegaConf.to_container(OmegaConf.create(cfg))

    bucket_client = minio_client_initializer()
    data_source_list = data_source_list_initializer()

    bucket_name = "data-sources"

    # Prepare/create the dataset
    dataset = dataset_creator(
        data_source_list=data_source_list, bucket_name=bucket_name, seed=1234
    )

    # Extract the dataset to a folder
    extraction_path = dataset_extractor(dataset, bucket_client, EXTRACTED_DATASETS_PATH)

    # If necessary, convert the dataset to a YOLO format
    dataset_to_yolo_converter(dataset, extraction_path)

    # Train the model
    # trained_model_path = model_trainer()

    # model_predict(trained_model_path, ["datasets/plastic_in_river/images/test/0a6acc8c147b25fd58f9c2b6a9e1c1e7af48d94738ec8421180cd264d71273a3.png"])

    # Evaluate the model
    # test_metrics_result = model_evaluator(trained_model_path, EXTRACTED_DATASETS_PATH)

    # Retrieve a decision if the model should be deployed
    # can_model_be_deployed = model_appraiser(
    #     ...
    # )

    # if can_model_be_deployed:
    #     model_registerer(...)
    #     model_deployer(...)
    #
    # else:
    #     print(...)
