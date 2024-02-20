from typing import List

from zenml import step
from zenml.logger import get_logger

from typing import Any, Dict

from src.config.settings import MINIO_DATASETS_BUCKET_NAME
from src.models.model_data_source import DataSourceList
from src.models.model_bucket_client import BucketClient
from src.models.model_bucket_client import BucketClient
from src.models.model_data_source import DataSourceList
from src.models.model_dataset import Dataset


@step
def data_creator() -> str:
    # Créer un objet Dataset
    dataset = Dataset(
        bucket_name=MINIO_DATASETS_BUCKET_NAME,
        seed=42,  # Vous pouvez choisir une autre graine aléatoire si nécessaire
    )

    return dataset


@step
def dataset_to_yolo_converter(dataset: Dataset, EXTRACTED_DATASETS_PATH: str) -> None:
    dataset.to_yolo_format(EXTRACTED_DATASETS_PATH)
