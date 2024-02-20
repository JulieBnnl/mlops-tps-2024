import os
import shutil
from typing import List

from zenml import step
from zenml.logger import get_logger

from src.models.model_dataset import Dataset
from src.models.model_bucket_client import BucketClient


@step
def dataset_extractor(
    dataset: Dataset, bucket_client: BucketClient, EXTRACTED_DATASETS_PATH: str
) -> None:

    dataset.download(bucket_client, EXTRACTED_DATASETS_PATH)
