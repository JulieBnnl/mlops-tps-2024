from zenml import step

from src.models.model_dataset import Dataset
from src.models.model_bucket_client import BucketClient

import os


@step
def dataset_extractor(
    dataset: Dataset, bucket_client: BucketClient, extraction_path: str
):

    dataset.download(bucket_client, extraction_path)

    return os.path.join(extraction_path, dataset.uuid)
