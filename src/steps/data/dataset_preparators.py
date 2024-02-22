from zenml import step

from src.models.model_data_source import DataSourceList
from src.models.model_dataset import Dataset
from src.materializers.materializer_dataset import DatasetMaterializer


@step(output_materializers=DatasetMaterializer)
def dataset_creator(
    data_source_list: DataSourceList,
    bucket_name: str,
    seed: int,
    uuid: str | None = None,
    annotations_path: str = "labels",
    images_path: str = "images",
    distribution_weights: list[float] | None = None,
    label_map: dict[int, str] | None = None,
) -> Dataset:

    data_source = data_source_list.data_sources[0]

    # Créer un objet Dataset
    dataset = Dataset(
        bucket_name=bucket_name,
        seed=seed,
        uuid=data_source.name,
        annotations_path=annotations_path,
        images_path=images_path,
        distribution_weights=distribution_weights,
        label_map={
            0: "road-traffic",
            1: "bicycles",
            2: "buses",
            3: "crosswalks",
            4: "fire hydrants",
            5: "motorcycles",
            6: "traffic lights",
            7: "vehicles",
        },
    )

    return dataset


@step
def dataset_to_yolo_converter(dataset: Dataset, extraction_path: str) -> None:
    dataset.to_yolo_format(extraction_path)
