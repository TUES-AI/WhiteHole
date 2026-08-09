from typing import Optional, NamedTuple
from enum import Enum, auto
from dataclasses import dataclass, field
from whitehole.configs import ConfigBase

from whitehole_envs.wall.data.offline_wall import OfflineWallDatasetConfig
from whitehole_envs.wall.data.wall import WallDatasetConfig
from whitehole_envs.wall.data.single import DotDatasetConfig
from whitehole_envs.wall.data.wall_expert import WallExpertDatasetConfig

class DatasetType(Enum):
    Single = auto()
    Multiple = auto()
    Wall = auto()
    WallExpert = auto()


class ProbingDatasets(NamedTuple):
    ds: DatasetType
    val_ds: DatasetType
    extra_datasets: dict = {}


class Datasets(NamedTuple):
    ds: DatasetType
    val_ds: DatasetType
    probing_datasets: Optional[ProbingDatasets] = None
    l2_probing_datasets: Optional[ProbingDatasets] = None


@dataclass
class DataConfig(ConfigBase):
    dataset_type: DatasetType = DatasetType.Single
    dot_config: DotDatasetConfig = field(default_factory=DotDatasetConfig)
    wall_config: WallDatasetConfig = field(default_factory=WallDatasetConfig)
    offline_wall_config: OfflineWallDatasetConfig = field(
        default_factory=OfflineWallDatasetConfig
    )
    wall_expert_config: WallExpertDatasetConfig = field(
        default_factory=WallExpertDatasetConfig
    )

    normalize: bool = False
    min_max_normalize_state: bool = False
    normalizer_hardset: bool = False
    quick_debug: bool = False
    num_workers: int = 0
