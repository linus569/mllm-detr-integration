from dataclasses import MISSING, dataclass, field
from typing import List, Optional, Tuple

from omegaconf import MISSING


@dataclass
class DatasetConfig:
    name: str = MISSING

    # --- Normalization parameters ---
    # pixel_mean: List[float] = MISSING
    # pixel_std: List[float] = MISSING

    # --- Dataset parameters ---
    data_dir: str = MISSING
    annotations_dir: str = MISSING


@dataclass
class ImageEncoderConfig:
    name: str = MISSING
    model_path: Optional[str] = None

    # Image processing parameters
    image_size: Tuple[int, int] = MISSING
    mean: List[float] = MISSING
    std: List[float] = MISSING
    interpolation: int = MISSING

    num_image_tokens: int = MISSING
    use_pooler_output: Optional[bool] = None


@dataclass
class ExperimentConfig:
    name: str = MISSING
    seed: int = MISSING

    train_dataset: DatasetConfig = MISSING
    val_dataset: DatasetConfig = MISSING
    test_dataset: DatasetConfig = MISSING

    image_encoder: ImageEncoderConfig = MISSING

    main_dir: str = MISSING
    model_name: str = "lmms-lab/llava-onevision-qwen2-0.5b-si"

    checkpoint_dir: str = "checkpoints"
    load_checkpoint: Optional[str] = None

    train: bool = True
    evaluate: bool = True
    eval_mode: str = "val"
    # eval_tasks: Dict[str, Any] = field(default_factory=dict)

    num_samples: Optional[int] = None
    val_num_samples: Optional[int] = None
    max_tokens: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    batch_size: int = MISSING
    total_batch_size: int = MISSING

    epochs: int = MISSING
    # max_steps: Optional[int] = None
    # max_epochs: Optional[int] = None
    lr: float = MISSING
    # min_lr: float = MISSING
    # warmup_lr: Optional[float] = MISSING
    warmup_ratio: float = MISSING
    weight_decay: Optional[float] = MISSING
    max_grad_norm: Optional[float] = MISSING
    # grad_clip_norm: Optional[float] = MISSING
    # early_stopping_patience: Optional[int] = None

    # metric: Optional[str] = None
    # metric_mode: str = 'max'

    val_freq: Optional[int] = None
    val_ep: Optional[int] = None
    print_freq: int = MISSING
    num_workers: int = MISSING
    device: str = MISSING
    debug: bool = False
    # compile: bool = True
    save_components: List[str] = field(default_factory=list)

    temperature: float = MISSING
    use_amp: bool = MISSING  # use automatic mixed precision
    torch_dtype: Optional[str] = None

    image_token: str = MISSING
    num_coordinate_bins: int = MISSING

    add_special_tokens: bool = True
    freeze_model: bool = True
    train_image_encoder: bool = False

    bbox_ordering: str = "none"  # Options: "none", "random", "size_desc"

    # detr parameters
    detr_type: str = "detr"  # Options: "detr", "dabdetr"
    num_query_tokens: int = 0
    detr_loss: bool = False
    add_detr_layers: bool = False
    train_detr: bool = True

    # precompute parameters
    use_precompute: bool = False
    precompute_path: Optional[str] = None
    precompute_batch_size: Optional[int] = None

    feedback_detr_to_llm: bool = False

    lora: bool = False
    lora_rank: int = 256
    lora_alpha: int = 512
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )