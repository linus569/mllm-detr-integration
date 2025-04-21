import datetime
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import h5py
import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.processor import Processor
from model.model import VisionLanguageModel
from utils.config import DatasetConfig, ExperimentConfig
from utils.train_utils import build_train_dataloader, build_val_dataloader

log = logging.getLogger(__name__)


@torch.inference_mode()
def precompute_image_features(
    config: ExperimentConfig,
    model: VisionLanguageModel,
    dataloader: DataLoader,
    output_path: str,
    dataset_name: str,
):
    log.info("Precompute script is running")

    sample_batch = next(iter(dataloader))
    feature_dim = model.image_encoder(
        sample_batch["images"].to(config.device)
    ).shape[1:]
    # log.info(f"Feature dimension: {feature_dim}")

    dataset_size = len(dataloader.dataset)
    numpy_dtype = sample_batch["images"].cpu().numpy().dtype

    # precomputed_feat = torch.zeros(
    #     (len(dataloader.dataset),) + feature_dim, dtype=sample_batch["images"].dtype
    # )
    # log.info(f"Precomputed feature shape: {precomputed_feat.shape}")
    with h5py.File(output_path, "w") as f:
        if dataset_name in f:
            del f[dataset_name]

        dset = f.create_dataset(
            dataset_name,
            (dataset_size,) + feature_dim,
            dtype=numpy_dtype, #TODO: dtype from numpy
            compression="gzip",
            compression_opts=9,
            chunks=True,
        )

        dset.attrs["shape"] = (dataset_size,) + feature_dim
        dset.attrs["created"] = str(datetime.datetime.now())

        dataloader_tqdm = tqdm(
            dataloader,
            desc="Precomputing image features",
            total=len(dataloader),
            unit="batch",
        )

        batch_size = config.batch_size

        start_idx = 0
        for batch in dataloader_tqdm:
            images = batch["images"]#to(config.device)
            batch_size = images.shape[0]

            image_features = model.image_encoder(images)

            end_idx = start_idx + batch_size
            dset[start_idx:end_idx] = image_features.cpu()
            start_idx = end_idx

            del image_features
            #torch.cuda.empty_cache()

    log.info("Precomputation completed.")


@torch.inference_mode()
def validate_precomputed_features(
    config: ExperimentConfig,
    model: VisionLanguageModel,
    dataloader: DataLoader,
    file_path: str,
):
    log.info("Validating precomputed features")

    with h5py.File(file_path, "r") as f:
        precomputed_img = torch.from_numpy(f["precomputed_val_img"][:])

    dataloader_tqdm = tqdm(
        dataloader,
        desc="Validating precomputed image features",
        total=len(dataloader),
        unit="batch",
    )

    sample_batch = next(iter(dataloader_tqdm))
    feature_dim = model._get_image_features(
        sample_batch["images"].to(config.device), sample_batch["image_sizes"]
    ).shape[1:]
    log.info(f"Feature dimension: {feature_dim}")

    precomputed_feat = torch.zeros(
        (len(dataloader.dataset),) + feature_dim, dtype=sample_batch["images"].dtype
    )
    log.info(f"Precomputed feature shape: {precomputed_feat.shape}")

    start_idx = 0
    for batch in dataloader_tqdm:
        images = batch["images"].to(config.device)
        image_sizes = batch["image_sizes"]
        batch_size = images.shape[0]

        image_features = model._get_image_features(images, image_sizes)

        end_idx = start_idx + batch_size
        precomputed_feat[start_idx:end_idx] = image_features.cpu()

        # assert torch.allclose(
        #    precomputed_feat[start_idx:end_idx], torch.tensor(precomputed_img[start_idx:end_idx])
        # ), f"Precomputed features do not match for batch {start_idx} to {end_idx}"

        # Compare the newly computed features with the loaded ones
        current_computed = precomputed_feat[start_idx:end_idx]
        stored_features = precomputed_img[start_idx:end_idx]

        # Check shapes first
        if current_computed.shape != stored_features.shape:
            raise ValueError(
                f"Shape mismatch: {current_computed.shape} vs {stored_features.shape}"
            )

        # Use appropriate tolerance for floating point comparison
        if not torch.allclose(current_computed, stored_features, rtol=1e-5, atol=1e-5):
            # If assertion would fail, log the differences for debugging
            max_diff = torch.max(torch.abs(current_computed - stored_features))
            mean_diff = torch.mean(torch.abs(current_computed - stored_features))
            log.error(
                f"Features don't match for batch {start_idx}:{end_idx}. Max diff: {max_diff}, Mean diff: {mean_diff}"
            )
            raise ValueError(
                f"Precomputed features do not match for batch {start_idx} to {end_idx}"
            )

        start_idx = end_idx

    dataloader_tqdm.close()

    log.info("Validation completed.")
    return precomputed_feat


@hydra.main(config_path="../conf", config_name="train", version_base=None)
def run_script(config: ExperimentConfig):
    log.info(OmegaConf.to_yaml(config))

    device = torch.device("cuda" if torch.cuda.is_available() else config.device)
    log.info(f"Using device: {device}")

    assert config.image_encoder.name == "siglip", "Only siglip is supported for now"

    processor = Processor.from_config(
        config, add_special_tokens=config.add_special_tokens
    )
    model = VisionLanguageModel(
        config=config,
        image_token_index=processor.image_token_index,
        num_new_tokens=(
            len(processor.special_tokens) if config.add_special_tokens else 0
        ),
        tokenizer_size=processor.loaded_tokenizer_len,
        initializers=(
            processor.special_tokens_initializer if config.add_special_tokens else None
        ),
        do_init=config.add_special_tokens,
        # TODO: fix, 00 is hardcoded
        query_tokens_id=(
            processor.tokenizer.encode("<query00/>")
            if config.num_query_tokens > 0
            else None
        ),
    ).to(device)

    if not config.debug:
        model = torch.compile(model)  # 2.3 it/s without -> 4.5 it/s with

    train_dataloader = build_train_dataloader(
        config=config,
        processor=processor,
        subset_size=config.num_samples,
    )
    val_dataloader = build_val_dataloader(
        config=config,
        processor=processor,
        subset_size=config.val_num_samples,
    )
 
    output_file = "precomputed_img.hdf5"
    precompute_image_features(config, model, train_dataloader, output_path=output_file, dataset_name="precomputed_train_img")
    precompute_image_features(config, model, val_dataloader, output_path=output_file, dataset_name="precomputed_val_img")
    

    # with h5py.File("mytestfile.hdf5", "w") as f:
    #     dset_train = f.create_dataset(
    #         "precomputed_train_img",
    #         data=precomputed_train_img,
    #         compression="gzip",
    #         compression_opts=9,
    #     )
    #     dset_val = f.create_dataset(
    #         "precomputed_val_img",
    #         data=precomputed_val_img,
    #         compression="gzip",
    #         compression_opts=9,
    #     )
    log.info("Precomputed image features saved to mytestfile.hdf5")

    # validate_precomputed_features(
    #     config, model, val_dataloader, "mytestfile.hdf5"
    # )
    # log.info("Precomputed image features validated")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cs = ConfigStore.instance()
    cs.store(name="ExperimentConfig", node=ExperimentConfig)
    cs.store(name="DatasetConfig", group="dataset", node=DatasetConfig)
    # OmegaConf.register_new_resolver("models_dir", lambda: MODELS_DIR)
    OmegaConf.register_new_resolver(
        "ifel", lambda flag, val_true, val_false: val_true if flag else val_false
    )
    # OmegaConf.register_new_resolver("project_dir", lambda: PROJECT_DIR)

    # import model
    # from model import img_encoder, txt_encoder, txt_decoder, detector
    # ModelRegistry.init_registries([model, img_encoder, txt_encoder, txt_decoder, detector])

    run_script()

# 131 debug
# 179 no debug
#60s no debug mps device