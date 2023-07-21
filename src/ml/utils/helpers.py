import importlib
import logging
import os
import pkgutil

import torch

from ml.engines.system import TweetSystem
from ml.utils.constants import EXPERIMENTS_DIR
from ml.utils.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


# get the best model checkpoint path and the version path for a specific version of a model
def get_model_at_version(model_name: str, version: str, model_type: str = "best-loss"):
    if not os.path.exists(os.path.join(EXPERIMENTS_DIR, model_name)):
        raise ValueError(
            f"Model {model_name} does not exist. Please select an existing model or train a new one \
            with `python train.py`"
        )

    if not os.path.exists(os.path.join(EXPERIMENTS_DIR, model_name, version)):
        raise ValueError(f"Version {version} does not exist. Please select a valid version")

    version_path = os.path.join(EXPERIMENTS_DIR, model_name, version)
    checkpoint_path = os.path.join(version_path, "checkpoints")
    model_path = None
    found = False
    for file in os.listdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, file)
        if model_type in file:
            model_path = os.path.join(checkpoint_path, file)
            found = True
            break

    if not found:
        logger.warning(f"Could not find a model checkpoint with type {model_type} in {checkpoint_path}")

    logger.info(f"Loading model from path {model_path} and from version {version_path}")
    return version_path, model_path


# get a model class by its class name
def get_model_by_name(class_name: str):
    models_module = "ml.models"
    parent_module = importlib.import_module(models_module)

    for loader, name, is_pkg in pkgutil.walk_packages(parent_module.__path__):
        if not is_pkg:
            try:
                module = importlib.import_module(f"{models_module}.{name}")
                model = getattr(module, class_name, None)
                if model is not None:
                    return model
            except (ImportError, AttributeError):
                continue
    raise ImportError(f"Cannot find class {class_name} in module {models_module}")


# get a trained model and its label encoder from a specific version of a model
def load_model_at_version(model_name: str, version: str, model_type: str = "best-loss"):
    version_path, model_path = get_model_at_version(model_name=model_name, version=version, model_type=model_type)
    checkpoint = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    model_class = get_model_by_name(model_name)

    tokenizer = Tokenizer()
    tokenizer.load(file_path=version_path)

    embeddings_matrix = checkpoint["state_dict"]["model.embedding.weight"].cpu()
    model = model_class(embeddings_matrix)

    system = TweetSystem.load_from_checkpoint(
        checkpoint_path=model_path,
        model=model,
        tokenizer=tokenizer,
        strict=False,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    logger.info(f"Successfully loaded {model_name} from {version}")

    return system
