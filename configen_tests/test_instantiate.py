from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf
from torch import Tensor

cfg = {} 

full_class = f"gen.configen_tests.utils.data.dataset.TensorDatasetConf"
schema = OmegaConf.structured(get_class(full_class))
cfg = OmegaConf.merge(schema, cfg)
obj = instantiate(cfg, tensors=(Tensor([1])))

print(obj)
