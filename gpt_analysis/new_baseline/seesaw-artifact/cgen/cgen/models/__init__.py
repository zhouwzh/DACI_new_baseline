import transformers

from cgen.models.base import PinnedModelWeights
from cgen.models.llama import DistLlamaForCausalLM

model_classes = {
    "llama": DistLlamaForCausalLM,
}


def create_model(model_config: transformers.PretrainedConfig, *args, **kwargs):
    model_cls = model_classes.get(model_config.model_type, None)
    if model_cls is None:
        raise ValueError(f"Model type {model_config.model_type} is not supported")
    return model_cls(model_config, *args, **kwargs)
