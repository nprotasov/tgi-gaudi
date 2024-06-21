import torch
import torch.distributed

from opentelemetry import trace
from typing import Optional
from transformers.models.gemma import GemmaTokenizerFast

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_gemma_modeling import (
    FlashGemmaForCausalLM,
    GemmaConfig,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)


class FlashGemma(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        dtype = torch.bfloat16 if dtype is None else dtype
        device = torch.device("hpu")

        tokenizer = GemmaTokenizerFast.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
            use_fast=True,
            from_slow=False,
        )

        config = GemmaConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize
        config.use_medusa = use_medusa


        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype)
        if config.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id, revision)

        model = FlashGemmaForCausalLM(config, weights)

        super(FlashGemma, self).__init__(
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.model.layers),
            num_kv_heads=model.model.num_key_value_heads,
            head_size=model.model.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
