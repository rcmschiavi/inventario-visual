"""Backends do modelo de visão-linguagem (VLM).

O processador fala com o modelo pelo protocolo :class:`Describer`: recebe um
recorte RGB e um prompt, devolve o texto gerado. Os testes usam um backend
falso; o backend real usa o ``transformers`` (instalado com o extra ``vlm``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .detectors import resolve_device

logger = logging.getLogger(__name__)

ImageRGB = NDArray[np.uint8]
Quantization = Literal["4bit", "8bit", "none"]


class Describer(Protocol):
    def describe(self, image: ImageRGB, prompt: str, max_new_tokens: int | None = None) -> str: ...


class TransformersVlm:
    """Qualquer VLM do Hugging Face compatível com ``AutoModelForImageTextToText``.

    Padrão do projeto: ``Qwen/Qwen3-VL-4B-Instruct`` quantizado em 4 bits (NF4),
    que ocupa ~3 GB de VRAM e cabe numa GPU de 6 GB junto com o YOLO-World.
    """

    def __init__(
        self,
        model_id: str,
        quantization: Quantization = "4bit",
        device: str = "auto",
        max_new_tokens: int = 200,
        cache_dir: Path | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise ImportError("Pacote 'transformers' não instalado. Rode:  uv sync --extra vlm") from exc

        self.device = resolve_device(device)
        self.max_new_tokens = max_new_tokens
        cuda = self.device.startswith("cuda")
        if cuda:
            # GPUs anteriores à série RTX 30 não têm bfloat16 eficiente: usa float16.
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        kwargs: dict[str, Any] = {"dtype": dtype, "cache_dir": cache_dir}
        quantized = cuda and quantization != "none"
        if quantized:
            # bitsandbytes só funciona com CUDA.
            if quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = self.device
        elif quantization != "none":
            logger.warning("Quantização %s requer GPU; carregando sem quantizar na CPU (lento)", quantization)

        modo = quantization if quantized else "sem quantização"
        logger.info("Carregando VLM %s (%s, %s)...", model_id, modo, self.device)
        self.model: Any = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
        if not quantized:
            self.model.to(self.device)
        self.processor: Any = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self._torch = torch
        if cuda:
            logger.info("VLM carregado: %.1f GB de VRAM", torch.cuda.memory_allocated() / 1e9)

    def describe(self, image: ImageRGB, prompt: str, max_new_tokens: int | None = None) -> str:
        from PIL import Image

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(image)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            # Desliga o modo "pensamento" em modelos que o têm (ex.: Qwen3.5);
            # templates que não usam esta variável simplesmente a ignoram.
            enable_thinking=False,
        ).to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False
            )
        generated = out[:, inputs["input_ids"].shape[1] :]
        return str(self.processor.batch_decode(generated, skip_special_tokens=True)[0]).strip()
