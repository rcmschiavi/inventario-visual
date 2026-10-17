"""Vocabulário de classes do detector (rótulo em português -> prompts em inglês).

O YOLO-World entende melhor nomes em inglês (o codificador de texto CLIP foi
treinado majoritariamente em inglês). Por isso o arquivo de classes separa o
**rótulo** que aparece no catálogo do **prompt** enviado ao modelo::

    # rótulo: prompt[, outro prompt]
    monitor: computer monitor
    cadeira: chair, office chair
    garrafa                        # sem ":" -> o próprio rótulo é o prompt

Vários prompts para o mesmo rótulo aumentam a chance de detecção; todos são
convertidos de volta para o rótulo na saída.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Lista ordenada de prompts e o rótulo de cada um (mesmo índice)."""

    prompts: tuple[str, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.prompts) != len(self.labels):
            raise ValueError("prompts e labels devem ter o mesmo tamanho")
        if not self.prompts:
            raise ValueError("vocabulário vazio")

    def label_for(self, class_index: int) -> str:
        return self.labels[class_index]

    def prompt_for(self, class_index: int) -> str:
        return self.prompts[class_index]

    @property
    def unique_labels(self) -> list[str]:
        return list(dict.fromkeys(self.labels))

    @classmethod
    def parse(cls, lines: Iterable[str]) -> Vocabulary:
        prompts: list[str] = []
        labels: list[str] = []
        for n, raw in enumerate(lines, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            label, sep, rest = line.partition(":")
            label = label.strip()
            if not label:
                raise ValueError(f"linha {n}: rótulo vazio")
            alternatives = [p.strip() for p in rest.split(",")] if sep else [label]
            for prompt in alternatives:
                # Ignora prompts vazios e repetidos (o primeiro rótulo vence).
                if prompt and prompt not in prompts:
                    prompts.append(prompt)
                    labels.append(label)
        return cls(tuple(prompts), tuple(labels))

    @classmethod
    def from_file(cls, path: Path) -> Vocabulary:
        return cls.parse(path.read_text(encoding="utf-8").splitlines())
