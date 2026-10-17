"""inventario-visual: captura de frames-chave da câmera do celular via WebRTC."""

from .keyframes import KeyframeSelector, KeyframeSelectorConfig
from .processors import FrameProcessor, Keyframe

__all__ = ["FrameProcessor", "Keyframe", "KeyframeSelector", "KeyframeSelectorConfig"]
__version__ = "0.1.0"
