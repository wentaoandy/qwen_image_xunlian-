from dataclasses import dataclass
from pathlib import Path
from PIL import Image


@dataclass
class Rendered3DConditions:
    """A container for 2D images rendered from one 3D asset.

    This file does not define a 3D encoder.
    It defines the first baseline: convert 3D information into image conditions.
    """

    depth: Path | None = None
    normal: Path | None = None
    edge: Path | None = None
    view_front: Path | None = None
    view_side: Path | None = None

    def load_images(self):
        images = []
        names = []
        for name in ("depth", "normal", "edge", "view_front", "view_side"):
            path = getattr(self, name)
            if path is not None:
                images.append(Image.open(path).convert("RGBA"))
                names.append(name)
        return names, images


def future_3d_token_interface(num_tokens: int, hidden_dim: int = 4096):
    """Document the target shape for a future direct 3D-token encoder.

    A future encoder can output [B, N3D, 4096].
    The project must then define position encoding and block metadata for those tokens.
    """
    return ("B", num_tokens, hidden_dim)
