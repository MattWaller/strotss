import sys
import torch
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import cog
from strotss import run_strotss


class Predictor(cog.Predictor):
    def setup(self):
        device = torch.device("cuda:0")
        print("Model loaded!")

    @cog.input(
        "content",
        type=Path,
        help="content image",
    )
    @cog.input(
        "style",
        type=Path,
        help="style image",
    )
    @cog.input(
        "content_weight", 
        type=float, 
        default=1.0, 
        help="how much the content is preserved (high->more close to original)"
    )
    def predict(self, content, style, content_weight):
        with torch.no_grad():
            print("Transferring Style :)")
            result_im = run_strotss(content, style, content_weight)
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        result_im.save(str(out_path))
        return out_path