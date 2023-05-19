import os

from modules import scripts_postprocessing, devices, scripts
import gradio as gr

from modules.ui_components import FormRow

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np




class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Pixelization2"
    order = 10000
    model = None

    def ui(self):
        with FormRow():
            with gr.Column():
                with FormRow():
                    enable = gr.Checkbox(False, label="Enable pixelization")
                    upscale_after = gr.Checkbox(False, label="Keep resolution")

            with gr.Column():
                pixel_size = gr.Slider(minimum=1, maximum=16, step=1, label="Pixel size", value=4, elem_id="pixelization2_pixel_size")

        return {
            "enable": enable,
            "upscale_after": upscale_after,
            "pixel_size": pixel_size,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, upscale_after, pixel_size):
        if not enable:
            return

        print(pp)
        print(pp.image)
        pp.info["Pixelization2 pixel size"] = pixel_size

