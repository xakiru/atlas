import modules.scripts as scripts
import gradio as gr
import os
import cv2
from PIL import Image
import numpy as np
from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from skimage import measure
from modules import scripts_postprocessing


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Transparency & Outline"
    order = 10000
    def ui(self):
        transparency = gr.Checkbox(True, label="transparency")
        outline_size = gr.Slider(minimum=0, maximum=8, step=1, Outline="Pixel size", value=1, elem_id="outline_size")
        return [transparency, outline_size]

    def process(self, pp: scripts_postprocessing.PostprocessedImage, transparency, outline_size):
        if not transparency:
            return

        print(pp.image)
        print(pp.images)
        pp.image = pp.image.resize((pp.image.width * 4 // pixel_size, pp.image.height * 4 // pixel_size))

        pp.info["Magenta"] = True


import os

from modules import scripts_postprocessing, devices, scripts
import gradio as gr

from modules.ui_components import FormRow

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np




class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Magenta"
    order = 9000
    model = None

    def ui(self):
        with FormRow():
            with gr.Column():
                with FormRow():
                    enable = gr.Checkbox(False, label="enable")

            with gr.Column():
                outline_size = gr.Slider(minimum=1, maximum=8, step=1, label="Pixel size", value=4, elem_id="outline_size")

        return {
            "transparency": transparency,
            "outline_size": outline_size,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, outline_size):
        if not enable:
            return


