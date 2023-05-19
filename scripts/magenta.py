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


