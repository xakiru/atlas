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
    name = "Remove Magenta"
    order = 9001
    model = None

    def ui(self):
        return {}

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, save_intermediate, save_magenta, forward_magenta,remove_magenta):
        if not pp.info["magenta"]:
            return


        # Convert to NumPy array
        data = np.array(pp.image)

        # Create a new 4-channel image (RGBA) with the same shape as the original image
        new_data = np.zeros(data.shape, dtype=np.uint8)

        # Copy the RGB values and alpha channel
        new_data[..., :3] = data[..., :3]
        new_data[..., 3] = data[..., 3]

        # Define the magenta color
        magenta = np.array([255, 0, 255, 255], dtype=np.uint8)

        # Find where the magenta pixels are
        magenta_pixels = (data == magenta).all(axis=-1)

        # Set alpha to 0 (transparent) where the image is magenta
        new_data[magenta_pixels] = [255, 0, 255, 0]  # RGB values won't matter here as alpha is set to 0

        # Convert back to PIL Image
        new_image = Image.fromarray(new_data, 'RGBA')

        # If you want to save the image, you can do so with:
        # new_image.save('output.png')
        pp.image=new_image
        

        images.save_image(new_image, pp.outpath_samples, "additional_", proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p) 
        


