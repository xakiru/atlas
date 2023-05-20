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
from modules.ui_components import FormRow



class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Remove Magenta"
    order = 9001
    model = None

    def ui(self):
        return {}

    def process(self, pp: scripts_postprocessing.PostprocessedImage):
        if "magenta" in pp.info:
            print(pp.info["magenta"])
        else:
            return

        print(pp.info)
        # Convert to NumPy array
        img = pp.image

        # Ensure the image has an alpha channel
        img = img.convert("RGBA")

        # Get the color of the pixel at (0, 0) - assuming this is the background
        background_color = img.getpixel((0, 0))

        # Get the image data
        data = img.getdata()

        # Create a new image data
        new_data = []
        for item in data:
            # Change all white (also shades of whites)
            # pixels to transparent
            if item[0] in list(range(200, 256)):
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
                
        # Update image data
        img.putdata(new_data)

        # If you want to save the image, you can do so with:
        # new_image.save('output.png')
        pp.image=img
        

        images.save_image(img,basename= "final_" ,path=opts.outdir_save,  extension=opts.samples_format, info= pp.info) 
       


