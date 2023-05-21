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

class ScriptPostprocessingFirstAtlas(scripts_postprocessing.ScriptPostprocessing):
    name = "FirstAtlas"
    order = 9000

    def ui(self):
        with FormRow():
            with gr.Column():
                enable = gr.Checkbox(False, label="Enable Saver")
                animate = gr.Checkbox(False, label="Animate")

        return {
            "enable": enable,
            "animate": animate,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable):
        print("ScriptPostprocessingFirstAtlas");
        if not enable:
            return

class ScriptPostprocessingSecondAtlas(scripts_postprocessing.ScriptPostprocessing):
    name = "SecondAtlas"
    order = 9000

    def ui(self):
        with FormRow():
            with gr.Column():
                enable = gr.Checkbox(False, label="Enable Saver")
                animate = gr.Checkbox(False, label="Animate")

        return {
            "enable": enable,
            "animate": animate,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable):
        print("ScriptPostprocessingFirstAtlas");
        if not enable:
            return



class Script(scripts.Script):  

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "Atlas Script"


    def show(self, is_img2img):
        return is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        save_input = gr.Checkbox(True, label="Save Input")
        save_atlas = gr.Checkbox(True, label="Save Atlas")
        save_transparent = gr.Checkbox(True, label="Save Transparent")
        return [save_input, save_atlas, save_transparent]


    def run(self, p, save_input, save_atlas, save_transparent):
        # function which takes an image from the Processed object, 
        # and the angle and two booleans indicating horizontal and
        # vertical flips from the UI, then returns the 
        # image rotated and flipped accordingly
        
        # If overwrite is false, append the rotation information to the filename
        # using the "basename" parameter and save it in the same directory.
        # If overwrite is true, stop the model from saving its outputs and
        # save the rotated and flipped images instead.
        basename = ""
        #p.do_not_save_samples = True

        print("Atlas Script1")
        proc = process_images(p)
        print("Atlas Script2")

        # rotate and flip each image in the processed images
        # use the save_images method from images.py to save
        # them.

        return proc