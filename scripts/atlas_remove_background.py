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



class ScriptPostprocessingAtlasR(scripts_postprocessing.ScriptPostprocessing):
    name = "Atlas Background Removal"
    order = 9001
    model = None

    def ui(self):
        return {}

    def process(self, pp: scripts_postprocessing.PostprocessedImage):
        if "extension_atlas_remove_bg" not in pp.info:
            return

        pp.image=remove_background(pp.image)
        images.save_image(pp.image,basename= "transparent_" ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info) 
       


def remove_background(pil_image):
    # convert the PIL image to OpenCV format (numpy array)
    img = np.array(pil_image.convert('RGB')) 

    # Create a grayscale version of the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image: this will create a binary image where
    # white pixels are those that were greater than 254 and black pixels the rest.
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to fill in the found contours
    mask = np.zeros_like(thresh)

    # Draw white filled contours on the black mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Create a 4-channel image (RGBA) from the original and the mask
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask

    # Convert the image back to PIL format
    pil_image = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))

    return pil_image