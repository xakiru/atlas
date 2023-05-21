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

class ScriptPostprocessingAtlas(scripts_postprocessing.ScriptPostprocessing):
    name = "Atlas"
    order = 9000
    model = None

    def ui(self):
        with FormRow():
            with gr.Column():
                enable = gr.Checkbox(False, label="Enable Atlas")

            with gr.Column():
                with FormRow():
                    save_input = gr.Checkbox(False, label="Save Input")
                    save_atlas = gr.Checkbox(False, label="Save Atlas")
                    save_transparent = gr.Checkbox(False, label="Save Transparent Atlas")
                    forward_atlas = gr.Checkbox(True, label="Forward Atlas")
        return {
            "enable": enable,
            "save_input": save_input,
            "save_atlas": save_atlas,
            "save_transparent": save_transparent,
            "forward_atlas": forward_atlas,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, save_input, save_atlas, forward_atlas):
        if not enable:
            return


        if (save_input):
            images.save_image(pp.image,basename= "input" ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info) 
        
            

        image = cv2.cvtColor(np.array(pp.image), cv2.COLOR_RGB2BGR)
        
        pil_output=create_atlas(pp.image)

        if (save_atlas):
            images.save_image(pil_output,basename= "atlas" ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info) 

        if (forward_atlas):
            pp.image=pil_output

        if save_transparent:
            pil_image=remove_bg(pil_output)
            images.save_image(pil_image,basename= "trans" ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info) 


def remove_bg(pil_image):
    # convert the PIL image to OpenCV format (numpy array)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Create a grayscale version of the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image: this will create a binary image where
    # white pixels are those that were greater than 254 and black pixels the rest.
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

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



def create_atlas(pil_image):

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #kernel = np.ones((3, 3), np.uint8)
    #img_gray_eroded = cv2.erode(img_gray, kernel, iterations=1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #morphed = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)

    _, img_gray = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)




    img_gray=255-img_gray
    # Find connected components (blobs) in the image
    labels = measure.label(img_gray, connectivity=2, background=0)

    # Create a new image to draw the mask on
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    cmask = np.zeros_like(img_gray, dtype=np.uint8)
    general_mask = np.zeros_like(img_gray, dtype=np.uint8)

    sprites = []

    # Create a list of tuples, where each tuple contains a label and the corresponding area
    blobs = []
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros_like(img_gray, dtype=np.uint8)
        labelMask[labels == label] = 255
        contours, _ = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = sum(cv2.contourArea(contour) for contour in contours)
        blobs.append((label, area))

    # Sort the blobs by area
    blobs.sort(key=lambda x: x[1], reverse=True)

    # Iterate over the sorted blobs
    for label, _ in blobs:
        # If this is the background label, ignore it
        if label == 0:
            continue

        # Construct the label mask
        labelMask = np.zeros_like(img_gray, dtype=np.uint8)
        labelMask[labels == label] = 255

        # Find contours in the label mask
        contours, _ = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # If the contour area is large enough, draw it on the mask
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:  # set this as per your requirement
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image, (x-4, y-4), (x + w+4, y + h+4), (255, 255, 255), 2)
                # Extract the ROI from the original image
                ROI = image[y-4:y+h+4, x-4:x+w+4]
                if ROI.shape[0] == 0 or ROI.shape[1] == 0:
                    continue
                hh, ww = ROI.shape[:2]

                
                roi_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                roi_thresh = cv2.threshold(roi_gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                kernel = np.ones((3,3), np.uint8)
                thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)
                roi_cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                big_contour = max(roi_cnts, key=cv2.contourArea)

                roi_mask = np.zeros_like(thresh)  
                cv2.drawContours(roi_mask, [big_contour], -1, (255, 0, 255), thickness=cv2.FILLED) 


                roi_mask2 = np.zeros_like(thresh) 
                cv2.drawContours(roi_mask2, [big_contour], -1, (255, 255, 255), thickness=cv2.FILLED) 
                kernel2 = np.ones((5, 5), np.uint8)
                roi_mask2 = cv2.erode(roi_mask2, kernel2)
                roi_mask2 = cv2.GaussianBlur(roi_mask2, (3,3), 0)


                clean_roi = cv2.bitwise_and(ROI, ROI, mask=roi_mask2)
              
                inversed_roi_mask=255-roi_mask
                background = np.full_like(ROI, (255,255,255))
                empty_background = cv2.bitwise_and(background, background, mask=inversed_roi_mask)

                result_without_alpha = cv2.bitwise_or(clean_roi, empty_background)

                result_with_alpha = cv2.cvtColor(result_without_alpha, cv2.COLOR_BGR2BGRA)

                #if transparency :
                #    result_with_alpha[..., 3] = roi_mask

                sprites.insert(0,result_without_alpha)

    hh, ww = image.shape[:2]

    # Define the tile size
    tile_width = tile_height = hh

    # Create an output image with a transparent background, of the size of the atlas
    #output = np.zeros((tile_height, tile_width * len(sprites), 4), dtype=np.uint8)
    output = np.full((tile_height, tile_width * len(sprites), 3), [255, 255, 255], dtype=np.uint8)

    # Position of the image in the output image
    # Position of the image in the output image
    # Position of the image in the output image
    x_offset = 0

    # Iterate over the images and add them to the output image
    for image in sprites:
        # The size of this image
        height, width = image.shape[:2]

        # Calculate the y-coordinate to place the image at the bottom of the tile
        y_offset = tile_height - height

        # Calculate the x-coordinate to place the image at the center of the tile
        x_center_offset = x_offset + (tile_width - width) // 2

        # Put the image on the output
        output[y_offset:y_offset+height, x_center_offset:x_center_offset+width] = image

        # Shift the x offset
        x_offset += tile_width



    #b, g, r, a = cv2.split(output)
    #pil_output = Image.fromarray(cv2.merge((r, g, b)))

    #pil_output = Image.fromarray(cv2.merge((r, g, b, a)))
    #output=cv2.cvtColor(output, cv2.COLOR_RGBA2RGB)
    #pil_output=Image.fromarray(output)

    pil_output=Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    return pil_output



class Script(scripts.Script):  

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "Atlas script"


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


        proc = process_images(p)


        # rotate and flip each image in the processed images
        # use the save_images method from images.py to save
        # them.

        return_images=[]
        for i in range(len(proc.images)):
            

            if (save_input):
                images.save_image(proc.images[i], path=p.outpath_samples,basename= str(proc.seed)+"-input" ,  extension=opts.samples_format, info= proc.info) 


            pil_output=create_atlas(proc.images[i])
            
            if (save_atlas):
                images.save_image(pil_output, path=p.outpath_samples,basename= str(proc.seed)+"-atlas" ,  extension=opts.samples_format, info= proc.info) 

            

            trans_output=remove_bg(pil_output)

            if (save_transparent):
                images.save_image(trans_output, path=p.outpath_samples,basename= str(proc.seed)+"-trans" ,  extension=opts.samples_format, info= proc.info) 

            return_images.append(trans_output)
            #images.save_image(pil_output, p.outpath_samples, "trans_" +str(proc.seed) + "_" + str(i), proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)
            
            #return_images.append(pil_output)

        proc.images=return_images
        return proc