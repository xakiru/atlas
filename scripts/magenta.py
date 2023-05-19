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
                    transparency = gr.Checkbox(False, label="transparency")

            with gr.Column():
                outline_size = gr.Slider(minimum=1, maximum=8, step=1, label="Pixel size", value=4, elem_id="outline_size")

        return {
            "transparency": transparency,
            "outline_size": outline_size,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, transparency, outline_size):
        if not transparency:
            return

        print("pp")
        print(pp.image)

        image = cv2.cvtColor(np.array(pp.image), cv2.COLOR_RGB2BGR)


        print("back pp")
        print(Image.fromarray(image))


        print("image")
        print(image)
        
            

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        #image_morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Create a kernel. The size of the kernel affects the operation; you may need to adjust this.
        kernel = np.ones((5,5), np.uint8)

        # Perform morphological closing
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours and remove small noise
        cnts ,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Find contours, obtain bounding box, extract and save ROI
        ROI_number = 0

        #mask = np.full_like(image, (0,0,0))
        #cv2.drawContours(mask,cnts, -1, (255,255,255), cv2.FILLED)

        sprites=[]
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x-4, y-4), (x + w+4, y + h+4), (255, 255, 255), 2)
            # Extract the ROI from the original image
            ROI = image[y-4:y+h+4, x-4:x+w+4]
            if ROI.shape[0] == 0 or ROI.shape[1] == 0:
                continue
            hh, ww = ROI.shape[:2]

            
            
            roi_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            roi_thresh = cv2.threshold(roi_gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # Morph open to remove noise
            #roi_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            #roi_morph = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, roi_kernel, iterations=1)
            kernel = np.ones((3,3), np.uint8)
            # Perform morphological closing
            thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)

          
            # Find contours and remove small noise
            roi_cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            big_contour = max(roi_cnts, key=cv2.contourArea)


            roi_mask = np.zeros_like(thresh)
            #roi_mask = np.zeros((hh,ww), dtype=np.uint8)   
            cv2.drawContours(roi_mask, [big_contour], -1, (255, 255, 255), thickness=cv2.FILLED) 
            #cv2.drawContours(roi_mask, roi_cnts, -1, (255,255,255), cv2.FILLED)
            
          
            result1 = cv2.bitwise_and(ROI, ROI, mask=roi_mask)



            roi_mask2 = np.zeros_like(thresh)
            #roi_mask = np.zeros((hh,ww), dtype=np.uint8)   
            cv2.drawContours(roi_mask2, [big_contour], -1, (255, 255, 255), thickness=cv2.FILLED) 
            #cv2.drawContours(roi_mask, roi_cnts, -1, (255,255,255), cv2.FILLED)
            # Creating kernel
            kernel2 = np.ones((4, 4), np.uint8)
            roi_mask2 = cv2.erode(roi_mask2, kernel2)
            roi_mask2 = cv2.GaussianBlur(roi_mask2, (5,5), 0)


            result2 = cv2.bitwise_and(ROI, ROI, mask=roi_mask2)

          
            inversed_roi_mask=255-roi_mask
            magenta = np.full_like(ROI, (255,0,255))
            background = cv2.bitwise_and(magenta, magenta, mask=inversed_roi_mask)

            test = cv2.bitwise_or(result1, background)

            # Create a 4-channel image (3 for RGB and 1 for alpha)
            result_with_alpha = cv2.cvtColor(test, cv2.COLOR_BGR2BGRA)

            if transparency :
                result_with_alpha[..., 3] = roi_mask

            sprites.append(result_with_alpha)
            

            #sprite_path = f'{output_folder}/{ROI_number}.png'
            #cv2.imwrite(sprite_path, result_with_alpha)

            ROI_number += 1

            b, g, r, a = cv2.split(result_with_alpha)

            
            #imshow("",result_with_alpha)
            #images.save_image(Image.fromarray(cv2.merge((r, g, b, a))), p.outpath_samples, basename + "_" + str(ROI_number), proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)

            #images.save_image(Image.fromarray(cv2.merge((r, g, b, a))), p.outpath_samples, basename + "_" + str(ROI_number), proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p) 
            #raf.append(Image.fromarray(cv2.merge((r, g, b, a))))
            output=result_with_alpha #Image.fromarray(cv2.merge((r, g, b, a)))
        #raf = img
        #pp.image=output

        hh, ww = image.shape[:2]

        # Define the tile size
        tile_width = tile_height = hh

        # Create an output image with a transparent background, of the size of the atlas
        output = np.zeros((tile_height, tile_width * len(sprites), 4), dtype=np.uint8)
        output = np.full((tile_height, tile_width * len(sprites), 4), [255, 0, 255, 255], dtype=np.uint8)

        # Position of the image in the output image
        x_offset = 0

        # Iterate over the images and add them to the output image
        for image in sprites:
            # The size of this image
            height, width = image.shape[:2]

            # Check if image is not RGBA and convert it
            if image.shape[2] == 3:  # if the image has no alpha channel
                alpha_channel = np.ones((height, width, 1), dtype=image.dtype) * 255  # creating a new alpha channel with all values set to 255
                image = np.concatenate((image, alpha_channel), axis=2)  # add the new alpha channel to the image

            # Calculate the y-coordinate to place the image at the bottom of the tile
            y_offset = tile_height - height

            # Calculate the x-coordinate to place the image at the center of the tile
            x_center_offset = x_offset + (tile_width - width) // 2

            # Define the area for this sprite on the output image
            sprite_area = output[y_offset:y_offset+height, x_center_offset:x_center_offset+width]

            # Convert the alpha channel to a weight between 0 and 1
            alpha = image[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha

            # Blend the sprite with the sprite_area using the alpha channel as weight
            blended = cv2.addWeighted(image, alpha, sprite_area, alpha_inv, 0)

            # Put the blended image back onto the output image
            output[y_offset:y_offset+height, x_center_offset:x_center_offset+width] = blended

            # Shift the x offset
            x_offset += tile_width

        output=cv2.cvtColor(output, cv2.COLOR_RGBA2RGB)
        #output = cv2.cvtColor(output.astype('uint8'), cv2.COLOR_RGBA2BGRA)


        print("output")
        print(output)

        print(Image.fromarray(output))

        pp.image=Image.fromarray(output)
        
        #pp.info["Magenta pixel size"] = pixel_size

