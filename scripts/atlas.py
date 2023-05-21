
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
import os

from modules import scripts_postprocessing, devices, scripts
import gradio as gr

from modules.ui_components import FormRow

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from pixelization.models.networks import define_G
import pixelization.models.c2pGen

pixelize_code = [
    233356.8125, -27387.5918, -32866.8008, 126575.0312, -181590.0156,
    -31543.1289, 50374.1289, 99631.4062, -188897.3750, 138322.7031,
    -107266.2266, 125778.5781, 42416.1836, 139710.8594, -39614.6250,
    -69972.6875, -21886.4141, 86938.4766, 31457.6270, -98892.2344,
    -1191.5887, -61662.1719, -180121.9062, -32931.0859, 43109.0391,
    21490.1328, -153485.3281, 94259.1797, 43103.1992, -231953.8125,
    52496.7422, 142697.4062, -34882.7852, -98740.0625, 34458.5078,
    -135436.3438, 11420.5488, -18895.8984, -71195.4141, 176947.2344,
    -52747.5742, 109054.6562, -28124.9473, -17736.6152, -41327.1562,
    69853.3906, 79046.2656, -3923.7344, -5644.5229, 96586.7578,
    -89315.2656, -146578.0156, -61862.1484, -83956.4375, 87574.5703,
    -75055.0469, 19571.8203, 79358.7891, -16501.5000, -147169.2188,
    -97861.6797, 60442.1797, 40156.9023, 223136.3906, -81118.0547,
    -221443.6406, 54911.6914, 54735.9258, -58805.7305, -168884.4844,
    40865.9609, -28627.9043, -18604.7227, 120274.6172, 49712.2383,
    164402.7031, -53165.0820, -60664.0469, -97956.1484, -121468.4062,
    -69926.1484, -4889.0151, 127367.7344, 200241.0781, -85817.7578,
    -143190.0625, -74049.5312, 137980.5781, -150788.7656, -115719.6719,
    -189250.1250, -153069.7344, -127429.7891, -187588.2500, 125264.7422,
    -79082.3438, -114144.5781, 36033.5039, -57502.2188, 80488.1562,
    36501.4570, -138817.5938, -22189.6523, -222146.9688, -73292.3984,
    127717.2422, -183836.3750, -105907.0859, 145422.8750, 66981.2031,
    -9596.6699, 78099.4922, 70226.3359, 35841.8789, -116117.6016,
    -150986.0156, 81622.4922, 113575.0625, 154419.4844, 53586.4141,
    118494.8750, 131625.4375, -19763.1094, 75581.1172, -42750.5039,
    97934.8281, 6706.7949, -101179.0078, 83519.6172, -83054.8359,
    -56749.2578, -30683.6992, 54615.9492, 84061.1406, -229136.7188,
    -60554.0000, 8120.2622, -106468.7891, -28316.3418, -166351.3125,
    47797.3984, 96013.4141, 71482.9453, -101429.9297, 209063.3594,
    -3033.6882, -38952.5352, -84920.6719, -5895.1543, -18641.8105,
    47884.3633, -14620.0273, -132898.6719, -40903.5859, 197217.3750,
    -128599.1328, -115397.8906, -22670.7676, -78569.9688, -54559.7070,
    -106855.2031, 40703.1484, 55568.3164, 60202.9844, -64757.9375,
    -32068.8652, 160663.3438, 72187.0703, -148519.5469, 162952.8906,
    -128048.2031, -136153.8906, -15270.3730, -52766.3281, -52517.4531,
    18652.1992, 195354.2188, -136657.3750, -8034.2622, -92699.6016,
    -129169.1406, 188479.9844, 46003.7500, -93383.0781, -67831.6484,
    -66710.5469, 104338.5234, 85878.8438, -73165.2031, 95857.3203,
    71213.1250, 94603.1094, -30359.8125, -107989.2578, 99822.1719,
    184626.3594, 79238.4531, -272978.9375, -137948.5781, -145245.8125,
    75359.2031, 26652.7930, 50421.4141, 60784.4102, -18286.3398,
    -182851.9531, -87178.7969, -13131.7539, 195674.8906, 59951.7852,
    124353.7422, -36709.1758, -54575.4766, 77822.6953, 43697.4102,
    -64394.3438, 113281.1797, -93987.0703, 221989.7188, 132902.5000,
    -9538.8574, -14594.1338, 65084.9453, -12501.7227, 130330.6875,
    -115123.4766, 20823.0898, 75512.4922, -75255.7422, -41936.7656,
    -186678.8281, -166799.9375, 138770.6250, -78969.9531, 124516.8047,
    -85558.5781, -69272.4375, -115539.1094, 228774.4844, -76529.3281,
    -107735.8906, -76798.8906, -194335.2812, 56530.5742, -9397.7529,
    132985.8281, 163929.8438, -188517.7969, -141155.6406, 45071.0391,
    207788.3125, -125826.1172, 8965.3320, -159584.8438, 95842.4609,
    -76929.4688
]

path_checkpoints = os.path.join(scripts.basedir(), "checkpoints")
path_pixelart_vgg19 = os.path.join(path_checkpoints, "pixelart_vgg19.pth")
path_160_net_G_A = os.path.join(path_checkpoints, "160_net_G_A.pth")
path_alias_net = os.path.join(path_checkpoints, "alias_net.pth")


class TorchHijackForC2pGen:
    def __getattr__(self, item):
        if item == 'load':
            return self.load

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    def load(self, filename, *args, **kwargs):
        if filename == "./pixelart_vgg19.pth":
            filename = path_pixelart_vgg19

        return torch.load(filename, *args, **kwargs)


pixelization.models.c2pGen.torch = TorchHijackForC2pGen()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.G_A_net = None
        self.alias_net = None

    def load(self):
        os.makedirs(path_checkpoints, exist_ok=True)

        missing = False

        if not os.path.exists(path_pixelart_vgg19):
            print(f"Missing {path_pixelart_vgg19} - download it from https://drive.google.com/uc?id=1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM")
            missing = True

        if not os.path.exists(path_160_net_G_A):
            print(f"Missing {path_160_net_G_A} - download it from https://drive.google.com/uc?id=1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az")
            missing = True

        if not os.path.exists(path_alias_net):
            print(f"Missing {path_alias_net} - download it from https://drive.google.com/uc?id=17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_")
            missing = True

        assert not missing, 'Missing checkpoints for pixelization - see console for doqwnload links.'

        with torch.no_grad():
            self.G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [0])
            self.alias_net = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02, [0])

            G_A_state = torch.load(path_160_net_G_A)
            for p in list(G_A_state.keys()):
                G_A_state["module." + str(p)] = G_A_state.pop(p)
            self.G_A_net.load_state_dict(G_A_state)

            alias_state = torch.load(path_alias_net)
            for p in list(alias_state.keys()):
                alias_state["module." + str(p)] = alias_state.pop(p)
            self.alias_net.load_state_dict(alias_state)


def process(img):
    ow, oh = img.size

    nw = int(round(ow / 4) * 4)
    nh = int(round(oh / 4) * 4)

    left = (ow - nw) // 2
    top = (oh - nh) // 2
    right = left + nw
    bottom = top + nh

    img = img.crop((left, top, right, bottom))

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return trans(img)[None, :, :, :]


def to_image(tensor, pixel_size, upscale_after):
    img = tensor.data[0].cpu().float().numpy()
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((img.size[0]//4, img.size[1]//4), resample=Image.Resampling.NEAREST)
    if upscale_after:
        img = img.resize((img.size[0]*pixel_size, img.size[1]*pixel_size), resample=Image.Resampling.NEAREST)

    return img




def create_gif(atlas_image, frame_width, frame_height):
    # Calculate the number of frames in each column
    num_frames = atlas_image.width // frame_width

    # Calculate the number of columns
    num_columns = atlas_image.width // frame_width

    # Create a new list of lists to hold vertically looped images for each column
    looped_images = []
    for col in range(num_columns):
        column_images = []
        for frame in range(num_frames):
            # Crop the current frame from the atlas image
            left = col * frame_width
            upper = frame * frame_height
            right = left + frame_width
            lower = upper + frame_height
            cropped_image = atlas_image.crop((left, upper, right, lower))

            # Append the cropped image to the column images list
            column_images.append(cropped_image)

        # Append the column images list to the looped images list
        looped_images.append(column_images)

    return looped_images

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

def concatenate_images(images):
    # Determine output dimensions
    output_width = max(image.width for image in images)
    output_height = sum(image.height for image in images)

    # Create a new blank image with transparent background
    output_image = Image.new('RGBA', (output_width, output_height), (0, 0, 0, 0))

    # Paste the images vertically
    y_offset = 0
    for image in images:
        output_image.paste(image, (0, y_offset))
        y_offset += image.height

    return output_image



def create_animation(pil_image):

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #kernel = np.ones((3, 3), np.uint8)
    #img_gray_eroded = cv2.erode(img_gray, kernel, iterations=1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #morphed = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)

    _, img_gray = cv2.threshold(img_gray, 224, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)




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

    max_contours = 2  # Maximum number of contours to consider
    count_contours = 0 ximum number of contours to consider
    used_contours = False
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
            if area > 1200:  # set this as per your requirement
                x, y, w, h = cv2.boundingRect(c)
                used_contours=True
                #cv2.rectangle(image, (x-4, y-4), (x + w+4, y + h+4), (255, 255, 255), 2)
                # Extract the ROI from the original image
                ROI = image[y-4:y+h+4, x-4:x+w+4]
                #ROI = image[y:y+h, x:x+w]
                
                if ROI.shape[0] == 0 or ROI.shape[1] == 0:
                    continue
                hh, ww = ROI.shape[:2]

                
                roi_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                roi_thresh = cv2.threshold(roi_gray, 224, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                kernel = np.ones((3,3), np.uint8)
                thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)
                roi_cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                big_contour = max(roi_cnts, key=cv2.contourArea)

                roi_mask = np.zeros_like(thresh)  
                cv2.drawContours(roi_mask, [big_contour], -1, (255, 255, 255), thickness=cv2.FILLED) 


                roi_mask2 = np.zeros_like(thresh) 
                cv2.drawContours(roi_mask2, [big_contour], -1, (255, 255, 255), thickness=cv2.FILLED) 
                kernel2 = np.ones((5, 5), np.uint8)
                roi_mask2 = cv2.erode(roi_mask2, kernel2)
                roi_mask2 = cv2.GaussianBlur(roi_mask2, (3,3), 0)


                clean_roi = cv2.bitwise_and(ROI, ROI, mask=roi_mask)
              
                inversed_roi_mask=255-roi_mask
                background = np.full_like(ROI, (255,255,255))
                empty_background = cv2.bitwise_and(background, background, mask=inversed_roi_mask)

                result_without_alpha = cv2.bitwise_or(clean_roi, empty_background)

                result_with_alpha = cv2.cvtColor(result_without_alpha, cv2.COLOR_BGR2BGRA)

                #if transparency :
                #    result_with_alpha[..., 3] = roi_mask

                #sprites.insert(0,result_without_alpha)
                sprites.insert(0,ROI)
            
            if used_contours :
                count_contours += 1
        if count_contours >= max_contours:
            break

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

    output3 = np.full((tile_height, tile_width * len(sprites), 3), [255, 255, 255], dtype=np.uint8)

    x_offset = 0

    # Iterate over the images and add them to the output image
    for image in sprites:
        # The size of this image
        height, width = image.shape[:2]

        stretched_width = width-16
        stretched_height = height+8
        stretched_image = cv2.resize(image, (stretched_width, stretched_height))

        # Calculate the y-coordinate to place the image at the bottom of the tile
        y_offset = tile_height - stretched_height

        # Calculate the x-coordinate to place the image at the center of the tile
        x_center_offset = x_offset + (tile_width - stretched_width) // 2

        # Put the image on the output
        output3[y_offset:y_offset+stretched_height, x_center_offset:x_center_offset+stretched_width] = stretched_image

        # Shift the x offset
        x_offset += tile_width

    output2 = np.full((tile_height, tile_width * len(sprites), 3), [255, 255, 255], dtype=np.uint8)

    x_offset = 0

    # Iterate over the images and add them to the output image
    for image in sprites:
        # The size of this image
        height, width = image.shape[:2]

        stretched_width = width+8
        stretched_height = height-16
        stretched_image = cv2.resize(image, (stretched_width, stretched_height))

        # Calculate the y-coordinate to place the image at the bottom of the tile
        y_offset = tile_height - stretched_height

        # Calculate the x-coordinate to place the image at the center of the tile
        x_center_offset = x_offset + (tile_width - stretched_width) // 2

        # Put the image on the output
        output2[y_offset:y_offset+stretched_height, x_center_offset:x_center_offset+stretched_width] = stretched_image

        # Shift the x offset
        x_offset += tile_width



    #b, g, r, a = cv2.split(output)
    #pil_output = Image.fromarray(cv2.merge((r, g, b)))

    #pil_output = Image.fromarray(cv2.merge((r, g, b, a)))
    #output=cv2.cvtColor(output, cv2.COLOR_RGBA2RGB)
    #pil_output=Image.fromarray(output)

    pil_output=Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    pil_output2=Image.fromarray(cv2.cvtColor(output2, cv2.COLOR_BGR2RGB))
    pil_output3=Image.fromarray(cv2.cvtColor(output3, cv2.COLOR_BGR2RGB))

    return [pil_output,pil_output2,pil_output3]

class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Atlas"
    order = 10000
    model = None

    def ui(self):
        with FormRow():
            with gr.Column():
                with FormRow():
                    enable = gr.Checkbox(True, label="Enable pixelization")
                    save_original = gr.Checkbox(True, label="Save original")
                    save_atlas = gr.Checkbox(True, label="Save atlas")

                    save_pixelization = gr.Checkbox(False, label="Save Pixelization")
                    save_transparent = gr.Checkbox(True, label="Save Trans Pixelization")
                    upscale_after = gr.Checkbox(False, label="Keep resolution")

            with gr.Column():
                pixel_size = gr.Slider(minimum=1, maximum=16, step=1, label="Pixel size", value=4, elem_id="pixelization_pixel_size")

        return {
            "enable": enable,
            "save_original": save_original,
            "save_atlas": save_atlas,
            "save_pixelization": save_pixelization,
            "save_transparent": save_transparent,
            "upscale_after": upscale_after,
            "pixel_size": pixel_size,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, save_original,save_atlas,save_pixelization,save_transparent,upscale_after, pixel_size):
        if not enable:
            return


        animated_images=create_animation(pp.image)
        pixel_images=[]
        trans_images=[]

        if (save_original):
            images.save_image(pp.image,basename= "original" ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info)
        if (save_atlas):
            images.save_image(remove_bg(animated_images[0]),basename= "atlas" ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info) 


        if self.model is None:
            model = Model()
            model.load()

            self.model = model

        self.model.to(devices.device)


        for image_index, image in enumerate(animated_images):

            
            image = image.resize((image.width * 4 // pixel_size, image.height * 4 // pixel_size))

            with torch.no_grad():
                in_t = process(image).to(devices.device)

                feature = self.model.G_A_net.module.RGBEnc(in_t)
                code = torch.asarray(pixelize_code, device=devices.device).reshape((1, 256, 1, 1))
                adain_params = self.model.G_A_net.module.MLP(code)
                my_images = self.model.G_A_net.module.RGBDec(feature, adain_params)
                out_t = self.model.alias_net(my_images)
                pixel_image = to_image(out_t, pixel_size=pixel_size, upscale_after=upscale_after)
                pixel_images.append(pixel_image)
                trans_images.append(remove_bg(pixel_image))

  
        self.model.to(devices.cpu)

        pixel_output=concatenate_images(pixel_images)
        trans_output=concatenate_images(trans_images)
        
        if (save_pixelization):
            images.save_image(pixel_output,basename= "pixel"  ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info) 

        if (save_transparent):
            images.save_image(trans_output,basename= "trans_pixel"  ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info) 

        pp.image=pixel_output
        pp.info["Pixelization pixel size"] = pixel_size

        animations = create_gif(pixel_output, 128, 128)
        column_index = 0

        for column in animations:
            # Process each column animation, which is a list of vertically looped images
            output_path = f'{opts.outdir_img2img_samples}_animation_{column_index}.gif'  # Specify the output path for the GIF file
            column[0].save(output_path, format='GIF', append_images=column[1:], save_all=True, duration=500, loop=0)
            column_index += 1

