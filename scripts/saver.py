import gradio as gr
from modules import images
from modules.shared import opts
from modules import scripts_postprocessing
from modules.ui_components import FormRow



class ScriptPostprocessingAtlas(scripts_postprocessing.ScriptPostprocessing):
    name = "Saver"
    order = 9000

    def ui(self):
        with FormRow():
            with gr.Column():
                enable = gr.Checkbox(False, label="Enable Saver")

        return {
            "enable": enable,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable):
        if not enable:
            return
        
        images.save_image(pp.image,basename= "Original" ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info) 
