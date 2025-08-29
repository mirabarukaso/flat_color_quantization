import gradio as gr
from pathlib import Path
from flat_color_cli import FlatColorizer
from color_transfer import ColorTransfer

def parse_ints(s):
    parts = [int(v) for v in s.split(',')]
    if len(parts) != 4:
        print(f"[ERROR] Argument error {s}")
        print("[INFO] Use default [4,4,5,15]")
        return [4,4,5,15]
    return parts

def process_image(input_image, n_colors, temperature, spatial_scale, sharpen_strength, block_size, upscale, upscale_model, denoise, color_transfer):
    if input_image is None:
        return None

    # Initialize FlatColorizer
    fc = FlatColorizer()

    # Convert input image to RGB if needed
    if input_image.shape[2] == 4:  # If image has alpha channel
        input_image = input_image[:, :, :3]

    path = Path(upscale_model)
    path_str = str(path.as_posix()).replace('"','')

    # Process image using FlatColorizer's flat_color_multi_scale
    processed_img = fc.flat_color_multi_scale(
        img_path=None,  # We'll pass the image directly
        img_rgb=input_image,  # Pass numpy array directly
        n_colors=n_colors,
        temperature=temperature,
        spatial_scale=spatial_scale,
        sharpen_strength=sharpen_strength,
        block_size=block_size,
        upscale=upscale,
        model_path=path_str,
        denoising=parse_ints(denoise) if denoise else None
    )
    
    if color_transfer != "None":
        ct = ColorTransfer()
        if color_transfer == "Mean":
            processed_img = ct.mean_std_transfer(img_arr_ref=input_image, img_arr_in=processed_img)
        elif color_transfer == "Lab":
            processed_img = ct.lab_transfer(img_arr_ref=input_image, img_arr_in=processed_img)
        elif color_transfer == "Pdf":
            processed_img = ct.pdf_transfer(img_arr_ref=input_image, img_arr_in=processed_img, regrain=False)
        elif color_transfer == "Pdf+Regrain":
            processed_img = ct.pdf_transfer(img_arr_ref=input_image, img_arr_in=processed_img, regrain=True)
    
    return processed_img

def denoise_enable(trigger):
    return gr.Textbox(interactive=trigger)

with gr.Blocks() as demo:
    gr.Markdown("# Flat Color Multi-Scale Image Processing")
    with gr.Row():
        with gr.Column():
            n_colors = gr.Slider(minimum=2, maximum=4096, value=256, step=128, label="Color Count (256-4096, more colors = more VRAM)")
            block_size = gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="Tiled Processing Block Size")
            upscale = gr.Slider(minimum=1.0, maximum=8.0, value=1.0, step=0.1, label="Super-Sampling Scale (1.0=disabled, max 8.0)")
            upscale_model = gr.Textbox(label="Upscale Model Path (.safetensors/.pth)", placeholder="Full Path to ESRGAN/RealESRGAN model", value="")
            color_transfer = gr.Dropdown(choices=["Mean", "Lab", "Pdf", "Pdf+Regrain", "None"], value="None", label="Image Color Transfer")
        with gr.Column():
            temperature = gr.Slider(minimum=1.0, maximum=5.0, value=2.0, step=0.1, label="Temperature (Soft Assignment)")
            spatial_scale = gr.Slider(minimum=40, maximum=500, value=60, step=10, label="Spatial Scale (40-500)")
            sharpen_strength = gr.Slider(minimum=0.0, maximum=3.0, value=0.0, step=0.05, label="Sharpen Strength (0=disabled)")
            denoise = gr.Textbox(label="Denoising Parameters [h,hColor,templateWindowSize,searchWindowSize]", placeholder="e.g., 4,4,5,15", value="4,4,5,15")
            denoise_toggle = gr.Checkbox(label="Enable Denoising", value=False)
    with gr.Row():
        process_button = gr.Button("Start Processing", variant="primary")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", type="numpy")
        with gr.Column():
            output_image = gr.Image(format="png", label="Output Image")

    denoise_toggle.change(
        fn=denoise_enable,
        inputs=[denoise_toggle],
        outputs=[denoise]
    )

    process_button.click(
        fn=process_image,
        inputs=[
            input_image,
            n_colors,
            temperature,
            spatial_scale,
            sharpen_strength,
            block_size,
            upscale,
            upscale_model,
            denoise,
            color_transfer
        ],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch()