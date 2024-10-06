import gradio as gr
import torch
from PIL import Image
import numpy as np
import argparse
import os
from datetime import datetime
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
import webbrowser


# Constants for directory paths
INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
WEIGHTS_DIR = "weights"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Global variable for the upscaler
global_upscaler = None

# VRAM optimization constants
TILE_SIZE = 100  # Adjust based on available VRAM
BATCH_SIZE = 4  # Reduce if running out of memory


def initialize_upscaler():
    global global_upscaler

    try:
        if global_upscaler is not None:
            return global_upscaler

        # Set up device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.empty_cache()  # Clear CUDA cache

            # Limit VRAM usage
            torch.cuda.set_per_process_memory_fraction(
                0.7
            )  # Use only 70% of available VRAM
        else:
            device = torch.device("cpu")

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        model_path = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")

        if not os.path.exists(model_path):
            model_path = load_file_from_url(model_url, model_dir=WEIGHTS_DIR)

        global_upscaler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=TILE_SIZE,  # Use tiling to process large images
            tile_pad=10,
            pre_pad=0,
            half=True if device.type == "cuda" else False,  # Use FP16 on GPU
        )

        return global_upscaler
    except Exception as e:
        print(f"Error initializing upscaler: {str(e)}")
        raise


def get_unique_filename(directory, base_filename):
    name, ext = os.path.splitext(base_filename)
    counter = 1
    new_filename = base_filename

    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1

    return new_filename


def save_input_image(image_data):
    if image_data is None:
        return None

    try:
        if isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        elif isinstance(image_data, dict) and "image" in image_data:
            image = Image.fromarray(image_data["image"])
        else:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"input_{timestamp}.png"
        unique_filename = get_unique_filename(INPUT_DIR, base_filename)

        input_path = os.path.join(INPUT_DIR, unique_filename)
        image.save(input_path, format="PNG")

        return input_path
    except Exception as e:
        print(f"Error saving input image: {str(e)}")
        return None


def get_output_filename(input_filename, scale_factor):
    name, ext = os.path.splitext(os.path.basename(input_filename))
    return f"{name}_x{scale_factor}.png"


@torch.inference_mode()  # More memory efficient than torch.no_grad()
def upscale_image(input_image, scale_factor):
    try:
        input_path = save_input_image(input_image)
        if input_path is None:
            return None, None, "Failed to save input image"

        upscaler = initialize_upscaler()
        if upscaler is None:
            return None, None, "Failed to initialize upscaler"

        # Load and preprocess image
        input_img = Image.open(input_path)

        # Resize very large images to reduce memory usage
        max_size = 2000  # Maximum dimension
        if input_img.width > max_size or input_img.height > max_size:
            input_img.thumbnail((max_size, max_size), Image.LANCZOS)

        input_array = np.array(input_img)

        if len(input_array.shape) == 2:
            input_array = np.stack([input_array] * 3, axis=-1)
        elif input_array.shape[2] == 4:
            input_array = input_array[:, :, :3]

        # Process image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache before processing

        output, _ = upscaler.enhance(input_array, outscale=scale_factor)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache after processing

        # Save output image
        output_filename = get_output_filename(input_path, scale_factor)
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        output_image = Image.fromarray(output)
        output_image.save(output_path, format="PNG")

        return input_path, output_path, None
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache in case of error
        print(f"Error in upscale_image: {str(e)}")
        return None, None, f"An error occurred: {str(e)}"


def open_folder_in_explorer(folder_path):
    try:
        if os.name == "nt":  # Windows
            os.startfile(os.path.abspath(folder_path))
        elif os.name == "posix":  # macOS or Linux
            subprocess.run(["open" if os.name == "darwin" else "xdg-open", folder_path])
        return f"Opened {folder_path} in file explorer"
    except Exception as e:
        return f"Failed to open folder: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Image Upscaler using ESRGAN")
    parser.add_argument("--share", action="store_true", help="Create a public URL")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    args = parser.parse_args()

    try:
        initialize_upscaler()
    except Exception as e:
        print(f"Failed to initialize upscaler at startup: {str(e)}")

    with gr.Blocks() as iface:
        gr.Markdown("# Image Upscaler using ESRGAN")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Upload Image")
                scale_slider = gr.Slider(
                    minimum=1, maximum=4, step=1, value=2, label="Upscale Factor"
                )
                upscale_button = gr.Button("🔄 Upscale")

            with gr.Column():
                output_image = gr.Image(type="filepath", label="Upscaled Image")
                error_output = gr.Textbox(label="Status/Error", visible=False)
                input_file_path = gr.Textbox(label="Input File Path", visible=False)
                output_file_path = gr.Textbox(label="Output File Path", visible=False)

        with gr.Row():
            output_folder_button = gr.Button("📂 Open Output Folder")

        folder_status = gr.Textbox(label="Folder Open Status", visible=False)

        def process_image(image, scale):
            if image is None:
                return [None, "No image uploaded", None, None, False]
            input_path, output_path, error = upscale_image(image, scale)
            if error:
                return [None, error, None, None]
            return [output_path, "Success!", input_path, output_path]

        # Event handlers
        upscale_button.click(
            fn=process_image,
            inputs=[input_image, scale_slider],
            outputs=[
                output_image,
                error_output,
                input_file_path,
                output_file_path,
            ],
        )

        output_folder_button.click(
            fn=lambda: open_folder_in_explorer(OUTPUT_DIR), outputs=folder_status
        )

    # Launch the interface
    webbrowser.open_new(f"http://localhost:{args.port}")
    iface.launch(share=args.share, server_name=args.server_name, server_port=args.port)


if __name__ == "__main__":
    main()
