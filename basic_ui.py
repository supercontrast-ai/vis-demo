import gradio as gr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image
from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import (
    OCRRequest,
    OCRResponse,
    Task,
    TranscriptionRequest,
    TranslationRequest,
)

# Constants
TEST_IMAGE_URL = "https://jeroen.github.io/images/testocr.png"
TEST_AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/master/mono_44100/127389__acclivity__thetimehascome.wav"
TEST_TEXT = "Hello, world! This is a test translation."

# Define providers
PROVIDERS = [
    Provider.AWS,
    Provider.GCP,
    Provider.AZURE,
    Provider.SENTISIGHT,
    Provider.CLARIFAI,
    Provider.API4AI,
]

OCR_PROVIDERS = ["API4AI", "AWS", "AZURE", "CLARIFAI", "GCP", "SENTISIGHT"]

# Define output directory for saving plots
OUTPUT_DIR = "test_data/ocr"


def get_image_path(image_input):
    if isinstance(image_input, str):
        if os.path.isdir(image_input):
            # If it's a directory, find the first image file
            for file in os.listdir(image_input):
                if file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
                ):
                    return os.path.join(image_input, file)
            raise ValueError(f"No image file found in directory: {image_input}")
        elif os.path.isfile(image_input):
            return image_input
        else:
            raise ValueError(f"Invalid image path: {image_input}")
    elif isinstance(image_input, Image.Image):
        # If it's already a PIL Image, save it temporarily and return the path
        temp_path = os.path.join(OUTPUT_DIR, "temp_input_image.png")
        image_input.save(temp_path)
        return temp_path
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def plot_bounding_boxes(
    image_path: str, responses: dict[Provider, OCRResponse], output_dir: str
):
    # Ensure we have a valid file path
    image_path = get_image_path(image_path)

    img = Image.open(image_path)

    results = {}
    for provider, ocr_response in responses.items():
        # Create a new figure for each provider
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        for box in ocr_response.bounding_boxes:
            # Create a Rectangle patch
            rect = patches.Rectangle(
                (box.coordinates[0][0], box.coordinates[0][1]),
                box.coordinates[2][0] - box.coordinates[0][0],
                box.coordinates[2][1] - box.coordinates[0][1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)

            # Add text annotation
            ax.text(
                box.coordinates[0][0],
                box.coordinates[0][1],
                box.text,
                color="blue",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7),
            )

        ax.axis("off")

        # Remove any extra white space around the image
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Convert plot to PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_result = Image.fromarray(img_array)

        # Save the plot as an image file
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"ocr_{provider.value}_{image_name}")
        img_result.save(output_path)
        print(f"Saved {provider.value} plot to: {output_path}")

        # Store the image and text result
        results[provider] = {"image": img_result, "text": ocr_response.all_text}

        plt.close(fig)

    return results


def process_ocr(image, providers):
    image_path = get_image_path(image)
    results = {}
    for provider in providers:
        client = supercontrast_client(task=Task.OCR, providers=[Provider[provider]])
        response = client.request(OCRRequest(image=image_path))
        results[Provider[provider]] = response

    plot_results = plot_bounding_boxes(image_path, results, OUTPUT_DIR)

    response = []
    for provider in OCR_PROVIDERS:
        if Provider[provider] in plot_results:
            response.extend(
                [
                    plot_results[Provider[provider]]["image"],
                    plot_results[Provider[provider]]["text"],
                ]
            )
        else:
            response.extend([Image.open("dummy.png"), "foo bar"])

    return response


def process_transcription(audio, providers):
    results = {}
    for provider in providers:
        client = supercontrast_client(
            task=Task.TRANSCRIPTION, providers=[Provider[provider]]
        )
        response = client.request(TranscriptionRequest(audio_file=audio))
        results[provider] = response.text
    return [results.get(provider, "") for provider in ["AZURE", "OPENAI"]]


def process_translation(text, providers, source_lang, target_lang):
    results = {}
    for provider in providers:
        client = supercontrast_client(
            task=Task.TRANSLATION,
            providers=[Provider[provider]],
            source_language=source_lang,
            target_language=target_lang,
        )
        response = client.request(TranslationRequest(text=text))
        results[provider] = response.text
    return [
        results.get(provider, "")
        for provider in ["ANTHROPIC", "AWS", "AZURE", "GCP", "MODERNMT", "OPENAI"]
    ]


with gr.Blocks() as demo:
    gr.Markdown("# SuperContrast Demo")

    with gr.Tab("OCR"):
        ocr_input = gr.Image(type="filepath", label="Input Image")
        selected_providers = gr.CheckboxGroup(choices=OCR_PROVIDERS, label="Providers")
        ocr_button = gr.Button("Process OCR")

        # Dynamic output creation
        ocr_outputs = []
        for provider in OCR_PROVIDERS:
            with gr.Row():
                with gr.Column(visible=False) as provider_column:
                    ocr_outputs.append(gr.Image(label=f"{provider} OCR Result"))
                    ocr_outputs.append(gr.Textbox(label=f"{provider} OCR Text"))

                selected_providers.change(
                    lambda p, prov=provider: gr.update(visible=prov in p),
                    inputs=[selected_providers],
                    outputs=[provider_column],
                )

        ocr_button.click(
            process_ocr, inputs=[ocr_input, selected_providers], outputs=ocr_outputs
        )

    with gr.Tab("Transcription"):
        transcription_input = gr.Audio(type="filepath", label="Input Audio")
        transcription_providers = gr.CheckboxGroup(
            choices=["AZURE", "OPENAI"], label="Providers"
        )
        transcription_button = gr.Button("Process Transcription")
        transcription_outputs = {
            provider: gr.Textbox(label=f"{provider} Transcription Result")
            for provider in ["AZURE", "OPENAI"]
        }
        transcription_button.click(
            process_transcription,
            inputs=[transcription_input, transcription_providers],
            outputs=list(transcription_outputs.values()),
        )

    with gr.Tab("Translation"):
        translation_input = gr.Textbox(label="Input Text")
        translation_providers = gr.CheckboxGroup(
            choices=["ANTHROPIC", "AWS", "AZURE", "GCP", "MODERNMT", "OPENAI"],
            label="Providers",
        )
        source_lang = gr.Textbox(label="Source Language", value="en")
        target_lang = gr.Textbox(label="Target Language", value="fr")
        translation_button = gr.Button("Process Translation")
        translation_outputs = {
            provider: gr.Textbox(label=f"{provider} Translation Result")
            for provider in ["ANTHROPIC", "AWS", "AZURE", "GCP", "MODERNMT", "OPENAI"]
        }
        translation_button.click(
            process_translation,
            inputs=[translation_input, translation_providers, source_lang, target_lang],
            outputs=list(translation_outputs.values()),
        )

demo.launch()
