import gradio as gr
from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import Task, OCRRequest, TranscriptionRequest, TranslationRequest

# Constants
TEST_IMAGE_URL = "https://jeroen.github.io/images/testocr.png"
TEST_AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/master/mono_44100/127389__acclivity__thetimehascome.wav"
TEST_TEXT = "Hello, world! This is a test translation."

def process_ocr(image, providers):
    results = {}
    for provider in providers:
        client = supercontrast_client(task=Task.OCR, providers=[Provider[provider]])
        response = client.request(OCRRequest(image=image))
        results[provider] = response.all_text
    return [results.get(provider, "") for provider in ["API4AI", "AWS", "AZURE", "CLARIFAI", "GCP", "SENTISIGHT"]]

def process_transcription(audio, providers):
    results = {}
    for provider in providers:
        client = supercontrast_client(task=Task.TRANSCRIPTION, providers=[Provider[provider]])
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
    return [results.get(provider, "") for provider in ["ANTHROPIC", "AWS", "AZURE", "GCP", "MODERNMT", "OPENAI"]]

with gr.Blocks() as demo:
    gr.Markdown("# SuperContrast Demo")
    
    with gr.Tab("OCR"):
        ocr_input = gr.Image(type="filepath", label="Input Image")
        ocr_providers = gr.CheckboxGroup(choices=["API4AI", "AWS", "AZURE", "CLARIFAI", "GCP", "SENTISIGHT"], label="Providers")
        ocr_outputs = {provider: gr.Textbox(label=f"{provider} OCR Result") for provider in ["API4AI", "AWS", "AZURE", "CLARIFAI", "GCP", "SENTISIGHT"]}
        ocr_button = gr.Button("Process OCR")
        ocr_button.click(process_ocr, inputs=[ocr_input, ocr_providers], outputs=list(ocr_outputs.values()))
    
    with gr.Tab("Transcription"):
        transcription_input = gr.Audio(type="filepath", label="Input Audio")
        transcription_providers = gr.CheckboxGroup(choices=["AZURE", "OPENAI"], label="Providers")
        transcription_outputs = {provider: gr.Textbox(label=f"{provider} Transcription Result") for provider in ["AZURE", "OPENAI"]}
        transcription_button = gr.Button("Process Transcription")
        transcription_button.click(process_transcription, inputs=[transcription_input, transcription_providers], outputs=list(transcription_outputs.values()))
    
    with gr.Tab("Translation"):
        translation_input = gr.Textbox(label="Input Text")
        translation_providers = gr.CheckboxGroup(choices=["ANTHROPIC", "AWS", "AZURE", "GCP", "MODERNMT", "OPENAI"], label="Providers")
        source_lang = gr.Textbox(label="Source Language", value="en")
        target_lang = gr.Textbox(label="Target Language", value="fr")
        translation_outputs = {provider: gr.Textbox(label=f"{provider} Translation Result") for provider in ["ANTHROPIC", "AWS", "AZURE", "GCP", "MODERNMT", "OPENAI"]}
        translation_button = gr.Button("Process Translation")
        translation_button.click(process_translation, inputs=[translation_input, translation_providers, source_lang, target_lang], outputs=list(translation_outputs.values()))

demo.launch()
