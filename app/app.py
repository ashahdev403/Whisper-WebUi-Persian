from transformers import pipeline
import gradio as gr

# --- 1. Define Model Choices ---
# A dictionary to map user-friendly names to the actual model IDs
MODELS = {
    "Whisper Small": "AmirMohseni/whisper-small-persian-bf16",
    "Whisper Large v3": "AmirMohseni/whisper-large-v3-persian-bf16"
}

# A dictionary to cache loaded models so we don't reload them on every run
model_pipelines = {}

# --- 2. Modify the function to work with the new model choices ---
def transcribe(audio, model_choice):
    """
    Transcribes the given audio using the selected Whisper model.
    """
    if audio is None:
        return "No audio provided."

    # Handle case where model_choice is None or not in MODELS
    if model_choice is None or model_choice not in MODELS:
        model_choice = "Whisper Small"  # Default to Whisper Small if None or invalid choice

    # Get the model ID from our dictionary based on the user's choice
    model_id = MODELS[model_choice]

    # Caching: Load the model if it's not already in our cache
    if model_id not in model_pipelines:
        print(f"Loading model: {model_id}...")
        # Create the ASR pipeline with explicit configuration to fix warnings
        pipe = pipeline(
            "automatic-speech-recognition", 
            model=model_id,
            generate_kwargs={"language": "persian", "task": "transcribe"}
        )
        model_pipelines[model_id] = pipe
        print("Model loaded.")
    
    # Retrieve the selected pipeline from the cache
    selected_pipe = model_pipelines[model_id]
    
    # The pipeline function takes the file path of the audio and transcribes it
    result = selected_pipe(audio)
    return result["text"]

# --- 3. Update the Interface with the new model selection input ---
iface = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Record Audio 🎤"),
        gr.Radio(
            choices=list(MODELS.keys()),  # Get choices directly from the dictionary keys
            value="Whisper Small",        # Default to the faster, smaller model
            label="Choose Model",
            info="Select the model to use for transcription"
        )
    ],
    outputs="text",
    title="Whisper Farsi 🎙️",
    description="Realtime demo for Persian speech recognition. Choose a model, press the record button, and speak.",
)

# Launch the Gradio app
iface.launch()