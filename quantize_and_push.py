import torch
import argparse
import os
from huggingface_hub import login, create_repo, upload_folder
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig

def quantize_model(model_id: str, precision: str, output_dir: str):
    """
    Loads a model, applies quantization, and saves it to a local directory.
    """
    print(f"--- Starting local quantization for '{model_id}' to '{precision}' ---")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_id)
    model = None

    # Apply the specified quantization
    if precision == 'bf16':
        print("Loading fp32 model for bf16 conversion...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        # Save with bf16 dtype
        model.save_pretrained(output_dir, torch_dtype=torch.bfloat16)
        
    elif precision == 'int8':
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["proj_out"])
        print("Loading and quantizing model to int8...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        # Save the 8-bit capable model
        model.save_pretrained(output_dir)
        
    else:
        raise ValueError("Invalid precision. Choose 'bf16' or 'int8'.")

    # Save the processor to the same directory
    processor.save_pretrained(output_dir)
    print(f"✅ Model and processor saved locally to '{output_dir}'.")


def push_to_hub(local_dir: str, hub_model_id: str, precision: str):
    """
    Pushes the contents of a local directory to the Hugging Face Hub.
    """
    # Get HF token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token or not hub_model_id:
        raise ValueError("Hugging Face token (HF_TOKEN env var) and Hub model ID are required to push.")

    print(f"--- Starting push to Hub for repository '{hub_model_id}' ---")
    
    # Login to the Hub
    login(token=hf_token)
    
    # Create the repository (if it doesn't exist)
    create_repo(repo_id=hub_model_id, exist_ok=True)
    
    # Upload the folder contents
    print(f"Uploading files from '{local_dir}'...")
    upload_folder(
        folder_path=local_dir,
        repo_id=hub_model_id,
        commit_message=f"Upload {precision} quantized model"
    )
    print(f"✅ Successfully pushed to https://huggingface.co/{hub_model_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize and optionally push a Whisper model.")
    
    # --- Arguments ---
    parser.add_argument("--model_id", type=str, default="AmirMohseni/whisper-small-persian", help="HF model ID.")
    parser.add_argument("--precision", type=str, choices=['bf16', 'int8'], required=True, help="Target precision.")
    parser.add_argument("--output_dir", type=str, required=True, help="Local directory to save the model.")
    parser.add_argument("--push_to_hub", action='store_true', help="Flag to push the model to the Hub after quantization.")
    parser.add_argument("--hub_model_id", type=str, help="Repo ID for the Hub (e.g., YourUsername/MyModel).")

    args = parser.parse_args()

    # --- Step 1: Quantize and save the model locally ---
    quantize_model(
        model_id=args.model_id,
        precision=args.precision,
        output_dir=args.output_dir
    )

    # --- Step 2: Push to Hub if requested ---
    if args.push_to_hub:
        push_to_hub(
            local_dir=args.output_dir,
            hub_model_id=args.hub_model_id,
            precision=args.precision
        )