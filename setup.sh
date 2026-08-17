#!/bin/bash

echo "🚀 Setting up the Whisper Persian environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ffmpeg is required to decode audio and video, and to download from URLs
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  ffmpeg was not found on your PATH. The WebUI and CLI need it to read media files."
    echo "   Ubuntu/Debian: sudo apt install ffmpeg"
    echo "   macOS:         brew install ffmpeg"
    echo "   Windows:       winget install Gyan.FFmpeg"
fi

echo "📦 Creating virtual environment..."
uv venv

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "📚 Installing dependencies..."
uv pip install -r requirements.txt

echo "🔐 Setting up environment file..."
if [ ! -f .env.example ]; then
    echo "⚠️  .env.example not found, creating basic .env file..."
    echo "HF_TOKEN=your_token_here" > .env
else
    cp .env.example .env
    echo "✅ Copied .env.example to .env"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🔑 Next steps:"
echo "   1. Edit .env file and replace 'your_token_here' with your Hugging Face token"
echo "      (only needed to publish models - transcription works without it)"
echo "   2. Get your token from: https://huggingface.co/settings/tokens"
echo ""
echo "🎧 To transcribe:      python app.py"
echo "🖥️  To batch transcribe: python cli.py audio.mp3 --model 'Persian Small'"
echo "🎯 To train:           python training/whisper_trainer.py"
echo ""
