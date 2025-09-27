#!/bin/bash

echo "🚀 Setting up Whisper Persian training environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
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
echo "   2. Get your token from: https://huggingface.co/settings/tokens"
echo "   3. Run: python whisper_trainer.py"
echo ""
echo "🎯 You're ready to train your Whisper model!"