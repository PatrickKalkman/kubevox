# KubeVox: Your Kubernetes Voice Assistant (Now Running on Local LLMs!)

[![KubeVox in Action](https://img.youtube.com/vi/N2WdsJc92UU/maxresdefault.jpg)](https://www.youtube.com/watch?v=N2WdsJc92UU)

[![GitHub stars](https://img.shields.io/github/stars/PatrickKalkman/kubevox)](https://github.com/PatrickKalkman/kubevox/stargazers)
[![GitHub contributors](https://img.shields.io/github/contributors/PatrickKalkman/kubevox)](https://github.com/PatrickKalkman/kubevox/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/PatrickKalkman/kubevox)](https://github.com/PatrickKalkman/kubevox)
[![open issues](https://img.shields.io/github/issues/PatrickKalkman/kubevox)](https://github.com/PatrickKalkman/kubevox/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)

Want to manage your Kubernetes cluster with just your voice? KubeVox turns "Hey, show me all pods in production" into reality. No more typing long kubectl commands – just talk naturally to your cluster.

Read the full story behind KubeVox in my [Medium article](https://medium.com/p/d9baed31d62b).

## ✨ Key Features

- **100% Local Processing**: Your voice and commands stay on your machine with mlx-whisper and Llama 3.2
- **Lightning Fast**: Runs 10x faster than cloud-based alternatives
- **Privacy First**: No data leaves your machine (except for optional ElevenLabs TTS)
- **Natural Voice Control**: Talk to your cluster like you're chatting with a colleague
- **Smart Command Translation**: Automatically converts speech to the right kubectl commands
- **Crystal Clear Responses**: Optional high-quality voice responses using ElevenLabs
- **Secure by Design**: Uses your existing kubectl credentials and permissions
- **Fully Documented API**: Easy to extend with your own custom commands

## 🏗️ Architecture

KubeVox uses a three-part architecture optimized for speed and privacy:

1. **Local Speech-to-Text**: Uses mlx-whisper for fast, private voice transcription
2. **Local LLM**: Leverages Llama 3.2 through llama.cpp for command understanding
3. **Voice Synthesis**: Optional ElevenLabs integration for natural-sounding responses

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- A working Kubernetes cluster
- An ElevenLabs API key (optional, but recommended for voice output)
- A decent microphone
- llama.cpp installed and configured (see below)

### Setting Up llama.cpp

1. First, clone and build llama.cpp:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

2. Download the Llama 3.2 model:
```bash
# Download from Hugging Face (model URL will be provided)
wget https://huggingface.co/path/to/Llama-3.2-3B-Instruct-Q4_K_M
```

3. Start the llama.cpp server:
```bash
./llama-server -m ./models/Llama-3.2-3B-Instruct-Q4_K_M --port 8080
```

### Installing KubeVox

1. **Install UV** (if you haven't already):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone & Navigate**:
```bash
git clone https://github.com/PatrickKalkman/kubevox
cd kubevox
```

3. **Set API Keys** (if using ElevenLabs):
```bash
export ELEVENLABS_API_KEY='your-elevenlabs-api-key-here'
```

### Usage Examples

```bash
# Text mode (ask a question directly)
uv run kubevox text "show me all pods in the default namespace"

# Voice mode (start listening)
uv run kubevox --voice

# Voice mode with specific input device
uv run kubevox --voice --device <device_index>

# Voice mode with voice output
uv run kubevox --voice --output voice

# Verbose output
uv run kubevox -v --text "show all my services"
```

## 🔒 Security

KubeVox takes security seriously:

- Uses your existing Kubernetes RBAC permissions
- Function-based access control for commands
- Local processing means your data stays private
- No permanent storage of voice data

## 🛠️ Extending KubeVox

Adding new commands is straightforward with our decorator system:

```python
@FunctionRegistry.register(
    description="Get the logs from a specified pod",
    response_template="The logs from pod {pod} in namespace {namespace} are: {logs}"
)
async def get_pod_logs(pod_name: str, namespace: str = "default"):
    # Your implementation here
    pass
```

## 🤝 Contributing

Love KubeVox? Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🔮 Future Plans

- Fine-tuned smaller models for faster response times
- Expanded command library
- Multi-language support
- Local text-to-speech options

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- llama.cpp team for the amazing inference engine
- MLX team for the optimized Whisper implementation
- ElevenLabs for high-quality voice synthesis
- The Kubernetes Python client team
- All our contributors and users!

---

Built with ❤️ by the community. Star us on GitHub if you find this useful!
