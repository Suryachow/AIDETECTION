<div align="center">
  
# 🤖➡️👤 **AI-Text-Humanizer**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000.svg)](https://vercel.com/)

**Transform AI-generated content into natural, human-like text that bypasses detection**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Advanced Settings](#advanced-settings) • [Tips](#%EF%B8%8F-tips-for-better-bypassing) • [Contributing](#contributing)

</div>

---

## 📋 Overview

The **AI-Text-Humanizer**, developed by **Muhammad Khizer Zakir**, is a sophisticated FastAPI application designed to refine AI-generated text into authentic, engaging, and human-like content. By leveraging lightweight NLP techniques rather than heavy LLMs or trained models, it effectively bypasses AI content detectors while preserving the original message.

Choose from three distinct writing styles:
- 💬 **Casual** - Conversational and relaxed tone
- 👔 **Professional** - Formal and business-appropriate language
- 🎨 **Creative** - Expressive and imaginative phrasing

This open-source project welcomes contributions from the community!

## ✨ Features

- **🔄 Natural Text Transformation**: Converts AI-generated text into authentic human-like prose
- **🎭 Multiple Writing Styles**: Supports casual, professional, and creative tones
- **🔌 REST API**: Seamlessly integrates with other applications and services
- **🖥️ Interactive Web Interface**: User-friendly design for real-time text humanization
- **📦 Batch Processing**: Efficiently processes multiple texts simultaneously
- **📝 Feedback Mechanism**: Continuously improves based on user input

## 🔧 Technology Stack

This project uses lightweight NLP techniques instead of resource-intensive LLMs to make the application more accessible:

```
fastapi==0.104.1
uvicorn==0.23.2
pydantic==2.4.2
python-dotenv==1.0.0
numpy==1.26.1
scikit-learn==1.3.2
nltk==3.8.1
gensim==4.3.3
pyspellchecker==0.7.0
language-tool-python==2.9.3
textblob==0.17.1
```

- **Backend**: FastAPI with Python NLP libraries
- **Frontend**: Created with Vercel v0
- **Deployment**: Docker container support

## 🗂️ Directory Structure

```
AI-Text-Humanizer/
├── app/
│   ├── __init__.py
│   ├── utils.py              # Utility functions
│   ├── main.py               # FastAPI application entry point
│   ├── api/                  # API route handlers
│   │   └── routes.py
│   ├── core/                 # Core logic for text processing
│   │   ├── agent.py
│   │   ├── rag.py
│   │   └── reward.py
│   ├── models/               # Machine learning model loading
│   └── utils/                # Pre/post-processing helpers
├── data/                     # Corpus for RAG and vector database
├── .gitignore
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── run.sh                    # Startup script
```

## 🚀 Installation

Follow these steps to set up the **AI-Text-Humanizer** on your local machine.

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized setup)
- Git

### Steps

1. **Clone the repository:**
   ```Powershell
   git clone https://github.com/Khizer-Data/AI-Text-Humanizer.git
   cd AI-Text-Humanizer
   ```

2. **Install Node Modules**
   ```Powershell
   npm install --legacy-peer-deps
   ```
   
3. **First-time setup with Docker:**
   ```Powershell
   docker build -t ai-text-humanizer .
   docker run ai-text-humanizer
   ```

4. **For subsequent runs (when Docker container is running):**
   
   Just run exisitng Container in Docker desktop
   
6. **Start Frontend Development Server (Next.js)**
   ```Powershell
   npm run dev
   ```

## 💻 Usage

1. Access the interactive web interface at `http://localhost:3000`
2. Paste your AI-generated text into the input field
3. Select your preferred writing style: Casual, Professional, or Creative
4. Adjust advanced settings if needed
5. Click "Humanize" to transform your text
6. Copy the humanized text or use the "Re-Humanize" option for multiple passes

## ⚙️ Advanced Settings

Fine-tune the humanization process with these powerful options:

| Setting | Description |
|---------|-------------|
| **Transformation Strength (0.9)** | Higher values produce more creative and varied transformations that better evade detection |
| **Preserve Original Meaning** | Toggle off to allow more aggressive transformations that may slightly alter meaning but are more effective at bypassing detection |
| **Multi-Pass Processing** | Applies multiple transformation passes for better results. Recommended for bypassing sophisticated detection systems like undetectable.ai |

## 🛠️ Tips for Better Bypassing

- Process smaller text chunks (1–3 paragraphs) for **best results**
- Use the **"Re-Humanize"** button to apply **multiple passes**
- Try different writing styles (Casual, Professional, Creative) for variations
- After processing, manually tweak 1–2 words for maximum bypass
- Combine this tool with other minor edits for **perfect undetectability**

## 💡 How It Works

Unlike many text humanizers that rely on expensive LLMs, this tool uses a combination of lightweight NLP techniques to:

- Analyze text patterns typical of AI-generated content
- Restructure sentences while maintaining core meaning
- Apply stylistic transformations based on selected writing style
- Use word replacements and syntax variations to bypass detection
- Implement multiple transformation passes for better results

The application may not be perfect but provides an efficient and accessible solution for bypassing AI content detection.

## 🤝 Contributing

**AI-Text-Humanizer** is an open-source project, and we welcome contributions from developers, researchers, and enthusiasts!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
  
**Developed with ❤️ by Muhammad Khizer Zakir**

[GitHub](https://github.com/Khizer-Data) • [Project Repository](https://github.com/Khizer-Data/AI-Text-Humanizer)

*Open-source and community-driven. Join us in making AI text more human!*
</div>
