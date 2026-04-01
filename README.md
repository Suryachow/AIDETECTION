# JustDone AI Migration Bundle

This bundle contains the core logic for **AI Detection** and **Humanization**.

## Contents:
- `backend/services/ai_service.py`: Main service for detection and humanization.
- `backend/agent/humanizer_agent.py`: Semantic rewriting logic.
- `backend/core/config.py`: Configuration settings.

## Instructions:
1. Copy the `backend/` folder contents into your new project's backend.
2. **Crucial:** Copy the `ScribePro/JD/JD/backend/khizer_humanizer` folder from the original project into your new `backend/` directory. This is the Node.js service required for high-quality humanization.
3. Install dependencies:
   ```bash
   pip install fastapi uvicorn transformers torch nltk spacy groq sentence-transformers pydantic-settings
   ```
4. Download the spacy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
5. Ensure you have your `GROQ_API_KEY` set in your `.env` file.

## AI Detection:
The detection engine will automatically download the RoBERTa model on its first run (approx 500MB).

## Humanization:
The system will attempt to call the Node.js service on port 3001. If it's not running, it will fall back to the Python `humanizer_agent`.
