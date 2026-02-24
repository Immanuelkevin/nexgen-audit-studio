# NexGen | AI Contact Center Audit Studio 📡

NexGen is a high-performance, AI-driven auditing platform designed for modern contact centers. It automates the quality assurance process by transcribing agent-side audio, performing multi-language compliance audits, and generating professional PDF insights in seconds.

## 🚀 Features

*   **⚡ Ultra-Fast Transcription**: Powered by Groq's Whisper-Large-V3 (Neural decoding at peak speeds).
*   **🛡️ Automated PII Redaction**: Intelligent masking of sensitive data (Phone numbers, IDs, Addresses) for privacy compliance.
*   **🚩 Agent Risk Detection**: Real-time identification of unprofessional language, competitor mentions, or lack of brand compliance.
*   **📈 Sentiment Journey Arc**: Visualize the emotional flow of a call through a temporal line graph.
*   **🎙️ Live Voice Recording**: Direct browser-based recording for instant analysis.
*   **📋 Professional PDF Hub**: One-click generation of branded audit reports for management review.
*   **🌍 Multi-Language Support**: English, Tamil, Telugu, Hindi, Malayalam, and Kannada.

## 🛠️ Tech Stack

*   **Backend**: Python, FastAPI
*   **AI Engines**: Groq Cloud API (Whisper L3, Llama 3.3 70B)
*   **Frontend**: Tailwind CSS, Chart.js, WaveSurfer.js
*   **Reporting**: FPDF2

## 📦 Local Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd nexgen-audit-studio
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file or set the environment variable:
   ```bash
   export GROQ_API_KEY=your_api_key_here
   ```

5. **Run the App**:
   ```bash
   python app_v3_main.py
   ```

## 🌐 Deployment

This app is ready for deployment on **Render**, **Railway**, or **AWS**.
- **Port**: 8080
- **Entry Point**: `uvicorn app_v3_main.py:app --host 0.0.0.0 --port 8080`

---
*Created by [Your Name] - Focused on bridging AI and Operational Excellence.*
