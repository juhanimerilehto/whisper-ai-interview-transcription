# Interview Transcriber

**Version 1.0**
### Creator: Juhani Merilehto - @juhanimerilehto - Jyväskylä University of Applied Sciences (JAMK), Likes institute

![JAMK Likes Logo](./assets/jamklikes.png)

## Overview

Interview Transcriber is a Python-based tool that processes audio interviews and generates transcriptions with speaker diarization using AI. It was developed at JAMK University of Applied Sciences in collaboration with Likes institute. The tool utilizes Whisper for speech recognition and pyannote.audio for speaker diarization, producing transcriptions in multiple formats with precise speaker identification and timestamps. The solution is fully local, which makes it ideal from data privacy perspective.

 - **Transcription Model:** https://huggingface.co/openai/whisper-large-v3
 - **Diarization model:** https://hf.co/pyannote/speaker-diarization

## Features

- **Automated Transcription**: Uses OpenAI's Whisper for accurate speech recognition
- **Speaker Diarization**: Identifies and separates different speakers using pyannote.audio
- **Multi-format Output**: Generates transcriptions in JSON, Markdown, and plain text
- **GPU Acceleration**: Automatic CUDA support for faster processing
- **Timestamp Integration**: Includes precise timing for each speech segment
- **Finnish Language Support**: Optimized for Finnish language transcription
- **Configurable Processing**: Easy modification of transcription parameters

## Hardware Requirements
The tool automatically uses GPU acceleration when available, which significantly improves processing speed:

- **CPU:** Works on any modern CPU, but processing will be slower
- **GPU:** NVIDIA GPU with CUDA support recommended for faster processing

  - Minimum 4GB VRAM for optimal performance
  - CUDA toolkit will be automatically installed with PyTorch

Note: The tool will automatically detect if CUDA is available and switch to CPU if it's not, so no additional configuration is needed.

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/juhanimerilehto/jamk-interview-transcriber.git
cd jamk-interview-transcriber
```
### 2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```
### 3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
### 4. Get your HuggingFace & pyannote token:

- Go to https://huggingface.co/
- Create an account if you don't have one
- Go to Settings -> Access Tokens
- Create a new token and copy it
- Pyannote token can be retrieved here: https://hf.co/pyannote/speaker-diarization


## Usage

### 1. Place your audio file in the project directory.

### 2. Open _whisper-transcribe.py_ and modify the following parameters at the bottom of the file:
```python
# Replace with your HuggingFace token
HF_TOKEN = "your_token_here"

# Modify these according to your needs
interview_path = "your_audio_file.mp3"
output_base_path = "desired_output_name"
```
### 3. Run the transcription:
```bash
python whisper-transcribe.py
```

## Configuration
The transcriber is configured with the following default settings which can be modified directly in the code:

- **Whisper Model:** Using "large-v3" for optimal transcription quality
- **Language:** Set to Finnish ("fi")
- **Audio Processing:** 16kHz sample rate, mono channel
- **Output Formats:** Generates three files:

  - JSON (detailed data with timestamps)
  - Markdown (formatted readable version)
  - Text (simple readable version)



To modify these settings, adjust the relevant parameters in the _InterviewTranscriber_ class:
```python
# In transcribe_interview method:
segment_result = self.whisper_model.transcribe(segment_audio,
                                           language="fi",  # Change language here
                                           task="transcribe",
                                           fp16=False)

# In __init__ method:
self.whisper_model = whisper.load_model("large-v3")  # Change model size here
```
Note: This is a fully local solution. After the initial model downloads, no data is sent to external servers, making it suitable for processing sensitive interview data.


## Output

The tool generates three types of output files:
1. JSON format (`output.json`) - Machine-readable format with detailed metadata
2. Markdown format (`output.md`) - Human-readable format with nice formatting
3. Plain text format (`output.txt`) - Simple text format for easy reading


## File Structure

```plaintext
jamk-interview-transcriber/
├── assets/
│   └── jamklikes.png
├── whisper-transcribe.py
├── requirements.txt
└── .gitignore
```

## Credits

- **Juhani Merilehto (@juhanimerilehto)** – Specialist, Data and Statistics
- **JAMK Likes** – Organization sponsor, providing use case and requirements for interview analysis

## License

This project is licensed for free use under the condition that proper credit is given to Juhani Merilehto (@juhanimerilehto) and JAMK Likes institute. You are free to use, modify, and distribute this project, provided that you mention the original author and institution and do not hold them liable for any consequences arising from the use of the software.
