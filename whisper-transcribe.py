import torch
import whisper
from pyannote.audio import Pipeline
import datetime
import json
import numpy as np
from pathlib import Path
import librosa
import warnings
warnings.filterwarnings("ignore")

class InterviewTranscriber:
    # Modify the __init__ method to use torch.device
    def __init__(self, hf_token):
        """
        Initialize the transcriber with HuggingFace token for pyannote.audio
        Args:
            hf_token (str): HuggingFace access token
        """
        # Convert string to torch.device object
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
        # Initialize Whisper model (i.e., large-v3, see https://huggingface.co/openai/whisper-large-v3)
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("large-v3", device=self.device)
    
        # Initialize Speaker Diarization
        print("Loading diarization pipeline...")
        self.diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=hf_token
        ).to(self.device)

    def transcribe_interview(self, audio_path, output_base_path):
        """
        Transcribe interview with speaker diarization and save in multiple formats
        Args:
            audio_path (str): Path to input audio file
            output_base_path (str): Base path for output files (without extension)
        """
        print(f"Processing audio file: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Get speaker diarization
        diarization_result = self.diarization({"waveform": torch.from_numpy(audio).unsqueeze(0), 
                                             "sample_rate": sr})
        
        # Process segments with speakers
        segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end
            
            # Extract audio segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Transcribe segment
            segment_result = self.whisper_model.transcribe(segment_audio,
                                                         language="fi", #set language here (now, Finnish)
                                                         task="transcribe",
                                                         fp16=False)
            
            # Format timestamp
            start_time_fmt = str(datetime.timedelta(seconds=int(start_time)))
            end_time_fmt = str(datetime.timedelta(seconds=int(end_time)))
            
            segments.append({
                "speaker": speaker,
                "start_time": start_time_fmt,
                "end_time": end_time_fmt,
                "text": segment_result["text"].strip()
            })
        
        # Save results in different formats
        output_data = {
            "metadata": {
                "file": audio_path,
                "date_processed": datetime.datetime.now().isoformat(),
                "model_info": {
                    "asr": "whisper-large-v3",
                    "diarization": "pyannote/speaker-diarization-3.0"
                }
            },
            "segments": segments
        }
        
        # Save JSON
        json_path = f"{output_base_path}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Save Markdown
        md_path = f"{output_base_path}.md"
        self._save_markdown(output_data, md_path)
        
        # Save Plain Text
        txt_path = f"{output_base_path}.txt"
        self._save_text(output_data, txt_path)
        
        print(f"Transcription completed and saved as:")
        print(f"- JSON: {json_path}")
        print(f"- Markdown: {md_path}")
        print(f"- Text: {txt_path}")
        
        return output_data

    def _save_markdown(self, data, output_path):
        """Save transcription in a reader-friendly Markdown format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("# Interview Transcript\n\n")
            f.write(f"*Processed: {data['metadata']['date_processed'][:10]}*\n\n")
            
            # Write summary
            total_segments = len(data['segments'])
            unique_speakers = len(set(seg['speaker'] for seg in data['segments']))
            f.write(f"## Summary\n")
            f.write(f"- Unique speakers: {unique_speakers}\n")
            f.write(f"- Number of segments: {total_segments}\n\n")
            
            # Write transcript
            f.write("## Transcript\n\n")
            for segment in data['segments']:
                f.write(f"### {segment['speaker']} ({segment['start_time']} - {segment['end_time']})\n\n")
                f.write(f"{segment['text']}\n\n")

    def _save_text(self, data, output_path):
        """Save transcription in a simple text format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("INTERVIEW TRANSCRIPT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Käsitelty: {data['metadata']['date_processed'][:10]}\n\n")
            
            # Write summary
            total_segments = len(data['segments'])
            unique_speakers = len(set(seg['speaker'] for seg in data['segments']))
            f.write("SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Unique speakers: {unique_speakers}\n")
            f.write(f"Number of segments: {total_segments}\n\n")
            
            # Write transcript
            f.write("LITTEROINTI:\n")
            f.write("-" * 20 + "\n\n")
            for segment in data['segments']:
                f.write(f"{segment['speaker']} ({segment['start_time']} - {segment['end_time']})\n")
                f.write("-" * 40 + "\n")
                f.write(f"{segment['text']}\n\n")

def main():
    # You need to get this token from https://hf.co/pyannote/speaker-diarization
    HF_TOKEN = "YOUR-TOKEN-HERE"
    
    # Initialize transcriber
    transcriber = InterviewTranscriber(HF_TOKEN)
    
    # Process interview, replace with your information
    interview_path = "interview.mp3"
    output_base_path = "interview_transcription"
    
    try:
        result = transcriber.transcribe_interview(interview_path, output_base_path)
        
        # Print sample of results
        print("\nSample from interview:")
        for segment in result["segments"][:3]:
            print(f"\n{segment['speaker']} ({segment['start_time']} - {segment['end_time']}):")
            print(f"{segment['text']}")
            
    except Exception as e:
        print(f"Virhe litteroinnissa: {str(e)}")

if __name__ == "__main__":
    main()