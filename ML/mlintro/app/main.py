import os
import logging
import uuid
import librosa
import numpy as np
import cv2
import torch
import gdown
import whisper
import subprocess
import psutil
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
from fpdf import FPDF
from datetime import datetime
from textwrap import wrap
from pathlib import Path
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi.middleware import Middleware
import nltk
from language_tool_python import LanguageTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(middleware=[Middleware(limiter)])

# Constants
WORK_DIR = Path("/var/video_processing")
MODEL_CACHE = WORK_DIR / "model_cache"
os.makedirs(MODEL_CACHE, exist_ok=True)

class VideoRequest(BaseModel):
    driveLink: str

def generate_pdf_report(results, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Color scheme
    navy = (0, 32, 96)
    dark_gray = (64, 64, 64)
    medium_gray = (128, 128, 128)

    # Header
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(*navy)
    pdf.cell(0, 10, "VIDEO ANALYSIS REPORT", 0, 1, 'C')
    pdf.ln(12)

    # Divider line
    pdf.set_draw_color(*navy)
    pdf.line(10, pdf.get_y()+5, 200, pdf.get_y()+5)
    pdf.ln(15)

    def add_section_header(title):
        y_before = pdf.get_y()
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(*navy)
        pdf.cell(0, 8, title, 0, 1)
        pdf.line(10, y_before + 11, 200, y_before + 11)
        pdf.ln(8)

    def add_wrapped_metric(label, value, label_color=navy, value_color=dark_gray):
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(*label_color)
        pdf.cell(55, 8, f"{label}:", 0, 0)
        pdf.set_font('Arial', '', 12)
        pdf.set_text_color(*value_color)
        wrapped = '\n'.join(wrap(str(value), width=60))
        pdf.multi_cell(0, 8, wrapped)
        pdf.ln(8)

    # Report sections
    add_section_header("Basic Checks")
    add_wrapped_metric("Face Detected", results.get('face_detected', 'No'))

    if 'transcription' in results:
        add_section_header("Transcript")
        pdf.set_font('Arial', '', 11)
        pdf.set_text_color(*dark_gray)
        pdf.multi_cell(0, 6, results['transcription'].get('english', ''))
        pdf.ln(10)

    add_section_header("Speech Analysis")
    filler_display = "0 found"
    if results.get('filler_words'):
        filler_list = ', '.join(set(results['filler_words']))
        filler_display = f"{len(results['filler_words'])} found ({filler_list})"
    
    speech_metrics = [
        ("Speech Rate (wpm)", f"{results.get('speech_rate', 0):.1f}"),
        ("Grammar Accuracy", f"{results.get('grammar_score', 0):.1f}%"),
        ("Filler Words", filler_display),
        ("Speech Tone", results.get('speech_tone', 'N/A'))
    ]
    for label, value in speech_metrics:
        add_wrapped_metric(label, value)

    add_section_header("Content Evaluation")
    add_wrapped_metric("Confidence Level", results.get('confidence_score', 'N/A'))
    add_wrapped_metric("Nervousness Level", results.get('nervousness_score', 'N/A'))
    add_wrapped_metric("Dominant Emotion", results.get('emotion', 'neutral').capitalize())
    
    # Footer
    pdf.set_y(-20)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(*medium_gray)
    pdf.cell(0, 10, f'Report generated on {datetime.now().strftime("%Y-%m-%d at %H:%M")}', 0, 0, 'C')

    pdf.output(filename)

@app.post("/process_video/")
@limiter.limit("5/minute")
async def process_video(request: VideoRequest):
    try:
        process_id = uuid.uuid4().hex[:8]
        logger.info(f"Starting processing ID: {process_id}")
        
        process_dir = WORK_DIR / process_id
        os.makedirs(process_dir, exist_ok=True)
        
        output_path = process_dir / "source_video"
        mp4_path = process_dir / "converted.mp4"
        audio_path = process_dir / "audio.wav"
        pdf_path = process_dir / "report.pdf"

        try:
            file_id = request.driveLink.split("/d/")[1].split("/")[0]
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}", 
                str(output_path), 
                quiet=True
            )
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid Google Drive link")

        # Video conversion
        if str(output_path).endswith(".webm"):
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(output_path),
                        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                        "-c:a", "aac", "-b:a", "128k",
                        "-movflags", "+faststart",
                        "-max_muxing_queue_size", "1024",
                        "-strict", "experimental",
                        str(mp4_path)
                    ],
                    check=True,
                    timeout=300
                )
            except subprocess.TimeoutExpired:
                logger.error("Video conversion timed out")
                raise HTTPException(status_code=504, detail="Video conversion timeout")
        else:
            mp4_path = output_path

        # Audio extraction
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(mp4_path), "-vn", "-ar", "16000", "-ac", "1", str(audio_path)],
                check=True,
                timeout=300
            )
        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            raise HTTPException(status_code=504, detail="Audio processing timeout")

        results = {}

        def transcribe_audio():
            if not os.path.exists(audio_path):
                return

            results["transcription"] = {}
            english_transcript = app.state.whisper_model.transcribe(
                str(audio_path),
                language="en",
                fp16=torch.cuda.is_available(),
                temperature=0.2,
                beam_size=1,
                task="transcribe"
            )["text"]

            results["transcription"]["english"] = english_transcript
            text = results["transcription"]["english"]

            # Audio analysis
            y, sr = librosa.load(str(audio_path))
            duration = librosa.get_duration(y=y, sr=sr)
            speech_rate = len(text.split()) / (duration / 60)
            results["speech_rate"] = round(speech_rate, 2)

            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[pitches > 0]
            avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            results["speech_tone"] = "Excited" if avg_pitch > 200 else "Calm" if avg_pitch < 120 else "Neutral"

            # Filler words detection
            filler_words = ["um", "uh", "like", "you know", "so"]
            results["filler_words"] = [w for w in nltk.word_tokenize(text.lower()) if w in filler_words]

            # Grammar check
            lang_tool = LanguageTool("en-US")
            grammar_errors = len(lang_tool.check(text))
            words_count = len(text.split())
            grammar_score = max(100 - (grammar_errors / words_count * 100), 0) if words_count > 0 else 100
            results["grammar_score"] = round(grammar_score, 2)

            # Confidence scoring
            filler_count = len(results["filler_words"])
            if speech_rate > 180 and filler_count > 5:
                results["confidence_score"] = "Moderate"
                results["nervousness_score"] = "High"
            elif speech_rate < 130 and filler_count < 3:
                results["confidence_score"] = "High"
                results["nervousness_score"] = "Low"
            else:
                results["confidence_score"] = "Moderate"
                results["nervousness_score"] = "Moderate"

        def analyze_video():
            cap = cv2.VideoCapture(str(mp4_path))
            face_mesh = mp_face.solutions.face_mesh.FaceMesh()
            frame_skip = 60
            frame_count, emotions = 0, []
            face_detected = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    try:
                        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                        emotions.append(analysis[0]['dominant_emotion'])
                        face_detected = True
                    except Exception as e:
                        pass

                frame_count += 1
            cap.release()

            results["emotion"] = max(set(emotions), key=emotions.count) if emotions else "Neutral"
            results["face_detected"] = "Yes" if face_detected else "No"

        # Execute analysis
        transcribe_audio()
        analyze_video()

        # Generate PDF
        generate_pdf_report(results, str(pdf_path))
        
        with open(pdf_path, "rb") as f:
            return Response(
                content=f.read(),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=report_{process_id}.pdf"}
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if 'process_dir' in locals():
            for attempt in range(3):
                try:
                    for file in process_dir.glob("*"):
                        file.unlink(missing_ok=True)
                    process_dir.rmdir()
                    break
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup attempt {attempt+1} failed: {str(cleanup_error)}")

@app.on_event("startup")
async def startup_event():
    """Preload ML models on startup"""
    logger.info("Preloading ML models...")
    
    try:
        logger.info("Loading Whisper model...")
        app.state.whisper_model = whisper.load_model(
            "large",
            download_root=MODEL_CACHE,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logger.info("Initializing DeepFace...")
        DeepFace.build_model("Facenet")
        
        logger.info("Model preloading complete")
    except Exception as e:
        logger.critical(f"Model loading failed: {str(e)}")
        raise RuntimeError("Failed to initialize ML models")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "disk_space": psutil.disk_usage("/").free // (1024*1024),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=2,
        timeout_keep_alive=120
    )