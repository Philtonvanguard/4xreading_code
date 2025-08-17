import os
import subprocess
import pymupdf  # Correct import name
from ebooklib import epub
from bs4 import BeautifulSoup
from gtts import gTTS
from moviepy import TextClip, AudioFileClip
import torch
from transformers import pipeline, logging, AutoTokenizer, AutoModelForSeq2SeqLM

# ========== SETUP ==========
logging.set_verbosity_info()  # Hugging Face download logs

# ========== STEP 1: EXTRACT TEXT ==========
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        print("   Opening PDF...", flush=True)
        doc = pymupdf.open(file_path)
        return "".join(page.get_text() for page in doc)
    elif file_path.endswith(".epub"):
        print("   Opening EPUB...", flush=True)
        book = epub.read_epub(file_path)
        text = ""
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text()
        return text
    else:
        raise ValueError("Unsupported file type. Use PDF or EPUB.")

# ========== STEP 2: SUMMARIZE TEXT ==========
def summarize(text, force_device="auto"):
    """
    Summarize text using Hugging Face model.
    force_device can be: "auto", "cpu", or "cuda"
    """
    print("   Loading summarization model...", flush=True)

    # Determine device
    if force_device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
            print(f"   ✅ GPU detected: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            device_str = "cpu"
            print("   ⚠️ No GPU detected, running on CPU.", flush=True)
    elif force_device == "cuda":
        if torch.cuda.is_available():
            device_str = "cuda"
            print(f"   ✅ Forcing GPU: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            raise RuntimeError("CUDA requested but no GPU available.")
    else:
        device_str = "cpu"
        print("   ⚠️ Forcing CPU mode.", flush=True)

    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # If using auto placement, let accelerate handle it
    if device_str == "auto" or device_str == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    else:
        # CPU only mode
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

    # Split text into chunks for summarization
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"   Summarizing chunk {i}/{len(chunks)}...", flush=True)
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(result)

    return " ".join(summaries)

# ========== STEP 3: TEXT-TO-SPEECH (4X SPEED) ==========
def generate_audio(text, output_path="output/fast_audio.mp3"):
    print("   Generating TTS audio...", flush=True)
    raw_audio_path = "output/audio.mp3"
    tts = gTTS(text, lang='en', slow=False)
    tts.save(raw_audio_path)

    print("   Speeding audio with ffmpeg (4x)...", flush=True)
    subprocess.run(
        ['ffmpeg', '-y', '-i', raw_audio_path, '-filter:a', 'atempo=2.0,atempo=2.0', output_path],
        check=True
    )
    return output_path

# ========== STEP 4: CREATE SUBTITLES ==========
def generate_srt(text, output_path="output/subs.srt"):
    print("   Generating subtitles...", flush=True)
    words = text.split()
    lines = [" ".join(words[i:i+8]) for i in range(0, len(words), 8)]
    srt = ""
    for i, line in enumerate(lines):
        start = i * 2
        end = start + 2
        srt += f"{i+1}\n00:00:{start:02},000 --> 00:00:{end:02},000\n{line}\n\n"
    with open(output_path, "w") as f:
        f.write(srt)
    return output_path

# ========== STEP 5: CREATE VIDEO ==========
def create_video(text, audio_path, output_path="output/final_video.mp4"):
    print("   Creating video preview...", flush=True)
    preview_text = text[:200] + "..."
    audio_clip = AudioFileClip(audio_path)
    txt_clip = TextClip(
        preview_text,
        fontsize=48,
        color='white',
        size=(1280, 720),
        method='caption'
    ).set_duration(audio_clip.duration)
    
    video = txt_clip.set_audio(audio_clip)
    print("   Rendering final video (this may take a few minutes)...", flush=True)
    video.write_videofile(output_path, fps=24)

# ========== RUN EVERYTHING ==========
def main():
    input_path = r"C:\Users\pmspi\4X Book Speed Code\4x reading project\Your_book.pdf\
    os.makedirs("output", exist_ok=True)

    print("[1] Extracting text...", flush=True)
    text = extract_text(input_path)

    print("[2] Summarizing...", flush=True)
    summary = summarize(text, force_device="auto")  # change to "cpu" or "cuda" to override

    print("[3] Generating audio at 4x speed...", flush=True)
    audio_path = generate_audio(summary)

    print("[4] Creating subtitles...", flush=True)
    generate_srt(summary)

    print("[5] Rendering video...", flush=True)
    create_video(summary, audio_path)

    print("✅ Done! Video saved to output/final_video.mp4", flush=True)

if __name__ == "__main__":
    main()

