# 4xreading_code
📚 4X Reading Project  This project turns a book (PDF or EPUB) into a summarized, narrated, and captioned video — designed to help you consume books at 4x speed.
LLM anything that is not panning out
🚀 Features
📖 Extract Text → from PDF (via PyMuPDF) or EPUB (via EbookLib + BeautifulSoup)
✂️ Summarize → AI-powered summarization using HuggingFace Transformers (facebook/bart-large-cnn)
🎙 Text-to-Speech → Generate natural voice narration with gTTS
⏩ 4X Faster Audio → Speed boosted with FFmpeg (atempo=2.0,atempo=2.0)
📝 Subtitles (SRT) → Auto-generated subtitles, synced roughly by reading speed
🎥 Video Creation → Combines narration + preview text into a clean video with MoviePy
🛠️ Tech Stack
Python 3.10+
PyMuPDF → PDF parsing
EbookLib → EPUB parsing
BeautifulSoup4 → Text extraction from EPUB HTML
Transformers → AI Summarization
Torch with CUDA support for GPU acceleration
gTTS → Text-to-Speech
MoviePy → Video creation
FFmpeg → Audio processing

READ HERE FOR STEPS
1.Clone the repository
git clone https://github.com/YOUR_USERNAME/4x-reading-project.git
cd 4x-reading-project
2.Install dependencies
pip install -r requirements.txt
3.Install PyTorch with CUDA (GPU acceleration)
nvidia-smi
3a.Then install the correct Torch build ( CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121
4.Install FFmpeg,Windows: Download from FFmpeg builds and add bin/ to PATH.
*Linux (Debian/Ubuntu):
sudo apt install ffmpeg
*MacOS (Homebrew):
brew install ffmpeg

📂 Requirements:pip install x
x=(
pymupdf
ebooklib
beautifulsoup4
transformers
torch
gtts
moviepy
ffmpeg-python)

▶️ Usage
Place your book (.pdf or .epub) into the project folder.
Edit the input path in 4xreading_code.py:
input_path = "path/to/your/book.pdf"

Run the script:
python 4xreading_code.py
Outputs will be saved to the output/ folder:
fast_audio.mp3 → Summarized audiobook at 4x speed
subs.srt → Subtitles file
final_video.mp4 → Video with captions + narration
✅ Example Workflow
python 4xreading_code.py
Output:
output/fast_audio.mp3
output/subs.srt
output/final_video.mp4





