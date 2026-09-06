# PixelForge AI

A desktop AI-powered photo & video editor built with Python, PySide6, and OpenCV — combining classic image-processing tools with local AI models for enhancement, background removal, upscaling, and generation.

<!-- Add a screenshot or short GIF/demo here once available:
![PixelForge AI screenshot](docs/screenshot.png)
-->

## Features

- **Enhance** — brightness/contrast/color adjustments, crop & straighten, auto white balance, one-click auto-enhance
- **AI Upscale** — resolution upscaling using a local ONNX super-resolution model
- **Filters** — presets, custom filter stacks, live preview, favorites
- **Remove Background** — AI-powered background removal and replacement
- **AI Image Generation** — prompt-based image generation with prompt history
- **Portrait / Face Restore** — AI face restoration and retouching
- **Batch Processing** — apply edits across many images at once, with duplicate detection, disk-space checks, and per-item retry/skip
- **Video Studio** — clip sequencing, audio tracks, text overlays, watermarks, SRT subtitle import, and video export
- **Projects** — organize edits into projects with full edit history, thumbnails, and recovery
- **Session-based Undo/Redo** — a single shared working image with global undo/redo across every tool, plus save/load `.pfproj` project files

## Tech Stack

- **UI:** PySide6 (Qt for Python) + a QtWebEngine-based HTML/CSS/JS frontend, connected via `QWebChannel`
- **Image processing:** OpenCV, Pillow, NumPy
- **AI inference:** ONNX Runtime (upscaling, face restoration), rembg (background removal), PyTorch + diffusers (image generation)
- **Video:** FFmpeg (external dependency, not bundled)
- **Storage:** SQLite (projects, media, edit history, presets)
- **Testing:** pytest

## Getting Started

### Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available on your system `PATH` (only required for the Video Studio; the rest of the app works without it)

### Installation

```bash
git clone https://github.com/AbdulRehman3376/PixelForge-AI.git
cd PixelForge-AI

python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### Running

```bash
python main.py
```

On first use, the AI features (upscaling, background removal, face restoration) download their model weights automatically — an internet connection is needed for that first run.

## Running Tests

```bash
pytest
```

## Project Structure

```
ai/         AI models: upscaling, background removal, face restoration, image generation
core/       Image processing, video studio, database, sessions, batch processing
ui/         PySide6 window + the Python<->JS bridge
frontend/   HTML/CSS/JS UI rendered inside QtWebEngine
tests/      pytest test suite
main.py     Application entry point
```

## License

Not yet licensed — add a license file if you intend others to reuse this code.
