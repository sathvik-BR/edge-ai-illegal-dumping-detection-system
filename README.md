# 🚨 SENTINEL — Edge AI Illegal Dumping Detection System

> **An edge-AI surveillance system that detects illegal dumping in real time using a fine-tuned YOLOv8 model, a HOLDING → RELEASED → DEPARTED evidence-based state machine, Telegram alerts, and a live Flask dashboard.**
>
> Supports multiple simultaneous camera sources (laptop webcam + phone via IP Webcam).

**Project by:** AMC Engineering College, Bengaluru · VTU Batch 2023–27 · Edge AI Major Project

---

## ✨ Features

- **Custom-trained YOLOv8s model** (`garbage` + `person` classes), fine-tuned on a Roboflow illegal-dumping dataset
- **Evidence-based dumping detection** — tracks `HOLDING → RELEASED → DEPARTED` sequences instead of single-frame guesses, scored across multiple weighted signals (held, released, departed, stationary, background)
- **Multi-camera support** — run any number of cameras (laptop webcam, phone via IP Webcam app) simultaneously, each independently tracked
- **OpenVINO-accelerated inference** for real-time performance on CPU-only laptops
- **Telegram alerts** with photo evidence on confirmed detections
- **Live web dashboard (Flask)** — real system stats, live camera feeds, detection timeline, alert log with pagination, settings panel to clear old data

---

## 🏗️ Architecture

```text
detect.py / detect_cam2.py   →  writes static/live_frame_<CAM>.jpg,
   (one process per camera)     static/live_status_<CAM>.json, log.csv

app.py (Flask)                →  reads those files, serves the dashboard
   + templates/index.html        and JSON APIs (/api/status, /api/alerts, ...)
```

Each camera is a fully independent process — the dashboard reads their output files rather than sharing the camera device directly, which avoids device-lock conflicts and lets you scale to more cameras just by running more copies of `detect.py` with different environment variables.

---

## ⚙️ Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 2. Get the trained model weights

Model weights are not stored in this repo (kept out via `.gitignore` — binary files bloat git history).

Download `best.pt` from:

**https://drive.google.com/file/d/1nDwYC6pj08wDdcYP5WxFsSVfMpvBS7aq/view?usp=sharing**

Place it at:

```text
runs/detect/train/weights/best.pt
```

Update the `MODEL` path in `detect.py`'s `CFG` class if you place it elsewhere.

### 3. Get the training dataset (optional — only needed to retrain)

This project was trained on the Roboflow **"Illegal Dumping Detection"** dataset:

- **v1 (404 images):** https://universe.roboflow.com/bill-lhxqf/illegal-dumping-detection/dataset/2
- **v2 (8,879 images, used for the final model):** https://universe.roboflow.com/bill-lhxqf/illegal-dumping-detection-2/dataset/1

Download via the Roboflow **"Show download code"** option (YOLOv8 format) — do not commit the dataset itself to this repo.

### 4. Set up Telegram alerts (optional)

Create a bot via **[@BotFather](https://t.me/BotFather)**, then set:

```bash
# Windows PowerShell
$env:SENTINEL_TG_TOKEN="your-bot-token"
$env:SENTINEL_TG_CHAT_ID="your-chat-id"

# macOS/Linux
export SENTINEL_TG_TOKEN="your-bot-token"
export SENTINEL_TG_CHAT_ID="your-chat-id"
```

If unset, the system runs fine and just skips sending Telegram alerts.

---

## ▶️ Running it

### Start the dashboard

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

### Start detection — laptop webcam

Start detection in a separate terminal:

```powershell
# Windows PowerShell
$env:SENTINEL_CAM_ID="CAM-01"
$env:SENTINEL_CAM_SOURCE="0"
python detect.py
```

### Start detection — additional camera

For an additional camera (e.g. a phone via the IP Webcam Android app), in another terminal:

```powershell
$env:SENTINEL_CAM_ID="CAM-02"
$env:SENTINEL_CAM_SOURCE="http://<phone-ip>:8080/video"
python detect_cam2.py
```

---

## 🧰 Tech Stack

`YOLOv8` · `OpenCV` · `OpenVINO` · `Flask` · `Telegram Bot API` · `Chart.js` · `psutil`

---

## 📊 Results

Fine-tuned model validation performance (final version, 8,879-image dataset, ~68 epochs with early stopping):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|:---|---:|---:|---:|---:|
| garbage | 0.859 | 0.816 | 0.899 | 0.591 |
| person | 0.782 | 0.831 | 0.813 | 0.434 |
| **all** | **0.821** | **0.824** | **0.856** | **0.513** |

---

## 📄 License

Academic project — for coursework/demonstration purposes.
