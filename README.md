# 🚨 SENTINEL --- Edge AI Illegal Dumping Detection System

> **Real-time Edge AI surveillance for evidence-based illegal dumping
> detection**

SENTINEL is an **Edge AI--powered illegal dumping detection system**
designed to monitor camera feeds in real time, identify garbage and
people using a fine-tuned **YOLOv8** model, and confirm potential
dumping events using an evidence-based state machine rather than relying
on a single-frame prediction.

The system combines **computer vision, OpenVINO acceleration,
multi-camera processing, event-state tracking, Telegram notifications,
and a live Flask dashboard** into one deployable monitoring pipeline.

**Academic Major Project · AMC Engineering College, Bengaluru · VTU
Batch 2023--27**

------------------------------------------------------------------------

## ✨ Key Features

  -----------------------------------------------------------------------
  Capability                          Description
  ----------------------------------- -----------------------------------
  🧠 **Custom YOLOv8 Detection**      Fine-tuned model detecting
                                      `garbage` and `person` classes

  🔎 **Evidence-Based Detection**     Uses a
                                      `HOLDING → RELEASED → DEPARTED`
                                      state machine to reduce false
                                      detections

  📊 **Multi-Signal Scoring**         Combines held, released, departed,
                                      stationary and background evidence

  📹 **Multi-Camera Support**         Supports laptop webcams and
                                      network/phone cameras
                                      simultaneously

  ⚡ **OpenVINO Acceleration**        Optimized CPU inference for edge
                                      devices and CPU-only laptops

  🚨 **Telegram Alerts**              Sends confirmed-event notifications
                                      with image evidence

  🖥️ **Live Dashboard**               Flask dashboard with camera feeds,
                                      system status, timeline and alerts

  📝 **Event Logging**                Stores detection and alert
                                      information for later analysis

  🔄 **Independent Camera Processes** Each camera runs independently,
                                      avoiding camera-device conflicts
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🎯 Problem Statement

Illegal waste dumping is difficult to monitor continuously using
conventional surveillance because human operators cannot watch multiple
camera feeds at all times.

SENTINEL addresses this by using **Edge AI computer vision** to
automatically analyze camera streams and identify potential dumping
behavior.

Instead of treating every frame containing a person and garbage as an
illegal-dumping event, SENTINEL evaluates the **sequence of events over
time**:

``` text
        PERSON + GARBAGE
              │
              ▼
        ┌─────────────┐
        │   HOLDING   │
        │ Person holds│
        │    garbage  │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  RELEASED   │
        │ Garbage is  │
        │    placed   │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  DEPARTED   │
        │ Person moves│
        │     away    │
        └──────┬──────┘
               │
               ▼
      🚨 CONFIRMED EVENT
```

This evidence-based approach is intended to distinguish a potential
dumping action from ordinary situations such as a person simply standing
near garbage.

------------------------------------------------------------------------

## 🏗️ System Architecture

``` text
                 ┌───────────────────────┐
                 │      CAMERA SOURCES   │
                 │                       │
                 │  CAM-01 Laptop Webcam │
                 │  CAM-02 Phone Camera  │
                 │  CAM-03 ...           │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    YOLOv8 Detection   │
                 │                       │
                 │  • garbage            │
                 │  • person             │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Evidence / Tracking   │
                 │                       │
                 │ HOLDING               │
                 │ RELEASED              │
                 │ DEPARTED              │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Event Scoring       │
                 │                       │
                 │ held                  │
                 │ released              │
                 │ departed              │
                 │ stationary            │
                 │ background            │
                 └───────────┬───────────┘
                             │
                 ┌───────────┴────────────┐
                 ▼                        ▼
       ┌──────────────────┐     ┌──────────────────┐
       │ Telegram Alert   │     │ Flask Dashboard  │
       │ + Photo Evidence │     │ Live Monitoring  │
       └──────────────────┘     └──────────────────┘
```

### Process Architecture

Each camera is handled by an independent detection process:

``` text
detect.py / detect_cam2.py
        │
        ├── YOLOv8 inference
        ├── tracking / evidence logic
        ├── live frame output
        ├── status JSON
        └── event logging
                 │
                 ▼
             Flask app
                 │
                 ├── Dashboard
                 ├── Camera feeds
                 ├── Detection timeline
                 ├── Alert log
                 └── JSON APIs
```

This architecture allows additional cameras to be added without making
the Flask dashboard directly control camera devices.

------------------------------------------------------------------------

## 🧠 Detection Logic

SENTINEL uses temporal evidence instead of a simple:

> `person + garbage = dumping`

rule.

The system evaluates a sequence of observations:

### 1. HOLDING

A person is associated with garbage and the system gathers evidence that
the garbage is being carried or held.

### 2. RELEASED

The garbage becomes stationary or is otherwise separated from the
person's movement, providing evidence that it has been placed down.

### 3. DEPARTED

The person moves away from the released garbage.

### 4. Confirmation

The collected signals are combined into a weighted event score. When the
required evidence is satisfied, the system records a confirmed dumping
event and can send an alert.

``` text
HOLDING
   ↓
RELEASED
   ↓
DEPARTED
   ↓
Evidence threshold satisfied
   ↓
🚨 Illegal dumping event
```

------------------------------------------------------------------------

## 📁 Project Structure

``` text
edge-ai-illegal-dumping-detection-system/
│
├── dataset/                    # Training/evaluation data
├── static/                     # Dashboard assets and live outputs
├── templates/                  # Flask HTML templates
│
├── app.py                     # Flask dashboard
├── detect.py                  # Primary camera detection pipeline
├── detect_cam2.py             # Additional/network camera pipeline
│
├── dump_log.csv               # Dumping-event records
├── log.csv                    # Detection logs
├── log.json                   # JSON event/status data
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions
├── README.md                   # Project documentation
│
└── runs/
    └── detect/
        └── train/
            └── weights/
                └── best.pt    # Fine-tuned model weights
```

> **Note:** Model binaries and datasets should generally be kept outside
> Git history when they are large. If your repository already contains
> large model files, keep only the files required for reproducible
> deployment.

------------------------------------------------------------------------

## 🛠️ Tech Stack

### Artificial Intelligence

-   **YOLOv8**
-   **OpenCV**
-   **OpenVINO**
-   Python

### Backend & Dashboard

-   **Flask**
-   HTML
-   CSS
-   JavaScript
-   **Chart.js**

### Notifications & System Utilities

-   **Telegram Bot API**
-   **psutil**

### Development

-   Git
-   GitHub
-   VS Code

------------------------------------------------------------------------

## 🚀 Installation

### 1. Clone the repository

``` bash
git clone https://github.com/sathvik-BR/edge-ai-illegal-dumping-detection-system.git
cd edge-ai-illegal-dumping-detection-system
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

For a virtual environment:

``` bash
python -m venv .venv
```

**Windows PowerShell:**

``` powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🤖 Model Setup

The system uses a custom-trained YOLOv8 model containing:

``` text
0 → garbage
1 → person
```

If the trained weights are not included in the repository, download
`best.pt` from the project's model storage and place it at:

``` text
runs/detect/train/weights/best.pt
```

Alternatively, update the model path in the configuration used by
`detect.py`.

### Training Dataset

The model was trained using the Roboflow **Illegal Dumping Detection**
dataset.

-   **v1:** 404 images
-   **v2:** 8,879 images --- used for the final model

Dataset sources:

-   [Roboflow Illegal Dumping Detection ---
    v1](https://universe.roboflow.com/bill-lhxqf/illegal-dumping-detection/dataset/2)
-   [Roboflow Illegal Dumping Detection ---
    v2](https://universe.roboflow.com/bill-lhxqf/illegal-dumping-detection-2/dataset/1)

> The dataset does not need to be committed to this repository unless
> required for a specific academic submission.

------------------------------------------------------------------------

## 📱 Phone Camera Setup

SENTINEL can use an Android phone as a network camera through the **IP
Webcam** application.

Start the IP Webcam server on the phone and note its IPv4 address.

Example:

``` text
http://192.168.1.76:8080
```

The video endpoint is:

``` text
http://192.168.1.76:8080/video
```

Make sure the computer and phone can communicate over the same local
network.

------------------------------------------------------------------------

## ▶️ Running the System

### Start the Flask dashboard

Open a terminal:

``` bash
python app.py
```

Then open:

``` text
http://127.0.0.1:5000/
```

------------------------------------------------------------------------

### Camera 1 --- Laptop Webcam

Open a second terminal.

**Windows PowerShell:**

``` powershell
$env:SENTINEL_CAM_ID="CAM-01"
$env:SENTINEL_CAM_SOURCE="0"
python detect.py
```

------------------------------------------------------------------------

### Camera 2 --- Phone / IP Webcam

Open another terminal:

``` powershell
$env:SENTINEL_CAM_ID="CAM-02"
$env:SENTINEL_CAM_SOURCE="http://<phone-ip>:8080/video"
python detect_cam2.py
```

Example:

``` powershell
$env:SENTINEL_CAM_ID="CAM-02"
$env:SENTINEL_CAM_SOURCE="http://192.168.1.76:8080/video"
python detect_cam2.py
```

------------------------------------------------------------------------

## 🚨 Telegram Alerts

Telegram notifications are optional.

### 1. Create a Telegram bot

Use **BotFather** to create a bot and obtain the bot token.

### 2. Configure environment variables

**Windows PowerShell:**

``` powershell
$env:SENTINEL_TG_TOKEN="your-bot-token"
$env:SENTINEL_TG_CHAT_ID="your-chat-id"
```

**macOS/Linux:**

``` bash
export SENTINEL_TG_TOKEN="your-bot-token"
export SENTINEL_TG_CHAT_ID="your-chat-id"
```

If these variables are not configured, the detection system can still
run without Telegram alerts.

> **Security:** Never commit bot tokens, API keys, passwords, or other
> secrets to GitHub.

------------------------------------------------------------------------

## 📊 Model Performance

Final validation results reported for the fine-tuned model trained on
the 8,879-image dataset:

  Class           Precision      Recall       mAP50    mAP50-95
  ------------- ----------- ----------- ----------- -----------
  garbage             0.859       0.816       0.899       0.591
  person              0.782       0.831       0.813       0.434
  **Overall**     **0.821**   **0.824**   **0.856**   **0.513**

Training configuration reported for the final version:

``` text
Dataset:        8,879 images
Training:       ~68 epochs
Stopping:       Early stopping
Model:          Fine-tuned YOLOv8
Classes:        garbage, person
```

------------------------------------------------------------------------

## 🖥️ Dashboard

The Flask dashboard provides a centralized interface for monitoring the
detection system.

It is designed to expose:

-   📹 Live camera feeds
-   🟢 Camera/system status
-   🚨 Confirmed dumping events
-   📈 Detection statistics
-   🕒 Detection timeline
-   📋 Alert history
-   📄 Paginated logs
-   ⚙️ Data/settings controls

The dashboard reads the outputs produced by the individual camera
processes instead of directly taking ownership of the camera devices.

------------------------------------------------------------------------

## 🔧 Configuration

Important environment variables include:

  Variable                Purpose                    Example
  ----------------------- -------------------------- ----------------
  `SENTINEL_CAM_ID`       Unique camera identifier   `CAM-01`
  `SENTINEL_CAM_SOURCE`   Camera/video source        `0`
  `SENTINEL_TG_TOKEN`     Telegram bot token         `your-token`
  `SENTINEL_TG_CHAT_ID`   Telegram destination       `your-chat-id`

For multiple cameras, assign each camera a unique ID:

``` text
CAM-01
CAM-02
CAM-03
...
```

------------------------------------------------------------------------

## 🧪 Example Multi-Camera Deployment

``` text
Terminal 1
└── python app.py

Terminal 2
└── CAM-01 → Laptop webcam

Terminal 3
└── CAM-02 → Phone IP camera

Terminal 4
└── CAM-03 → Additional network camera
```

All camera processes can feed their independent outputs into the same
Flask dashboard.

------------------------------------------------------------------------

## 📈 Why Edge AI?

SENTINEL is designed around edge inference so that camera frames can be
processed locally rather than requiring every frame to be uploaded to a
remote cloud service.

### Benefits

-   ⚡ Lower inference latency
-   🔒 Local video processing
-   🌐 Reduced dependence on cloud connectivity
-   💻 Can operate on CPU-based systems
-   📹 Suitable for real-time surveillance scenarios

OpenVINO is used to optimize inference for supported CPU hardware.

------------------------------------------------------------------------

## 🔐 Security & Privacy

This project is intended for academic and demonstration purposes.

When deploying the system:

-   Do not commit API keys or Telegram tokens.
-   Do not expose camera streams publicly without authentication.
-   Do not expose the Flask development server directly to the internet.
-   Protect stored detection images and logs.
-   Follow applicable privacy and surveillance regulations.
-   Obtain appropriate authorization before monitoring real locations.

------------------------------------------------------------------------

## 🧩 Troubleshooting

### Camera does not open

Check that the camera source is correct.

For a webcam:

``` powershell
$env:SENTINEL_CAM_SOURCE="0"
```

For IP Webcam:

``` powershell
$env:SENTINEL_CAM_SOURCE="http://<phone-ip>:8080/video"
```

Test network connectivity from Windows:

``` powershell
Test-NetConnection <phone-ip> -Port 8080
```

The result should contain:

``` text
TcpTestSucceeded : True
```

### Model not found

Confirm that the model exists at the configured path:

``` text
runs/detect/train/weights/best.pt
```

### Telegram alerts not working

Verify:

``` text
SENTINEL_TG_TOKEN
SENTINEL_TG_CHAT_ID
```

and ensure the values are valid.

### Dashboard shows no live camera

Make sure the corresponding detection process is running and producing
its live frame/status files.

------------------------------------------------------------------------

## 🗺️ Future Enhancements

Potential future improvements include:

-   Multi-object tracking with stronger identity persistence
-   Improved dumping-event temporal reasoning
-   Automatic camera health monitoring
-   Cloud/mobile dashboard deployment
-   Additional waste-category detection
-   GPS/location-aware incident reporting
-   Automatic incident reports
-   Edge deployment on Raspberry Pi / NVIDIA Jetson-class hardware
-   Model quantization and further inference optimization
-   Long-term analytics for hotspot detection

------------------------------------------------------------------------

## 👨‍💻 Project

**SENTINEL --- Edge AI Illegal Dumping Detection System**

**Institution:** AMC Engineering College, Bengaluru\
**Program:** Computer Science & Engineering --- Artificial Intelligence
& Machine Learning\
**Batch:** VTU 2023--2027\
**Project Type:** Major Project

### Authors

**B R Sathvik**

------------------------------------------------------------------------

## 📄 License

This project is an **academic major project** developed for coursework,
research, demonstration, and educational purposes.

------------------------------------------------------------------------

::: {align="center"}
### 🚨 SENTINEL

**Detect. Verify. Alert.**

*Edge AI for smarter waste-management surveillance.*
:::
