"""
SENTINEL Dashboard Backend — v3.0  [MULTI-CAMERA]

WHAT CHANGED FROM v2.0:
  - Supports multiple cameras. Each camera (detect.py / detect_cam2.py,
    run with different SENTINEL_CAM_ID env vars) writes its own
    static/live_status_<CAM_ID>.json and static/live_frame_<CAM_ID>.jpg.
    This backend now checks ALL configured cameras (see CAMERAS list
    below) independently, so "Active Cameras" reflects how many are
    ACTUALLY running right now, not a single hardcoded source.
  - Alert log rows now report the real camera that logged them, parsed
    from the "Camera:CAM-0X" tag detect.py v8 writes into the Details
    column. Falls back to CAM-01 for older rows logged before this
    existed.
"""

from flask import Flask, render_template, jsonify, request
import csv
import os
import json
import time
import re
from datetime import datetime, timedelta
from collections import Counter

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

app = Flask(__name__)

LOG_FILE       = "log.csv"
STATIC_DIR     = "static"
LIVE_STALE_SEC = 10   # if a camera's status file is older than this, it's OFFLINE

# All cameras the dashboard knows about. CAM-03 stays permanently
# unassigned/offline unless you add a third detect.py process for it.
CAMERAS = ["CAM-01", "CAM-02", "CAM-03"]

PER_PAGE = 50


# ══════════════════════════════════════════════════════════
#  LOG READING / PARSING
# ══════════════════════════════════════════════════════════
def read_logs():
    """Returns list of dicts, newest first. CSV columns from detect.py:
    Time, Status, Image, Confidence, Evidence, Details"""
    rows = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

    if rows and rows[0] and rows[0][0] == "Time":
        rows = rows[1:]

    rows.reverse()  # newest first

    logs = []
    for i, r in enumerate(rows):
        r = r + [""] * (6 - len(r))  # pad short/old rows safely
        details = r[5]
        cam_match = re.search(r"Camera:(CAM-\d+)", details)
        camera = cam_match.group(1) if cam_match else "CAM-01"
        logs.append({
            "idx":        i,
            "time":       r[0],
            "status":     r[1],
            "image":      r[2],
            "confidence": r[3],
            "evidence":   r[4],
            "details":    details,
            "camera":     camera,
        })
    return logs


def confidence_pct(conf_str):
    try:
        v = float(conf_str)
        if v <= 1.0:
            v *= 100
        return round(v)
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════
#  LIVE STATUS / SYSTEM STATS  (per camera)
# ══════════════════════════════════════════════════════════
def get_live_status_for(cam_id):
    path = os.path.join(STATIC_DIR, f"live_status_{cam_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        age = time.time() - data.get("timestamp", 0)
        if age > LIVE_STALE_SEC:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def get_system_stats():
    if not HAS_PSUTIL:
        return {"available": False}
    return {
        "available": True,
        "cpu":  round(psutil.cpu_percent(interval=0.1)),
        "ram":  round(psutil.virtual_memory().percent),
        "disk": round(psutil.disk_usage("/").percent) if os.name != "nt"
                else round(psutil.disk_usage("C:\\").percent),
        "net":  min(round(sum(psutil.net_io_counters()[:2]) / 1_000_000 % 100), 100),
    }


def compute_threat_level(logs):
    if not logs:
        return 0, "NOMINAL"

    cutoff = datetime.now() - timedelta(minutes=15)
    recent = []
    for log in logs[:20]:
        try:
            t = datetime.strptime(log["time"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if t >= cutoff:
            recent.append(log)

    if not recent:
        return 5, "NOMINAL"

    confs = [confidence_pct(l["confidence"]) or 70 for l in recent]
    avg_conf = sum(confs) / len(confs)
    freq_score = min(len(recent) * 15, 60)
    level = min(round(freq_score + avg_conf * 0.4), 100)

    if level >= 70:   status = "CRITICAL"
    elif level >= 40: status = "ELEVATED"
    else:             status = "NOMINAL"
    return level, status


# ══════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════
@app.route("/")
def home():
    logs = read_logs()
    page1 = logs[:PER_PAGE]

    times = [l["time"][:13] for l in logs if l["time"]]
    counts = Counter(times)
    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]

    threat_level, threat_status = compute_threat_level(logs)

    return render_template(
        "index.html",
        logs=page1,
        total_logs=len(logs),
        per_page=PER_PAGE,
        labels=labels,
        values=values,
        threat_level=threat_level,
        threat_status=threat_status,
        has_psutil=HAS_PSUTIL,
    )


@app.route("/api/alerts")
def api_alerts():
    page = max(int(request.args.get("page", 1)), 1)
    logs = read_logs()
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    page_logs = logs[start:end]
    for l in page_logs:
        l["confidence_pct"] = confidence_pct(l["confidence"])
    return jsonify({
        "logs": page_logs,
        "page": page,
        "per_page": PER_PAGE,
        "total": len(logs),
        "has_more": end < len(logs),
    })


@app.route("/api/alerts/<int:idx>")
def api_alert_detail(idx):
    logs = read_logs()
    if idx < 0 or idx >= len(logs):
        return jsonify({"error": "not found"}), 404
    log = logs[idx]
    log["confidence_pct"] = confidence_pct(log["confidence"])
    return jsonify(log)


@app.route("/api/status")
def api_status():
    cams = {cid: get_live_status_for(cid) for cid in CAMERAS}
    active_count = sum(1 for v in cams.values() if v is not None)

    sysstats = get_system_stats()
    logs = read_logs()
    threat_level, threat_status = compute_threat_level(logs)

    recent_cutoff = time.time() - 3600
    recent_count = 0
    for l in logs[:30]:
        try:
            t = datetime.strptime(l["time"], "%Y-%m-%d %H:%M:%S").timestamp()
            if t >= recent_cutoff:
                recent_count += 1
        except ValueError:
            pass

    total_persons = sum((v.get("persons", 0) if v else 0) for v in cams.values())
    total_waste   = sum((v.get("waste", 0)   if v else 0) for v in cams.values())

    cam_details = {}
    for cid, v in cams.items():
        cam_details[cid] = {
            "active": v is not None,
            "state":  v.get("state", "OFFLINE") if v else "OFFLINE",
        }

    return jsonify({
        "cameras": cam_details,
        "active_camera_count": active_count,
        "camera_active": active_count > 0,
        "persons_tracked": total_persons,
        "waste_tracked": total_waste,
        "system": sysstats,
        "threat_level": threat_level,
        "threat_status": threat_status,
        "total_alerts": len(logs),
        "recent_alerts_1h": recent_count,
        "latest_alerts": logs[:5],
    })


@app.route("/api/settings/clear", methods=["POST"])
def api_clear():
    body = request.get_json(force=True, silent=True) or {}
    mode = body.get("mode", "days")
    days = int(body.get("days", 30))

    if not os.path.exists(LOG_FILE):
        return jsonify({"deleted": 0})

    with open(LOG_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header, data_rows = [], rows
    if rows and rows[0] and rows[0][0] == "Time":
        header, data_rows = rows[0], rows[1:]

    kept, deleted = [], 0
    cutoff = datetime.now() - timedelta(days=days)

    for r in data_rows:
        r_padded = r + [""] * (6 - len(r))
        should_delete = (mode == "all")
        if not should_delete and mode == "days":
            try:
                t = datetime.strptime(r_padded[0], "%Y-%m-%d %H:%M:%S")
                should_delete = t < cutoff
            except ValueError:
                should_delete = False

        if should_delete:
            deleted += 1
            img = r_padded[2]
            if img:
                img_path = os.path.join(STATIC_DIR, img)
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except OSError:
                        pass
        else:
            kept.append(r)

    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(kept)

    return jsonify({"deleted": deleted, "remaining": len(kept)})


if __name__ == "__main__":
    if not HAS_PSUTIL:
        print("[WARN] psutil not installed — system vitals will show as unavailable.")
        print("       Install with: pip install psutil")
    app.run(debug=True)