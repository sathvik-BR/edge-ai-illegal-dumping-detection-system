"""
SENTINEL — Illegal Dumping Detection Engine  v8.1  [STATE-RESET FIX]
=========================================================================
Tuned for: CPU-only laptops (Intel i5, no CUDA) — Dell Vostro class hardware.

WHAT CHANGED FROM v8.0 (this update):

  This is v8.0 with ONLY logic/state-management fixes added — NO distance
  thresholds (HOLD_DIST_PX, ARM_REACH_PX, etc.) were touched, since v8.0's
  values gave 90%+ accuracy in clean-scene testing. The problems below
  were state-management bugs, not distance-tuning problems.

  FIX 1 — stale "appeared near person" flag never decayed:
    Once a waste item got flagged appeared_near_person=True, that flag
    was PERMANENT for the life of the tracked object — there was no
    logic anywhere to un-flag it. So a pre-existing item that was near
    someone once, even in passing, stayed permanently "eligible" to
    trigger HOLDING again later.
    NEW: WTrack.frames_since_near counts how long it's been since a
    waste item was actually near ANY person. If it's been flagged
    appeared_near_person=True but hasn't been near anyone for
    NEAR_DECAY_FRAMES (~150 processed frames, ~20-30 real seconds), the
    flag resets to False. Genuine holding sequences reset this counter
    to 0 continuously, so they're unaffected.

  FIX 2 — a person's state never reset after DEPARTED:
    Once someone reached DEPARTED, nothing ever sent them back to IDLE.
    Combined with a longer-lived tracked ID, this meant a person could
    stay stuck in DEPARTED indefinitely, getting re-scored every cooldown
    period against whatever waste happened to be nearest — even far away
    and unrelated.
    NEW: after a real alert fires, the person's full state resets to
    IDLE (held_wid, rel_point, rel_time, fsr, pending_away_frames all
    cleared, sent reset to False so a genuine second incident later can
    still alert). MAX_DEPARTED_FRAMES also times out anyone stuck in
    DEPARTED who never actually crosses the alert threshold, resetting
    them to IDLE instead of lingering forever.

  Everything else — including HOLD_DIST_PX=110, ARM_REACH_PX=70, and
  every other tracking/distance value — is UNCHANGED from v8.0.
"""

import cv2, time, os, csv, math, threading, requests, json
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
class CFG:
    # ── Model ────────────────────────────────────────────
    MODEL        = r"C:\Users\DELL\runs\detect\train\weights\best.pt"   # YOUR fine-tuned model
    USE_OPENVINO = True
    CONF         = 0.30
    WASTE_CONF   = 0.35
    IOU_THRESH   = 0.45
    IMG_SIZE     = 480
    DUAL_SCALE   = False
    IMG_SIZE_2   = 800

    # ── Camera ───────────────────────────────────────────
    # CAM_ID identifies which camera this process IS (CAM-01, CAM-02...).
    # CAMERA_SOURCE is what it reads from: an integer for a local webcam
    # index (0, 1...), or a URL string for a phone/IP camera stream.
    # Both come from environment variables so ONE script file can run as
    # multiple different cameras — set the env vars before launching,
    # see the module docstring above for exact commands.
    CAM_ID        = os.getenv("SENTINEL_CAM_ID", "CAM-01")
    _src          = os.getenv("SENTINEL_CAM_SOURCE", "0")
    CAMERA_SOURCE = int(_src) if _src.isdigit() else _src

    FRAME_W       = 640
    FRAME_H       = 480
    FRAME_SKIP    = 2

    # ── Tracking ─────────────────────────────────────────
    MAX_TRACK_DIST = 280
    MAX_TRACK_AGE  = 8.0
    HISTORY_LEN    = 35

    # ── Dumping logic ────────────────────────────────────
    HOLD_DIST_PX       = 110
    RELEASE_DIST_PX    = 130
    DEPARTURE_DIST_PX  = 140
    STATIONARY_FRAMES  = 5
    STATIONARY_MOV_PX  = 16
    PREEXIST_FRAMES    = 25
    MIN_WASTE_DWELL    = 3
    MAX_WASTE_VEL      = 60
    ARM_REACH_PX        = 70
    REQUIRE_AWAY_FRAMES = 6
    MIN_ELAPSED_SEC     = 1.5
    NEAR_DECAY_FRAMES   = 150   # NEW — see FIX 1 in docstring above
    MAX_DEPARTED_FRAMES = 90    # NEW — see FIX 2 in docstring above

    # ── Background-subtraction fallback for un-classed litter ──
    ENABLE_BG_FALLBACK = False
    BG_FALLBACK_LABEL  = "unidentified item"
    BG_FALLBACK_CONF   = 0.50
    BG_MAX_AREA        = 22000

    # ── Evidence weights ─────────────────────────────────
    W_HELD       = 0.25
    W_RELEASED   = 0.20
    W_DEPARTED   = 0.25
    W_STATIONARY = 0.20
    W_BG         = 0.10
    MIN_SCORE    = 0.70

    # ── Background subtraction ───────────────────────────
    BG_HISTORY   = 120
    BG_THRESH    = 18
    BG_MIN_AREA  = 700
    BG_INTERVAL  = 3

    # ── Alert ────────────────────────────────────────────
    COOLDOWN_SEC   = 35
    CONFIRM_FRAMES = 6

    # ── Telegram (SECURITY: pulled from env, never hardcoded) ─
    TOKEN   = os.getenv("SENTINEL_TG_TOKEN", "")
    CHAT_ID = os.getenv("SENTINEL_TG_CHAT_ID", "")

    # ── Output ───────────────────────────────────────────
    STATIC_DIR = "static"
    LOG_FILE   = "log.csv"


# ══════════════════════════════════════════════════════════
#  WASTE LABELS
# ══════════════════════════════════════════════════════════
WASTE_LABELS = {"garbage"}


# ══════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════
def ctr(box):
    x1, y1, x2, y2 = box
    return (int((x1+x2)/2), int((y1+y2)/2))

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def intersect(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    return max(0, ix2-ix1) * max(0, iy2-iy1)

def expand(box, px):
    return (box[0]-px, box[1]-px, box[2]+px, box[3]+px)

def inside(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]

def iou(b1, b2):
    inter = intersect(b1, b2)
    if inter == 0: return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def containment_ratio(inner, outer):
    inter = intersect(inner, outer)
    if inter == 0: return 0.0
    a_inner = (inner[2]-inner[0]) * (inner[3]-inner[1])
    if a_inner <= 0: return 0.0
    return inter / a_inner

def nms_merge(dets, iou_th=0.45):
    if not dets:
        return []
    dets = sorted(dets, key=lambda x: x[2], reverse=True)
    kept = []
    used = [False] * len(dets)
    for i, d in enumerate(dets):
        if used[i]: continue
        kept.append(d)
        for j in range(i+1, len(dets)):
            if not used[j] and d[1] == dets[j][1]:
                if iou(d[0], dets[j][0]) > iou_th:
                    used[j] = True
    return kept


# ══════════════════════════════════════════════════════════
#  THREADED FRAME GRABBER
# ══════════════════════════════════════════════════════════
class FrameGrabber:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.FRAME_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock  = threading.Lock()
        self._frame = None
        self._ok    = False
        self._stop  = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            ok, f = self.cap.read()
            with self._lock:
                self._ok, self._frame = ok, f

    def read(self):
        with self._lock:
            return self._ok, (self._frame.copy() if self._frame is not None else None)

    def release(self):
        self._stop = True
        self.cap.release()


# ══════════════════════════════════════════════════════════
#  BACKGROUND SUBTRACTOR
# ══════════════════════════════════════════════════════════
class BG:
    def __init__(self):
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=CFG.BG_HISTORY, varThreshold=CFG.BG_THRESH, detectShadows=False)
        self.k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self.k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self._cache = []

    def update(self, frame):
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        mask = self.mog.apply(blurred, learningRate=0.006)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.k3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k5, iterations=2)
        blobs = []
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) >= CFG.BG_MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                blobs.append((x, y, x+w, y+h))
        self._cache = blobs
        return blobs

    def blobs(self):   return self._cache

    def contains(self, pt, margin=35):
        for b in self._cache:
            if inside(pt, expand(b, margin)):
                return True
        return False


# ══════════════════════════════════════════════════════════
#  SIMPLE KALMAN FILTER
# ══════════════════════════════════════════════════════════
class KalmanXY:
    def __init__(self, x, y):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],
                                                [0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost       = np.eye(4, dtype=np.float32)
        self.kf.statePost          = np.array([[x],[y],[0],[0]], np.float32)

    def predict(self):
        p = self.kf.predict()
        return int(p[0]), int(p[1])

    def correct(self, x, y):
        m = np.array([[np.float32(x)],[np.float32(y)]])
        self.kf.correct(m)

    @property
    def velocity(self):
        vx = float(self.kf.statePost[2])
        vy = float(self.kf.statePost[3])
        return math.hypot(vx, vy)


# ══════════════════════════════════════════════════════════
#  CENTROID TRACKER  (Kalman-enhanced)
# ══════════════════════════════════════════════════════════
class Tracker:
    def __init__(self, max_d, max_age):
        self._t   = {}
        self._kf  = {}
        self._nid = 0
        self.max_d   = max_d
        self.max_age = max_age

    def update(self, boxes):
        now = time.time()
        self._t  = {k: v for k, v in self._t.items()  if now - v['last_seen'] <= self.max_age}
        for tid in self._kf:
            if tid in self._t:
                px, py = self._kf[tid].predict()
                self._t[tid]['pred'] = (px, py)

        res = {}
        unmatched = list(boxes)

        for tid, t in list(self._t.items()):
            if not unmatched: break
            pred = t.get('pred', t['center'])
            bi, bd = None, float('inf')
            for i, b in enumerate(unmatched):
                d = dist(pred, ctr(b))
                if d < bd and d < self.max_d:
                    bd, bi = d, i
            if bi is not None:
                b = unmatched.pop(bi)
                cx, cy = ctr(b)
                self._kf[tid].correct(cx, cy)
                self._t[tid].update({'center': (cx, cy), 'last_seen': now, 'box': b})
                res[tid] = b

        for b in unmatched:
            tid = self._nid; self._nid += 1
            cx, cy = ctr(b)
            self._t[tid] = {'center': (cx, cy), 'last_seen': now, 'box': b}
            self._kf[tid] = KalmanXY(cx, cy)
            res[tid] = b
        return res

    def velocity(self, tid):
        kf = self._kf.get(tid)
        return kf.velocity if kf else 0.0


# ══════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════
class PS:
    IDLE = "IDLE"; HOLDING = "HOLDING"; RELEASED = "RELEASED"
    DEPARTED = "DEPARTED"; CONFIRMED = "CONFIRMED"

@dataclass
class PTrack:
    tid: int
    box: Tuple
    centers: deque = field(default_factory=lambda: deque(maxlen=CFG.HISTORY_LEN))
    last_seen: float = field(default_factory=time.time)
    state: str = PS.IDLE
    held_wid:  Optional[int]   = None
    rel_point: Optional[Tuple] = None
    rel_time:  Optional[float] = None
    fsr: int   = 0
    score: float = 0.0
    sent: bool   = False
    pending_away_frames: int = 0

    def upd(self, box):
        self.box = box
        self.centers.append(ctr(box))
        self.last_seen = time.time()

    @property
    def center(self): return ctr(self.box)

    def moving_away(self, pt, n=6):
        c = list(self.centers)
        if len(c) < n: return False
        return dist(c[-1], pt) > dist(c[-n], pt) + 10


@dataclass
class WTrack:
    tid: int
    box: Tuple
    label: str
    conf: float
    centers: deque = field(default_factory=lambda: deque(maxlen=CFG.HISTORY_LEN))
    last_seen: float = field(default_factory=time.time)
    stat_frames: int = 0
    frames_total: int = 0
    assoc_pid: Optional[int] = None
    appeared_near_person: bool = False
    dwell_frames: int = 0
    frames_since_near: int = 0   # NEW — see FIX 1 in module docstring

    def upd(self, box, conf):
        prev = ctr(self.box)
        self.box = box; self.conf = conf
        self.centers.append(ctr(box))
        self.last_seen = time.time()
        self.frames_total += 1
        self.dwell_frames += 1
        if dist(prev, ctr(box)) < CFG.STATIONARY_MOV_PX:
            self.stat_frames += 1
        else:
            self.stat_frames = max(0, self.stat_frames - 2)

    @property
    def center(self): return ctr(self.box)

    @property
    def stationary(self): return self.stat_frames >= CFG.STATIONARY_FRAMES

    @property
    def is_background(self):
        return (not self.appeared_near_person
                and self.frames_total > CFG.PREEXIST_FRAMES)

    @property
    def is_new_enough(self):
        return self.dwell_frames >= CFG.MIN_WASTE_DWELL

    def velocity(self):
        c = list(self.centers)
        if len(c) < 4: return 0.0
        return dist(c[-1], c[-4]) / 3.0


# ══════════════════════════════════════════════════════════
#  EVIDENCE SCORER
# ══════════════════════════════════════════════════════════
def score(pt: PTrack, wt: WTrack, bg: BG):
    ev = {}
    ev['held']     = CFG.W_HELD     if (pt.held_wid == wt.tid or wt.assoc_pid == pt.tid) else 0.0
    ev['released'] = CFG.W_RELEASED if pt.rel_point else 0.0

    if pt.rel_point:
        d = dist(pt.center, pt.rel_point)
        if   d >= CFG.DEPARTURE_DIST_PX:       ev['departed'] = CFG.W_DEPARTED
        elif d >= CFG.DEPARTURE_DIST_PX * 0.6: ev['departed'] = CFG.W_DEPARTED * 0.5
        else:                                   ev['departed'] = 0.0
    else:
        ev['departed'] = 0.0

    dwell_bonus = min(wt.dwell_frames / 30.0, 1.0)
    if wt.stationary:
        ev['stationary'] = CFG.W_STATIONARY * (0.8 + 0.2 * dwell_bonus)
    elif wt.stat_frames > 3:
        ev['stationary'] = CFG.W_STATIONARY * 0.4
    else:
        ev['stationary'] = 0.0

    ev['bg'] = CFG.W_BG if bg.contains(wt.center) else 0.0

    return min(sum(ev.values()), 1.0), ev


# ══════════════════════════════════════════════════════════
#  YOLO INFERENCE
# ══════════════════════════════════════════════════════════
def run_inference(model, frame):
    def _parse(results):
        dets = []
        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                lbl  = model.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(CFG.FRAME_W, x2), min(CFG.FRAME_H, y2)
                dets.append(((x1,y1,x2,y2), lbl, conf))
        return dets

    r1 = model(frame, conf=CFG.CONF, iou=CFG.IOU_THRESH,
               imgsz=CFG.IMG_SIZE, verbose=False)
    dets = _parse(r1)

    if CFG.DUAL_SCALE:
        r2 = model(frame, conf=CFG.CONF, iou=CFG.IOU_THRESH,
                   imgsz=CFG.IMG_SIZE_2, verbose=False)
        dets.extend(_parse(r2))
        dets = nms_merge(dets, iou_th=CFG.IOU_THRESH)

    return dets


# ══════════════════════════════════════════════════════════
#  TELEGRAM
# ══════════════════════════════════════════════════════════
def telegram(path, caption):
    if not CFG.TOKEN or not CFG.CHAT_ID:
        print("  [Telegram] skipped — SENTINEL_TG_TOKEN / SENTINEL_TG_CHAT_ID not set")
        return
    def _():
        try:
            with open(path, "rb") as f:
                requests.post(f"https://api.telegram.org/bot{CFG.TOKEN}/sendPhoto",
                    data={"chat_id": CFG.CHAT_ID, "caption": caption},
                    files={"photo": f}, timeout=8)
            print("  [Telegram] \u2713")
        except Exception as e:
            print(f"  [Telegram] \u2717 {e}")
    threading.Thread(target=_, daemon=True).start()


# ══════════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════════
C = {
    "person":(0,255,80), "waste":(0,80,255), "alert":(0,0,255),
    "warn":(0,165,255),  "ok":(0,220,90),    "info":(255,220,0),
    "white":(255,255,255),"dim":(140,140,140),"blk":(0,0,0),
    "vel":(180,60,255),
}

def draw_box(frm, box, col, lbl="", conf=None):
    x1, y1, x2, y2 = [int(v) for v in box]
    l = max(10, min(20, (x2-x1)//5, (y2-y1)//5))
    for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frm, (sx,sy), (sx+dx*l, sy), col, 2)
        cv2.line(frm, (sx,sy), (sx, sy+dy*l), col, 2)
    if lbl:
        txt = f"{lbl} {conf:.2f}" if conf else lbl
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(frm, (x1, y1-th-8), (x1+tw+6, y1), col, -1)
        cv2.putText(frm, txt, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C["blk"], 1)

def hud(frm, n_p, n_w, state, fps, pt_dict, wt_dict, sc, confirms):
    H, W = frm.shape[:2]
    bc = C["alert"] if "DUMP" in state else C["warn"] if "DEPART" in state or "RELEAS" in state else C["ok"]
    cv2.rectangle(frm, (0,0), (W,50), (8,8,16), -1)
    cv2.line(frm, (0,50), (W,50), bc, 1)
    cv2.putText(frm, f"SENTINEL {CFG.CAM_ID}", (10,33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, C["info"], 2)
    cv2.putText(frm, f"// {state}", (185,33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bc, 2)
    cv2.putText(frm, f"{fps:.0f} FPS", (W-85,20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C["dim"], 1)
    cv2.putText(frm, time.strftime("%H:%M:%S"), (W-85,40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["dim"], 1)

    cv2.rectangle(frm, (0,H-60), (W,H), (8,8,16), -1)
    cv2.line(frm, (0,H-60), (W,H-60), C["dim"] if sc==0 else bc, 1)
    cv2.putText(frm, f"PERSONS:{n_p}", (10,H-38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C["person"], 1)
    cv2.putText(frm, f"WASTE:{n_w}",   (10,H-18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C["waste"],  1)

    tag = "fine-tuned|OpenVINO" if CFG.USE_OPENVINO else "fine-tuned|PyTorch"
    cv2.putText(frm, tag, (W-165,H-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["dim"], 1)

    if sc > 0:
        bw=200; bx,by = 120,H-50
        cv2.rectangle(frm, (bx,by), (bx+bw,by+14), (35,35,35), -1)
        fc = int(bw * min(sc, 1.0))
        fc_col = C["alert"] if sc >= CFG.MIN_SCORE else C["warn"] if sc >= 0.4 else C["ok"]
        cv2.rectangle(frm, (bx,by), (bx+fc,by+14), fc_col, -1)
        cv2.rectangle(frm, (bx,by), (bx+bw,by+14), C["dim"], 1)
        cv2.putText(frm, f"EVIDENCE {sc:.2f}", (bx+3,by+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["blk"], 1)
        cv2.putText(frm, f"CONFIRM {confirms}/{CFG.CONFIRM_FRAMES}",
                    (bx+bw+6,by+11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["info"], 1)

    for pid, pt in pt_dict.items():
        if pt.state != PS.IDLE:
            h_half = int((pt.box[3]-pt.box[1])/2)
            cx, cy = pt.center
            cv2.putText(frm, f"P{pid}:{pt.state}", (cx-25, cy-h_half-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["info"], 1)


# ══════════════════════════════════════════════════════════
#  MODEL LOADING (with OpenVINO export/reuse for Intel CPUs)
# ══════════════════════════════════════════════════════════
def load_model():
    if not CFG.USE_OPENVINO:
        m = YOLO(CFG.MODEL)
        m.fuse()
        return m

    ov_dir = CFG.MODEL.replace(".pt", "_openvino_model")
    if not os.path.exists(ov_dir):
        print(f"[SENTINEL {CFG.CAM_ID}] No OpenVINO export found — exporting once (one-time cost)...")
        base = YOLO(CFG.MODEL)
        base.export(format="openvino", imgsz=CFG.IMG_SIZE)
        print(f"[SENTINEL {CFG.CAM_ID}] Exported to {ov_dir}/")
    return YOLO(ov_dir)


# ══════════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════════
os.makedirs(CFG.STATIC_DIR, exist_ok=True)
if not os.path.exists(CFG.LOG_FILE):
    with open(CFG.LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Time","Status","Image","Confidence","Evidence","Details"])

print(f"[SENTINEL {CFG.CAM_ID}] Loading fine-tuned model (garbage + person)...")
print(f"  \u2022 Camera ID: {CFG.CAM_ID}")
print(f"  \u2022 Source: {CFG.CAMERA_SOURCE}")
print(f"  \u2022 Weights: {CFG.MODEL}")
print(f"  \u2022 {'OpenVINO IR' if CFG.USE_OPENVINO else 'PyTorch'} runtime")
print(f"  \u2022 Single-scale inference @ {CFG.IMG_SIZE}px" + (" (dual-scale ON)" if CFG.DUAL_SCALE else ""))
print(f"  \u2022 Frame skip: every {CFG.FRAME_SKIP} frame(s)")
print(f"  \u2022 BG subtraction: every {CFG.BG_INTERVAL} processed frame(s)")

model = load_model()
_dummy = np.zeros((CFG.FRAME_H, CFG.FRAME_W, 3), dtype=np.uint8)
model(_dummy, imgsz=CFG.IMG_SIZE, verbose=False)
print(f"[SENTINEL {CFG.CAM_ID}] Model warmed up \u2713 — classes: {model.names}\n")

bg         = BG()
p_tracker  = Tracker(CFG.MAX_TRACK_DIST, CFG.MAX_TRACK_AGE)
w_tracker  = Tracker(CFG.MAX_TRACK_DIST, CFG.MAX_TRACK_AGE * 2)
p_tracks: Dict[int, PTrack] = {}
w_tracks: Dict[int, WTrack] = {}

print(f"[SENTINEL {CFG.CAM_ID}] Opening camera source {CFG.CAMERA_SOURCE}...")
grabber    = FrameGrabber(CFG.CAMERA_SOURCE)

frame_n    = 0
proc_n     = 0
last_alert = 0.0
confirms   = 0
state      = "MONITORING"
top_sc     = 0.0
fps_q      = deque(maxlen=30)
t_prev     = time.time()


# ══════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════
print(f"[SENTINEL {CFG.CAM_ID}] Running. Press Q to quit.\n")
while True:
    ok, raw = grabber.read()
    if not ok or raw is None:
        time.sleep(0.02); continue

    frame_n += 1
    if frame_n % CFG.FRAME_SKIP != 0:
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    proc_n += 1
    now = time.time()
    fps_q.append(1.0 / max(now - t_prev, 1e-4))
    t_prev = now
    fps = sum(fps_q) / len(fps_q)

    frame = cv2.resize(raw, (CFG.FRAME_W, CFG.FRAME_H))
    orig  = frame.copy()

    if proc_n % CFG.BG_INTERVAL == 0:
        bg.update(frame)

    detections = run_inference(model, frame)

    p_boxes: List[Tuple] = []
    w_boxes: List[Tuple] = []
    w_meta:  Dict        = {}

    for (box, lbl, conf) in detections:
        if lbl == "person":
            p_boxes.append(box)
        elif lbl in WASTE_LABELS and conf >= CFG.WASTE_CONF:
            w_boxes.append(box)
            w_meta[box] = (lbl, conf)

    if CFG.ENABLE_BG_FALLBACK:
        for blob in bg.blobs():
            area = (blob[2]-blob[0]) * (blob[3]-blob[1])
            if area < CFG.BG_MIN_AREA or area > CFG.BG_MAX_AREA:
                continue
            person_margin = [expand(pb, CFG.ARM_REACH_PX) for pb in p_boxes]
            if any(containment_ratio(blob, pb) > 0.4 for pb in person_margin):
                continue
            if any(iou(blob, wb) > 0.3 for wb in w_boxes):
                continue
            w_boxes.append(blob)
            w_meta[blob] = (CFG.BG_FALLBACK_LABEL, CFG.BG_FALLBACK_CONF)

    p_asgn = p_tracker.update(p_boxes)
    w_asgn = w_tracker.update(w_boxes)

    for tid, box in p_asgn.items():
        if tid not in p_tracks:
            p_tracks[tid] = PTrack(tid=tid, box=box)
        p_tracks[tid].upd(box)
    for tid in list(p_tracks):
        if tid not in p_asgn and now - p_tracks[tid].last_seen > CFG.MAX_TRACK_AGE:
            del p_tracks[tid]

    for tid, box in w_asgn.items():
        lbl, conf = w_meta.get(box, ("unknown", 0.5))
        if tid not in w_tracks:
            w_tracks[tid] = WTrack(tid=tid, box=box, label=lbl, conf=conf)
        else:
            w_tracks[tid].upd(box, conf)
    for tid in list(w_tracks):
        if tid not in w_asgn and now - w_tracks[tid].last_seen > CFG.MAX_TRACK_AGE * 2:
            del w_tracks[tid]

    for tid, pt in p_tracks.items():
        draw_box(frame, pt.box, C["person"], f"P{tid}")
    for tid, wt in w_tracks.items():
        if wt.is_background:
            col = C["dim"]
        elif wt.velocity() > CFG.MAX_WASTE_VEL:
            col = C["vel"]
        else:
            col = C["waste"]
        draw_box(frame, wt.box, col, wt.label, wt.conf)

    # ── NEW (FIX 1): determine, for THIS frame, which waste tracks are
    # actually near ANY person right now — using the SAME HOLD_DIST_PX /
    # ARM_REACH_PX values as the rest of the file, unchanged. Used to
    # decay stale appeared_near_person flags below. ──
    near_any_person_now = set()
    for pid, pt in p_tracks.items():
        for wid, wt in w_tracks.items():
            if wt.is_background:
                continue
            overlap = intersect(pt.box, wt.box)
            arm_reach_near = inside(wt.center, expand(pt.box, CFG.ARM_REACH_PX))
            cd = dist(pt.center, wt.center)
            if overlap > 0 or cd < CFG.HOLD_DIST_PX or arm_reach_near:
                near_any_person_now.add(wid)

    for wid, wt in w_tracks.items():
        if wid in near_any_person_now:
            wt.frames_since_near = 0
        elif wt.appeared_near_person:
            wt.frames_since_near += 1
            if wt.frames_since_near > CFG.NEAR_DECAY_FRAMES:
                # Stale flag — hasn't actually been near anyone in a
                # while, so stop treating it as "recently interacted
                # with." A fresh HOLDING will re-flag it the moment a
                # person is genuinely close again.
                wt.appeared_near_person = False
                wt.assoc_pid = None

    top_sc   = 0.0
    top_ev   = {}
    top_pair = (None, None)

    for pid, pt in p_tracks.items():
        cw, cd = None, float('inf')
        for wid, wt in w_tracks.items():
            if wt.is_background:
                continue
            if wt.velocity() > CFG.MAX_WASTE_VEL:
                continue
            d = dist(pt.center, wt.center)
            if d < cd:
                cd, cw = d, wt

        if cw is None:
            continue

        overlap = intersect(pt.box, cw.box)
        arm_reach_near = inside(cw.center, expand(pt.box, CFG.ARM_REACH_PX))
        near = overlap > 0 or cd < CFG.HOLD_DIST_PX or arm_reach_near

        if near and not cw.appeared_near_person:
            cw.appeared_near_person = True
            cw.assoc_pid = pid
            cw.frames_since_near = 0
            print(f"  [{CFG.CAM_ID} W{cw.tid}] '{cw.label}' first appeared near P{pid} — tracking")

        if pt.state == PS.IDLE:
            if near and cw.appeared_near_person and cw.is_new_enough:
                pt.state    = PS.HOLDING
                pt.held_wid = cw.tid
                cw.assoc_pid = pid
                pt.pending_away_frames = 0
                print(f"  [{CFG.CAM_ID} P{pid}] IDLE \u2192 HOLDING '{cw.label}' (dist={cd:.0f}px)")

        elif pt.state == PS.HOLDING:
            if not near:
                pt.pending_away_frames += 1
                if pt.pending_away_frames >= CFG.REQUIRE_AWAY_FRAMES:
                    pt.state     = PS.RELEASED
                    pt.rel_point = cw.center
                    pt.rel_time  = now
                    pt.fsr       = 0
                    pt.pending_away_frames = 0
                    print(f"  [{CFG.CAM_ID} P{pid}] HOLDING \u2192 RELEASED at {pt.rel_point}")
            else:
                pt.pending_away_frames = max(0, pt.pending_away_frames - 1)
                pt.held_wid  = cw.tid
                cw.assoc_pid = pid

        elif pt.state == PS.RELEASED:
            pt.fsr += 1
            if pt.rel_point:
                d_drop = dist(pt.center, pt.rel_point)
                away   = pt.moving_away(pt.rel_point)
                cv2.arrowedLine(frame, pt.rel_point, pt.center, C["warn"], 1, tipLength=0.15)
                cv2.circle(frame, pt.rel_point, 10, C["alert"], 2)
                elapsed_ok = (now - pt.rel_time) >= CFG.MIN_ELAPSED_SEC
                if elapsed_ok and (d_drop >= CFG.DEPARTURE_DIST_PX or (away and pt.fsr >= 5)):
                    pt.state = PS.DEPARTED
                    print(f"  [{CFG.CAM_ID} P{pid}] RELEASED \u2192 DEPARTED (d={d_drop:.0f}px)")

        elif pt.state == PS.DEPARTED:
            pt.fsr += 1
            if pt.fsr > CFG.MAX_DEPARTED_FRAMES:
                # NEW (FIX 2): stuck here too long without ever firing a
                # real alert — reset so this person doesn't keep getting
                # re-paired with whatever waste happens to be nearest,
                # indefinitely.
                pt.state = PS.IDLE
                pt.held_wid = None
                pt.rel_point = None
                pt.rel_time = None
                pt.fsr = 0
                pt.pending_away_frames = 0

        if pt.state in (PS.RELEASED, PS.DEPARTED, PS.CONFIRMED):
            sc, ev = score(pt, cw, bg)
            pt.score = sc
            if sc > top_sc:
                top_sc, top_ev, top_pair = sc, ev, (pid, cw.tid)

    if top_sc >= CFG.MIN_SCORE:
        confirms = min(confirms + 1, CFG.CONFIRM_FRAMES)
    else:
        confirms = max(0, confirms - 1)

    if top_sc >= CFG.MIN_SCORE and confirms >= CFG.CONFIRM_FRAMES:
        state = "DUMPING DETECTED"
    elif any(p.state == PS.DEPARTED for p in p_tracks.values()):
        state = "DEPARTED — SCORING"
    elif any(p.state == PS.RELEASED for p in p_tracks.values()):
        state = "RELEASED — TRACKING"
    elif any(p.state == PS.HOLDING  for p in p_tracks.values()):
        state = "HOLDING"
    else:
        state = "MONITORING"

    if (state == "DUMPING DETECTED"
            and top_pair[0] is not None
            and top_pair[1] is not None
            and now - last_alert > CFG.COOLDOWN_SEC):

        pid, wid = top_pair
        pt = p_tracks.get(pid)
        wt = w_tracks.get(wid)

        if pt and wt and not pt.sent:
            ts    = time.strftime("%Y-%m-%d %H:%M:%S")
            fname = f"dump_{CFG.CAM_ID}_{int(now)}.jpg"
            fpath = os.path.join(CFG.STATIC_DIR, fname)

            alert_frame = orig.copy()
            cv2.rectangle(alert_frame, (0,0), (CFG.FRAME_W,CFG.FRAME_H), C["alert"], 4)
            cv2.putText(alert_frame, "ILLEGAL DUMPING",
                        (CFG.FRAME_W//2-155, 75), cv2.FONT_HERSHEY_DUPLEX, 1.1, C["alert"], 3)
            draw_box(alert_frame, pt.box, C["alert"], f"SUSPECT P{pid}")
            draw_box(alert_frame, wt.box, C["waste"], wt.label, wt.conf)
            cv2.imwrite(fpath, alert_frame)

            ev_str  = ", ".join(f"{k}={v:.2f}" for k, v in top_ev.items())
            details = f"Camera:{CFG.CAM_ID} | Waste:{wt.label} | State:{pt.state} | Drop:{pt.rel_point} | Dwell:{wt.dwell_frames}f"
            with open(CFG.LOG_FILE, "a", newline="") as f:
                csv.writer(f).writerow([ts, "Dumping Detected", fname,
                                        f"{top_sc:.2f}", ev_str, details])

            caption = (f"\U0001F6A8 Illegal Dumping!\n"
                       f"Camera: {CFG.CAM_ID}\n"
                       f"Waste: {wt.label}\n"
                       f"Confidence: {top_sc:.0%}\n"
                       f"Evidence: {ev_str}\n"
                       f"Dwell: {wt.dwell_frames} frames\n"
                       f"Time: {ts}")
            telegram(fpath, caption)

            print(f"\n\U0001F6A8 ALERT [{CFG.CAM_ID}] [{ts}] Waste:{wt.label} Score:{top_sc:.2f}")
            print(f"   {ev_str}\n")

            # NEW (FIX 2): fully reset this person's state after
            # alerting — not just 'sent', but the whole HOLDING/
            # RELEASED/DEPARTED progress. Without this, they stay stuck
            # in DEPARTED and keep getting re-scored against whatever
            # waste is nearest each frame, long after the real event
            # ended. sent resets to False too, so a genuine second
            # incident from the same person later can still alert —
            # COOLDOWN_SEC already prevents spammy re-firing short-term.
            pt.state    = PS.IDLE
            pt.held_wid = None
            pt.rel_point = None
            pt.rel_time = None
            pt.fsr      = 0
            pt.pending_away_frames = 0
            pt.sent     = False
            last_alert  = now
            confirms    = 0

    hud(frame, len(p_boxes), len(w_boxes), state, fps,
        p_tracks, w_tracks, top_sc, confirms)

    # ── Write this camera's live frame + status for the dashboard.
    # Filenames include CAM_ID so multiple cameras never overwrite
    # each other's files — app.py reads static/live_status_<CAM_ID>.json
    # for each configured camera independently. ──
    if proc_n % 15 == 0:
        cv2.imwrite(os.path.join(CFG.STATIC_DIR, f"live_frame_{CFG.CAM_ID}.jpg"), frame)
        status = {
            "camera_id": CFG.CAM_ID,
            "persons": len(p_boxes),
            "waste": len(w_boxes),
            "state": state,
            "top_score": round(top_sc, 2),
            "timestamp": time.time(),
        }
        with open(os.path.join(CFG.STATIC_DIR, f"live_status_{CFG.CAM_ID}.json"), "w") as f:
            json.dump(status, f)

    cv2.imshow(f"SENTINEL {CFG.CAM_ID} — Fine-Tuned Model", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

grabber.release()
cv2.destroyAllWindows()
print(f"[SENTINEL {CFG.CAM_ID}] Shutdown complete.")