"""
target_selector.py — decide "is this the right object?" before the arm moves.

Pipeline
  1. SAM 3 proposes candidates from two prompts: the specific target ("blue pill box")
     and a broader category ("box"), so a slightly-off phrase doesn't make us miss it.
  2. Each candidate crop is checked by a VLM: "Is this <target>? yes/no + confidence".
  3. Decision rules:
       exactly 1 confident match          -> GO
       0 matches                          -> NOT_FOUND   (Reachy: "I can't find it")
       2+ matches                         -> AMBIGUOUS   (Reachy asks / use position hint)
  4. Consistency: run on N frames, only GO if every frame picks the same object.

Usage
  python target_selector.py --image table.jpg --target "blue pill box" --category "box"
  python target_selector.py --camera 0 --target "blue pill box" --category "box" --frames 3

Needs: pip install torch torchvision transformers accelerate anthropic opencv-python numpy pillow
       (on a Mac, the standard torch wheel includes Apple GPU / MPS support)
       hf auth login   (after accepting the conditions on huggingface.co/facebook/sam3)
       export ANTHROPIC_API_KEY=...
"""
import os
# On Apple Silicon, let any op that MPS doesn't support fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, base64, io, json, sys
import numpy as np
import cv2
from PIL import Image

# ---------------- config ----------------
SAM_SCORE_MIN = 0.35      # keep weak SAM candidates; the VLM does the real filtering
VLM_CONF_MIN = 0.80       # VLM must be this sure to count as a match
IOU_DUP = 0.6             # merge duplicate candidates from the two prompts
CROP_PAD = 0.25           # context around each crop (fraction of box size)
VLM_MODEL = "claude-sonnet-5-5"

# ---------------- SAM 3 ----------------
_processor = _model = None
_device = None
def sam3_candidates(img: Image.Image, prompts):
    # Uses the Hugging Face Transformers integration (model card: huggingface.co/facebook/sam3)
    global _processor, _model, _device
    import torch
    if _model is None:
        from transformers import Sam3Processor, Sam3Model
        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():   # Apple Silicon (M1/M2/M3)
            _device = "mps"
        else:
            _device = "cpu"
        print(f"SAM 3 running on: {_device}")
        _model = Sam3Model.from_pretrained("facebook/sam3").to(_device).eval()
        _processor = Sam3Processor.from_pretrained("facebook/sam3")
    img = img.convert("RGB")
    cands = []
    for p in prompts:
        inputs = _processor(images=img, text=p, return_tensors="pt").to(_device)
        with torch.no_grad():
            outputs = _model(**inputs)
        out = _processor.post_process_instance_segmentation(
            outputs, threshold=SAM_SCORE_MIN, mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist())[0]
        for m, b, s in zip(out["masks"], out["boxes"], out["scores"]):
            s = float(s)
            if s < SAM_SCORE_MIN:
                continue
            m = np.asarray(m.cpu() if hasattr(m, "cpu") else m).squeeze().astype(bool)
            b = [float(v) for v in (b.cpu() if hasattr(b, "cpu") else b)]
            cands.append({"mask": m, "box": b, "sam_score": s, "prompt": p})
    return dedupe(cands)

def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)

def dedupe(cands):
    cands = sorted(cands, key=lambda c: -c["sam_score"])
    kept = []
    for c in cands:
        if all(box_iou(c["box"], k["box"]) < IOU_DUP for k in kept):
            kept.append(c)
    return kept

# ---------------- VLM check ----------------
_client = None
def crop(img: Image.Image, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - CROP_PAD * w); y1 = max(0, y1 - CROP_PAD * h)
    x2 = min(img.width, x2 + CROP_PAD * w); y2 = min(img.height, y2 + CROP_PAD * h)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))

def to_b64(im: Image.Image):
    buf = io.BytesIO(); im.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def vlm_check(crop_img: Image.Image, target: str):
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    prompt = (
        f'A robot must pick up exactly: "{target}".\n'
        "Is the main object in this image that exact item? Be strict: similar-looking "
        "but different items (other color, other label, other box) are NOT a match.\n"
        'Reply with JSON only: {"match": true/false, "confidence": 0.0-1.0, "reason": "short"}'
    )
    msg = _client.messages.create(
        model=VLM_MODEL, max_tokens=200,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": to_b64(crop_img)}},
            {"type": "text", "text": prompt}]}])
    text = msg.content[0].text.strip()
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return {"match": False, "confidence": 0.0, "reason": f"unparseable: {text[:80]}"}

# ---------------- decision ----------------
def decide(img: Image.Image, target: str, category: str | None):
    prompts = [target] + ([category] if category else [])
    cands = sam3_candidates(img, prompts)
    for c in cands:
        v = vlm_check(crop(img, c["box"]), target)
        c["vlm"] = v
        c["is_match"] = bool(v.get("match")) and float(v.get("confidence", 0)) >= VLM_CONF_MIN
    matches = [c for c in cands if c["is_match"]]
    if len(matches) == 1:
        status = "GO"
    elif not matches:
        status = "NOT_FOUND"
    else:
        status = "AMBIGUOUS"
    return {"status": status, "candidates": cands, "chosen": matches[0] if status == "GO" else None}

def mask_center(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())

def consistent(results):
    """GO only if every frame said GO and the chosen object stayed in the same place."""
    if not all(r["status"] == "GO" for r in results):
        return False
    centers = [mask_center(r["chosen"]["mask"]) for r in results]
    c0 = np.array(centers[0])
    return all(np.linalg.norm(np.array(c) - c0) < 20 for c in centers)  # pixels

# ---------------- output ----------------
def draw(img: Image.Image, res, path):
    vis = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    for i, c in enumerate(res["candidates"]):
        col = (0, 200, 0) if c["is_match"] else (0, 0, 220)
        x1, y1, x2, y2 = map(int, c["box"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        label = f'#{i} vlm={c["vlm"].get("confidence", 0):.2f} sam={c["sam_score"]:.2f}'
        cv2.putText(vis, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.putText(vis, res["status"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(path, vis)

def summary(res):
    out = {"status": res["status"], "candidates": [
        {"i": i, "box": [round(v) for v in c["box"]], "prompt": c["prompt"],
         "sam_score": round(c["sam_score"], 2), "vlm": c["vlm"]}
        for i, c in enumerate(res["candidates"])]}
    if res["chosen"] is not None:
        out["target_pixel"] = [round(v) for v in mask_center(res["chosen"]["mask"])]
    return out

def grab_frames(cam, n):
    cap = cv2.VideoCapture(cam)
    for _ in range(5): cap.read()  # let exposure settle
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok: sys.exit("camera read failed")
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--camera", type=int)
    ap.add_argument("--target", required=True); ap.add_argument("--category")
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--out", default="decision.jpg")
    a = ap.parse_args()

    frames = [Image.open(a.image)] * 1 if a.image else grab_frames(a.camera, a.frames)
    results = [decide(f, a.target, a.category) for f in frames]
    final = results[-1]
    if len(results) > 1 and final["status"] == "GO" and not consistent(results):
        final = {**final, "status": "AMBIGUOUS", "chosen": None}
    draw(frames[-1], final, a.out)
    print(json.dumps(summary(final), indent=2))
