"""
pick_v3.py — SAM 3 finds the object you ask for, the arm grabs it.

It replaces the human click in your friend's "click to grab" page (arm/click.py):
    his page:   human clicks a pixel          -> pixel -> table x,y -> arm grabs
    this file:  SAM 3 + checker pick a pixel  -> pixel -> table x,y -> arm grabs
Nothing in his code changes. This script talks to his page over HTTP:
    GET  /status      calibration marks, busy flag
    GET  /frame.jpg   the camera image (the SAME image his calibration was made on)
    POST /api/grab    {u, v, height, above}  -> his Reacher.grab() does the motion

Steps
  1. Get the camera frame from his page (or --image for an offline test).
  2. target_selector_v5: SAM 3 (+ checker) -> GO / NOT_FOUND / AMBIGUOUS.
  3. Choose the pixel where the arm should go:
       contact point  = where the object touches the table (bottom of its mask).
                        Only table-level pixels map to correct table coordinates.
       target point   = contact point pushed away from the camera by half the
                        object's width (from items_v1.json), i.e. the middle of its base.
  4. Pixel -> table x, y in cm, with the same calibration marks his page uses.
  5. Dry run by default: prints x, y and saves pick.jpg.  With --go: the arm grabs.

Run (his page must be running:  python -m arm.click --camera ...  ->  http://localhost:8890)
  python pick_v3.py --target "energy drink can" --category can --vlm ollama          # dry run
  python pick_v3.py --target "energy drink can" --category can --vlm ollama --go     # grab

How accurate is the calibration? (no SAM 3, no motion)
  python pick_v3.py --check
  python pick_v3.py --check --calib /path/to/LosPollosForTheElders/calib/camera.json

Grab, then put it down at a fixed table point (needs the 'place' action, see place_snippet_v1.py):
  python pick_v3.py --target "blue pill box" --category box --vlm ollama --go --place -10 25

Offline test, no arm (uses his calibration file and a saved frame from the same camera):
  python pick_v3.py --image frame.jpg --calib /path/to/LosPollosForTheElders/calib/camera.json \
                    --target "energy drink can" --category can --vlm none

Needs in the same folder: target_selector_v5.py, grasp_planner_v3.py, items_v1.json
"""
import argparse, io, json, math, subprocess, sys
import numpy as np
import cv2
import requests
from PIL import Image

import target_selector_v5 as ts
import grasp_planner_v3 as gp

DEFAULT_PUSH_CM = 2.5      # contact point -> middle of the base, when the item's width is unknown
DEFAULT_CLOSE_CM = 4.0     # his defaults: close 4 cm above the table, approach from 12 cm
DEFAULT_ABOVE_CM = 12.0
CLEARANCE_CM = 5.0         # approach this far above the object's top


# ---------- calibration: pixel <-> table (same maths as arm/click.py) ----------

def _fit_h(pts, xy):
    """Least-squares homography: normalised pixel (u, v) -> table (x, y) cm."""
    rows, rhs = [], []
    for (u, v), (x, y) in zip(pts, xy):
        rows.append([u, v, 1, 0, 0, 0, -x * u, -x * v]); rhs.append(x)
        rows.append([0, 0, 0, u, v, 1, -y * u, -y * v]); rhs.append(y)
    h = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return np.append(h, 1.0).reshape(3, 3)


def _apply_h(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return np.array([p[0] / p[2], p[1] / p[2]])


class Calib:
    """pairs = [u, v, x_cm, y_cm] marks; size = width of the image they were clicked on.

    Two models are fitted from the same marks:
      plain  : pixel -> table with a homography only. This is what arm/click.py uses.
      lens   : first straighten the wide-angle lens (one curvature number k, fitted from the
               marks), then the homography. OFF by default: in simulation it did not help when
               the pick area is in the middle of the image; evenly spread, careful marks did.
               Try --lens only if 8+ careful marks still leave errors above 1 cm.
    We compute x, y with the better model, then hand his page the pixel that HIS plain
    model maps to exactly that x, y (server_pixel), so his code needs no change.
    """

    MIN_MARKS_FOR_LENS = 8

    def __init__(self, pairs, size, aspect=9 / 16, lens=False):
        if len(pairs) < 4:
            raise RuntimeError("Camera not calibrated: needs at least 4 marks (do it on his page).")
        self.pairs, self.size = [list(p) for p in pairs], float(size)
        self.c = np.array([0.5, 0.5 * aspect])                       # image centre, normalised
        self.px = [np.array([u, v]) / self.size for u, v, _, _ in self.pairs]
        self.xy = [np.array([x, y]) for _, _, x, y in self.pairs]
        self.H_plain = _fit_h(self.px, self.xy)
        self.Hinv_plain = np.linalg.inv(self.H_plain)
        self.k = 0.0
        self.loo_plain = self._loo(0.0)
        self.lens = lens
        if lens and len(self.pairs) >= self.MIN_MARKS_FOR_LENS:
            ks = np.linspace(-1.5, 1.5, 151)
            errs = [self._rms(self._loo(k)) for k in ks]
            best = int(np.argmin(errs))
            if errs[best] < 0.8 * self._rms(self.loo_plain):          # only if clearly better
                self.k = float(ks[best])
        self.H = _fit_h([self._undist(p, self.k) for p in self.px], self.xy)
        self.Hinv = np.linalg.inv(self.H)
        self.loo = self._loo(self.k)

    @staticmethod
    def _rms(e):
        return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")

    def _undist(self, p, k):
        d = p - self.c
        return self.c + d / (1.0 + k * float(d @ d))

    def _dist(self, q, k):
        """Inverse of _undist (a few fixed-point steps)."""
        p = q.copy()
        for _ in range(15):
            d = p - self.c
            p = self.c + (q - self.c) * (1.0 + k * float(d @ d))
        return p

    def _loo(self, k):
        """Predict each mark from the OTHER marks: honest accuracy. Needs 6+ marks."""
        n = len(self.pairs)
        if n < 6:
            return []
        und = [self._undist(p, k) for p in self.px]
        out = []
        for i in range(n):
            H = _fit_h(und[:i] + und[i + 1:], self.xy[:i] + self.xy[i + 1:])
            out.append(float(np.linalg.norm(_apply_h(H, *und[i]) - self.xy[i])))
        return out

    def to_table(self, u, v):
        q = self._undist(np.array([u, v]) / self.size, self.k)
        x, y = _apply_h(self.H, *q)
        return float(x), float(y)

    def to_pixel(self, x, y):
        """Table point -> pixel in the real (curved) image. For drawing."""
        q = _apply_h(self.Hinv, x, y)
        p = self._dist(q, self.k) * self.size
        return float(p[0]), float(p[1])

    def server_pixel(self, x, y):
        """The pixel his page's plain model maps to exactly (x, y)."""
        p = _apply_h(self.Hinv_plain, x, y) * self.size
        return float(p[0]), float(p[1])

    def fit_errors(self):
        return [round(float(np.linalg.norm(np.array(self.to_table(u, v)) - [x, y])), 1)
                for u, v, x, y in self.pairs]

    def expected_error(self):
        """(cm, how it was measured)"""
        if self.loo:
            return max(self.loo), "predicting each mark from the others"
        return max(self.fit_errors()), "on the marks themselves; too few marks to measure it properly"

    def inside_marks(self, u, v):
        hull = cv2.convexHull(np.array([[p[0], p[1]] for p in self.pairs], np.float32))
        return cv2.pointPolygonTest(hull, (float(u), float(v)), False) >= 0

    def report(self):
        acc, how = self.expected_error()
        lines = [f"Calibration: {len(self.pairs)} marks. Worst error {acc:.1f} cm ({how})."]
        if self.k:
            lines.append(f"  Lens correction ON (k={self.k:+.2f}): typical error "
                         f"{self._rms(self.loo_plain):.1f} cm -> {self._rms(self.loo):.1f} cm.")
        elif self.lens and len(self.pairs) < self.MIN_MARKS_FOR_LENS:
            lines.append(f"  Lens correction needs {self.MIN_MARKS_FOR_LENS}+ marks; using the plain model.")
        elif self.lens:
            lines.append("  Lens correction did not help; using the plain model.")
        fit = self.fit_errors()
        for i, (u, v, x, y) in enumerate(self.pairs):
            loo = f"   unseen: {self.loo[i]:.1f} cm" if self.loo else ""
            lines.append(f"  mark {i + 1}: x={x:6.1f} y={y:6.1f}   fit: {fit[i]:.1f} cm{loo}")
        if acc > 1.0:
            lines.append("  -> too loose for grasping (aim for under 1 cm). Redo the worst marks carefully and "
                         "spread 8+ marks evenly over the pick area (corners, edges, middle).")
        return "\n".join(lines)


# ---------- which pixel should the arm go to ----------

def contact_pixel(mask):
    """Where the object touches the table: lowest mask pixel under the mask's middle column."""
    ys, xs = np.nonzero(mask)
    u_mid = int(round((xs.min() + xs.max()) / 2))
    col = np.abs(xs - u_mid) <= 2
    if not col.any():
        col = np.ones_like(xs, bool)
    return float(u_mid), float(ys[col].max())


def choose_target(mask, calib, scale, push_cm, mode):
    """Returns dict with contact/target in frame pixels and table cm.
    scale = calibration image width / frame width (1.0 when they are the same image)."""
    if mode == "center":
        ys, xs = np.nonzero(mask)
        u, v = float(xs.mean()), float(ys.mean())
        x, y = calib.to_table(u * scale, v * scale)
        return {"mode": "mask center", "contact_px": None, "target_px": [u, v], "x_cm": x, "y_cm": y}
    cu, cv_ = contact_pixel(mask)
    c = np.array(calib.to_table(cu * scale, cv_ * scale))
    # a table point a few pixels further "up" the image = further from the camera
    f = np.array(calib.to_table(cu * scale, (cv_ - 5) * scale))
    away = (f - c) / (np.linalg.norm(f - c) + 1e-9)
    t = c + away * push_cm
    tu, tv = calib.to_pixel(*t)
    return {"mode": f"table contact + {push_cm:.1f} cm away from the camera",
            "contact_px": [cu, cv_], "contact_cm": [float(c[0]), float(c[1])],
            "target_px": [tu / scale, tv / scale], "x_cm": float(t[0]), "y_cm": float(t[1])}


def grab_heights(profile, close_cm, above_cm):
    h = profile.get("height_cm")
    if close_cm is None:
        close_cm = max(1.0, h / 2) if h else DEFAULT_CLOSE_CM
    if above_cm is None:
        above_cm = max(DEFAULT_ABOVE_CM, h + CLEARANCE_CM) if h else DEFAULT_ABOVE_CM
    return round(close_cm, 1), round(above_cm, 1)


# ---------- drawing ----------

def show(path):
    """Open the result picture in the default viewer (Preview on a Mac)."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            Image.open(path).show()
    except Exception as e:
        print(f"(couldn't open {path}: {e})")


def annotate(path, t, status_line):
    vis = cv2.imread(path)
    if t.get("contact_px"):
        c = tuple(int(round(v)) for v in t["contact_px"])
        cv2.circle(vis, c, 7, (255, 255, 255), -1); cv2.circle(vis, c, 7, (0, 0, 0), 2)
    p = tuple(int(round(v)) for v in t["target_px"])
    cv2.drawMarker(vis, p, (0, 0, 255), cv2.MARKER_CROSS, 34, 3)
    cv2.circle(vis, p, 14, (0, 0, 255), 2)
    text = f"arm target  x={t['x_cm']:.1f}  y={t['y_cm']:.1f} cm"
    for col, th in (((255, 255, 255), 4), ((0, 0, 255), 2)):
        cv2.putText(vis, text, (p[0] + 20, p[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th)
        cv2.putText(vis, status_line, (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255) if th == 4 else (0, 0, 0), th)
    cv2.imwrite(path, vis)


# ---------- main ----------

def main(argv=None):
    ap = argparse.ArgumentParser(description="SAM 3 picks the object, the arm grabs it")
    ap.add_argument("--target")
    ap.add_argument("--category")
    ap.add_argument("--vlm", choices=["none", "ollama", "claude"], default="none")
    ap.add_argument("--arm-url", default="http://localhost:8890", help="your friend's arm.click page")
    ap.add_argument("--image", help="offline test: use this saved frame instead of the live camera")
    ap.add_argument("--calib", help="offline test: path to his calib/camera.json")
    ap.add_argument("--items", default="items_v1.json")
    ap.add_argument("--point", choices=["auto", "contact", "center"], default="auto",
                    help="auto: table contact + half the item's width; center: middle of the mask")
    ap.add_argument("--height", type=float, help="fingertip height when closing, cm (default: from item)")
    ap.add_argument("--above", type=float, help="approach / lift height, cm (default: from item)")
    ap.add_argument("--go", action="store_true", help="really move the arm (default: dry run)")
    ap.add_argument("--check", action="store_true", help="only print how accurate the calibration is")
    ap.add_argument("--lens", action="store_true", help="try lens-curve correction (needs 8+ marks)")
    ap.add_argument("--place", nargs=2, type=float, metavar=("X", "Y"),
                    help="after a successful grab, put the object down at this table point (cm)")
    ap.add_argument("--place-height", type=float, help="fingertip height when releasing, cm (default: close height)")
    ap.add_argument("-y", "--yes", action="store_true", help="with --go: don't ask before moving")
    ap.add_argument("--out", default="pick.jpg")
    ap.add_argument("--no-show", action="store_true", help="don't open the result picture afterwards")
    a = ap.parse_args(argv)
    if not a.target and not a.check:
        ap.error("--target is required (or use --check)")
    ts.VLM_BACKEND = a.vlm
    live = a.calib is None
    if not live and not a.image and not a.check:
        sys.exit("Offline test needs --image as well as --calib.")
    if a.go and not live:
        sys.exit("--go needs the live camera (don't combine it with --image).")

    # 1. frame + calibration
    if live:
        try:
            st = requests.get(a.arm_url + "/status", timeout=5).json()
        except requests.RequestException as e:
            sys.exit(f"Can't reach the arm page at {a.arm_url} ({e}). Is `python -m arm.click` running?")
        if not st.get("calibrated"):
            sys.exit("The camera isn't calibrated yet. Do the 6+ marks on the arm page first.")
        if st.get("pending"):
            sys.exit("The arm page is waiting for a calibration click. Finish or undo it first.")
        r = requests.get(a.arm_url + "/frame.jpg", timeout=5)
        if r.status_code != 200:
            sys.exit("The arm page has no camera image right now.")
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        calib = Calib(st["pairs"], st["size"], img.height / img.width, a.lens)
    else:
        d = json.load(open(a.calib))
        img = Image.open(a.image).convert("RGB") if a.image else None
        calib = Calib(d["pairs"], d.get("size", 224), img.height / img.width if img else 9 / 16, a.lens)
    print(calib.report())
    if a.check:
        return "CHECK"
    scale = calib.size / img.width
    if abs(scale - 1) > 0.01:
        print(f"Note: frame is {img.width}px wide, calibration was made at {calib.size:.0f}px; rescaling.")

    # 2. which object
    res = ts.decide(img, a.target, a.category)
    ts.draw_sam3_style(img, res, a.out, show_all=True)
    if res["status"] != "GO":
        print(json.dumps({"status": res["status"], "target": a.target}, indent=2))
        print(f"saved {a.out}")
        if not a.no_show:
            show(a.out)
        return res["status"]
    cand = res["chosen"]

    # 3. which pixel, 4. table x, y
    profile = gp.load_profile(a.items, a.target)
    posture, how = gp.detect_posture(ts, img, cand, a.vlm)
    width = profile.get("width_cm")
    mode = a.point
    if mode == "auto":
        mode = "contact" if (width or posture == "upright") else "center"
    push = (width / 2 if width else DEFAULT_PUSH_CM)
    t = choose_target(cand["mask"], calib, scale, push, mode)
    close_cm, above_cm = grab_heights(profile, a.height, a.above)
    tu, tv = calib.server_pixel(t["x_cm"], t["y_cm"])     # the pixel his page maps to this x, y
    out = {"status": "GO", "target": a.target, "x_cm": round(t["x_cm"], 1), "y_cm": round(t["y_cm"], 1),
           "point": t["mode"], "posture": posture, "close_height_cm": close_cm, "approach_height_cm": above_cm,
           "item_profile": profile or f"none (add it to {a.items})"}
    if not calib.inside_marks(t["target_px"][0] * scale, t["target_px"][1] * scale):
        out["warning"] = "target is outside the area covered by the calibration marks: position is a guess"
    annotate(a.out, t, f"{a.target}: close at {close_cm} cm, approach from {above_cm} cm")
    print(json.dumps(out, indent=2))
    print(f"saved {a.out}")
    if not a.no_show:
        show(a.out)

    # 5. grab
    if not a.go:
        print(f"Dry run: the arm did not move. Check {a.out}, then add --go.")
        return "DRY_RUN"
    if st.get("busy"):
        sys.exit("The arm is already moving.")
    if not a.yes:
        input("Follower on (Self-controlled)? Table clear? Press Enter to grab, Ctrl+C to cancel... ")
    r = requests.post(a.arm_url + "/api/grab", timeout=120,
                      json={"u": tu, "v": tv, "height": close_cm, "above": above_cm}).json()
    if not r.get("ok"):
        print("Arm refused:", r.get("error"))
        return "ARM_ERROR"
    log = requests.get(a.arm_url + "/status", timeout=5).json().get("log", "")
    print("Arm:", log)
    if "holding it" not in log:
        return "EMPTY"
    if a.place:
        px, py = a.place
        r = requests.post(a.arm_url + "/api/place", timeout=120,
                          json={"x": px, "y": py, "above": above_cm,
                                "height": a.place_height or close_cm}).json()
        if not r.get("ok"):
            print("Could not place:", r.get("error"))
            if "Unknown action" in str(r.get("error")):
                print("  -> his page has no 'place' action yet. Add the snippet from place_snippet_v1.py to arm/click.py.")
            return "HOLDING"
        print(f"Placed at x={px:.1f}, y={py:.1f} cm.")
        return "PLACED"
    return "HOLDING"


if __name__ == "__main__":
    main()
