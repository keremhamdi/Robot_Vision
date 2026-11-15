"""
arm_service.py — the "arm PC" side of Medicine Buddy's medicine_buddy.arm.v1 contract.

Reachy Mini (LosPollosForTheElders) asks, after the person says yes:
    POST /v1/dispense     {"medicine": {"name": "Aspirin", ...}, "dry_run": false, ...}   medicine -> bowl
    POST /v1/fetch        {"object": "water bottle", "dry_run": false, ...}               object -> hand-over spot
    POST /v1/check_taken  {...}
This service answers with the JSON the Reachy side expects (dispense_result / check_taken_result)
and does the real work with pick_v4 (SAM 3 finds the box -> table x,y -> arm grabs -> put in the bowl).

On the Reachy side only one setting changes:
    MEDICINE_BUDDY_ARM_URL=http://<this-mac>:9090      (and the app's dry_run must be off)

Run
    python arm_service.py --bowl -10 25 --handover 0 30 --vlm ollama --arm-url http://localhost:8890
    python arm_service.py --fake            # no camera, no arm, no models: for testing the wiring

    curl localhost:9090/health

Fetch: "object" is whatever the person said ("the bottle please"); it is matched against items_v1.json
with the same closed-vocabulary matching as voice_pick.py (unknown or ambiguous -> failed, never a guess).
--handover X Y is where the object is put down for the person (defaults to --bowl).

Medicine name -> object: items_v1.json. An entry matches when the medicine name equals its key or one
of its "aliases" (case-insensitive); its "category" is passed to the detector. Unknown medicine -> the
request fails with a clear error (we never guess which box to grab).

check_taken is NOT implemented: it always answers taken=null (inconclusive), which makes Reachy ask
the person by voice. That is the safe answer until a camera check exists.

Safety: requests with "dry_run": true never move the arm (no --go). One request at a time; a second
one gets "aborted: arm busy". Reachy's default arm timeout is short, so raise
MEDICINE_BUDDY_ARM_TIMEOUT_S (the ollama checker alone can take ~20 s per detection).
"""
import argparse, contextlib, io, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCHEMA = "medicine_buddy.arm.v1"
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- medicine name -> item ----------

def resolve_item(items_path, medicine_name):
    """Returns (target, category) or None. Match on the item name or one of its aliases."""
    try:
        items = json.load(open(items_path))
    except FileNotFoundError:
        return None
    want = (medicine_name or "").strip().lower()
    if not want:
        return None
    for name, prof in items.items():
        names = [name] + list(prof.get("aliases", []))
        if want in (n.lower() for n in names):
            return name, prof.get("category") or "box"
    return None


# ---------- reading pick_v4's output ----------

def extract_json(text):
    """The first JSON object in text that has a 'status' key (pick_v4 prints a report, then JSON)."""
    dec, i = json.JSONDecoder(), 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return {}
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict) and "status" in obj:
                return obj
            i = end
        except ValueError:
            i += 1


# pick_v4.main() return value -> (ok, final_state, detail)
def map_status(status, dry_run):
    if status == "PLACED":
        return True, "placed", "grabbed and placed in the bowl"
    if status == "DRY_RUN":
        return True, "placed", "dry_run: detected the object and computed the grasp; the arm did not move"
    table = {
        "NOT_FOUND": "object not found on the table",
        "AMBIGUOUS": "more than one matching object; refusing to guess",
        "NO_GRASP": "object found but no grasp fits the gripper",
        "EMPTY": "grabbed but the gripper is empty",
        "HOLDING": "grabbed but could not place it in the bowl",
        "ARM_ERROR": "the arm refused or failed the grab",
    }
    return False, "failed", table.get(status, f"pick ended with status {status}")


# ---------- the work ----------

class Service:
    def __init__(self, items, arm_url, vlm, bowl, fake=False, out_dir="runs", runner=None, handover=None):
        self.items, self.arm_url, self.vlm, self.bowl = items, arm_url, vlm, bowl
        self.handover = handover
        self.fake, self.out_dir = fake, out_dir
        self.runner = runner            # tests: replaces pick_v4.main
        self.lock = threading.Lock()

    def _run_pick(self, argv):
        if self.runner:
            return self.runner(argv)
        import pick_v4                  # heavy imports (torch, cv2) only when really needed
        return pick_v4.main(argv)

    def result(self, rid, ok, state, **kw):
        out = {"schema": SCHEMA, "message_type": "dispense_result", "request_id": rid, "ok": ok,
               "final_state": state, "slot_id": kw.get("slot_id"), "label": kw.get("label"),
               "score": None, "place_verified": kw.get("place_verified", False),
               "detect_detail": kw.get("detail", ""), "error": kw.get("error")}
        return out

    def dispense(self, body):
        rid = str(body.get("request_id") or "")
        med = body.get("medicine") or {}
        name = med.get("name")
        dry_run = bool(body.get("dry_run", False))
        found = resolve_item(self.items, name)
        if not found:
            return self.result(rid, False, "failed", error=f"no object profile for medicine {name!r}; "
                               f"add it (or an alias) to {os.path.basename(self.items)}")
        target, category = found
        return self._pick_and_place(rid, target, category, dry_run, self.bowl, "bowl",
                                    slot_id=body.get("slot_hint"))

    def fetch(self, body):
        import voice_pick
        rid = str(body.get("request_id") or "")
        dry_run = bool(body.get("dry_run", False))
        said = str(body.get("object") or "")
        m = voice_pick.match_item(said, voice_pick.load_items(self.items))
        res = lambda err: dict(self.result(rid, False, "failed", error=err), message_type="fetch_result")
        if m[0] == "unknown":
            return res(f"I don't know the object {said!r}; add it (or an alias) to {os.path.basename(self.items)}")
        if m[0] == "ambiguous":
            return res(f"{said!r} could mean {m[1][0]} or {m[1][1]}; refusing to guess")
        drop = self.handover or self.bowl
        out = self._pick_and_place(rid, m[1], m[2], dry_run, drop, "hand-over spot (--handover or --bowl)")
        return dict(out, message_type="fetch_result")

    def _pick_and_place(self, rid, target, category, dry_run, drop, drop_name, slot_id=None):
        if not self.lock.acquire(blocking=False):
            return self.result(rid, False, "aborted", error="arm busy with another request")
        try:
            if self.fake:
                return self.result(rid, True, "placed", slot_id=slot_id, label=target.upper(),
                                   place_verified=False, detail="fake mode: nothing was detected or moved")
            if not dry_run and drop is None:
                return self.result(rid, False, "failed", label=target,
                                   error=f"no {drop_name} position configured (start with --bowl X Y / --handover X Y)")
            os.makedirs(self.out_dir, exist_ok=True)
            argv = ["--target", target, "--category", category, "--vlm", self.vlm,
                    "--arm-url", self.arm_url, "--items", self.items, "--no-show", "-y",
                    "--out", os.path.join(self.out_dir, f"{rid or int(time.time())}.jpg")]
            if not dry_run:
                argv += ["--go", "--place", str(drop[0]), str(drop[1])]
            buf, status = io.StringIO(), None
            try:
                with contextlib.redirect_stdout(buf):
                    status = self._run_pick(argv)
            except SystemExit as e:     # pick_v4 exits with a message when the arm page is unreachable etc.
                return self.result(rid, False, "failed", label=target, error=str(e.code or "pick exited"))
            except Exception as e:      # noqa: BLE001 — never let a vision error kill the server
                return self.result(rid, False, "failed", label=target, error=f"{type(e).__name__}: {e}")
            ok, state, detail = map_status(status, dry_run)
            info = extract_json(buf.getvalue())
            if info.get("x_cm") is not None:
                detail += f" (target x={info['x_cm']} y={info['y_cm']} cm)"
            if info.get("warning"):
                detail += f" WARNING: {info['warning']}"
            return self.result(rid, ok, state, label=target, place_verified=(status == "PLACED"),
                               detail=detail, error=None if ok else detail)
        finally:
            self.lock.release()

    def check_taken(self, body):
        return {"schema": SCHEMA, "message_type": "check_taken_result",
                "request_id": str(body.get("request_id") or ""), "ok": True, "taken": None,
                "via": "camera", "detail": "no camera check implemented; ask the person", "error": None}


def make_handler(svc):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "schema": SCHEMA, "fake": svc.fake, "busy": svc.lock.locked(),
                                 "bowl": svc.bowl, "handover": svc.handover, "vlm": svc.vlm})
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(n, 1_000_000)) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
            except ValueError as e:
                return self._send(400, {"ok": False, "error": f"bad JSON: {e}"})
            if body.get("schema") not in (None, SCHEMA):
                return self._send(400, {"ok": False, "error": f"unsupported schema {body.get('schema')!r}"})
            if self.path == "/v1/dispense":
                return self._send(200, svc.dispense(body))
            if self.path == "/v1/fetch":
                return self._send(200, svc.fetch(body))
            if self.path == "/v1/check_taken":
                return self._send(200, svc.check_taken(body))
            self._send(404, {"ok": False, "error": "not found"})

        def log_message(self, fmt, *args):
            print("[arm_service]", fmt % args, file=sys.stderr)
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description="medicine_buddy.arm.v1 server backed by pick_v4")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--items", default=os.path.join(HERE, "items_v1.json"))
    ap.add_argument("--arm-url", default="http://localhost:8890", help="the arm's own click page (pick_v4 --arm-url)")
    ap.add_argument("--vlm", choices=["none", "ollama", "claude"], default="ollama")
    ap.add_argument("--bowl", nargs=2, type=float, metavar=("X", "Y"), help="table x,y (cm) of the bowl")
    ap.add_argument("--handover", nargs=2, type=float, metavar=("X", "Y"),
                    help="table x,y (cm) where fetched objects are put for the person (default: the bowl)")
    ap.add_argument("--fake", action="store_true", help="answer without detection or motion (wiring test)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    a = ap.parse_args(argv)
    svc = Service(a.items, a.arm_url, a.vlm, a.bowl, a.fake, a.out_dir, handover=a.handover)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(svc))
    print(f"arm_service on {a.host}:{a.port}  fake={a.fake} vlm={a.vlm} bowl={a.bowl}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
