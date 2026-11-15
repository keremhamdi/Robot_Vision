"""
voice_pick.py — say what you want, the arm finds it and grabs it.

    microphone -> speech-to-text (local Whisper) -> which known object? -> pick_v4 (SAM 3 finds it,
    table x,y, arm grabs) -> optional put-down at a fixed spot

Only objects listed in items_v1.json (names + "aliases") can be asked for. Anything else is refused
("I don't know that object") — it never grabs the "closest" thing.

Install once (on the Mac):
    pip install faster-whisper sounddevice
Run (arm page running, camera calibrated, same as pick_v4):
    python voice_pick.py --vlm ollama                      # press Enter, speak, press Enter
    python voice_pick.py --vlm ollama --bowl -10 25        # then put it down at table x=-10, y=25 cm
    python voice_pick.py --vlm ollama --auto               # no "press Enter to grab" prompt (demo day)
Try without microphone or arm:
    python voice_pick.py --text "please bring me the water bottle" --dry-run

By default pick_v4 still asks "Press Enter to grab" before the arm moves; --auto removes that.
--dry-run detects and shows the target but never moves the arm.
"""
import argparse, difflib, json, os, re, sys, tempfile, wave

HERE = os.path.dirname(os.path.abspath(__file__))
MATCH_MIN = 0.80      # how close the heard words must be to an item name
TIE_GAP = 0.05        # two different items this close -> ask again instead of guessing


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def load_items(path):
    items = json.load(open(path))
    names = {}                                   # phrase -> (target, category)
    for target, prof in items.items():
        for ph in [target] + list(prof.get("aliases", [])):
            names[" ".join(_norm(ph))] = (target, prof.get("category") or "box")
    return names


def match_item(text, names):
    """Returns ('ok', target, category) | ('unknown',) | ('ambiguous', [targets]).
    Looks for each known phrase inside the heard words (exact, or close spelling like 'pillbox')."""
    words = _norm(text)
    if not words:
        return ("unknown",)
    best = {}                                    # target -> (score, category)
    for phrase, (target, cat) in names.items():
        n = len(phrase.split())
        score = 0.0
        for size in {max(1, n - 1), n, n + 1}:
            for i in range(0, len(words) - size + 1):
                gram = " ".join(words[i:i + size])
                r = difflib.SequenceMatcher(None, gram, phrase).ratio()
                r = max(r, difflib.SequenceMatcher(None, gram.replace(" ", ""), phrase.replace(" ", "")).ratio())
                score = max(score, r)
        if score > best.get(target, (0, ""))[0]:
            best[target] = (score, cat)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    if not ranked or ranked[0][1][0] < MATCH_MIN:
        return ("unknown",)
    if len(ranked) > 1 and ranked[1][1][0] >= MATCH_MIN and ranked[0][1][0] - ranked[1][1][0] < TIE_GAP:
        return ("ambiguous", [ranked[0][0], ranked[1][0]])
    return ("ok", ranked[0][0], ranked[0][1][1])


# ---------- microphone + Whisper ----------

def record(rate=16000):
    import numpy as np, sounddevice as sd
    input("Press Enter, then speak... ")
    frames = []
    stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                            callback=lambda d, n, t, s: frames.append(d.copy()))
    with stream:
        input("Listening... press Enter when you are done. ")
    return np.concatenate(frames)[:, 0] if frames else np.zeros(0, "float32")


class Listener:
    def __init__(self, model, lang, names):
        from faster_whisper import WhisperModel
        print(f"Loading speech model '{model}' (first run downloads it)...")
        self.m = WhisperModel(model, device="cpu", compute_type="int8")
        self.lang = lang
        # telling Whisper which words to expect makes it much better at e.g. "Feetech"
        self.prompt = "Objects: " + ", ".join(sorted({t for t, _ in names.values()})) + "."

    def hear(self):
        audio = record()
        if audio.size < 16000 * 0.4:
            return ""
        segs, _ = self.m.transcribe(audio, language=self.lang, initial_prompt=self.prompt, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()


# ---------- one command ----------

def run_pick(target, category, a, pick_main=None):
    argv = ["--target", target, "--category", category, "--vlm", a.vlm, "--arm-url", a.arm_url,
            "--items", a.items, "--no-show"]
    if not a.dry_run:
        argv.append("--go")
        if a.auto:
            argv.append("-y")
        if a.bowl:
            argv += ["--place", str(a.bowl[0]), str(a.bowl[1])]
    if pick_main is None:
        import pick_v4
        pick_main = pick_v4.main
    try:
        return pick_main(argv)
    except SystemExit as e:
        return f"STOPPED: {e.code}"


SAY = {"NOT_FOUND": "I can't see it on the table.",
       "AMBIGUOUS": "I see more than one that fits; I won't guess.",
       "NO_GRASP": "I see it, but I can't grip it.",
       "EMPTY": "I grabbed, but my gripper is empty.",
       "HOLDING": "Got it, I'm holding it.",
       "PLACED": "Done, it is placed.",
       "DRY_RUN": "Dry run: I found it and the target is shown in pick.jpg."}


def handle(text, names, a, pick_main=None):
    """One spoken sentence -> maybe one pick. Returns the outcome string (for tests and logs)."""
    print(f'Heard: "{text}"')
    m = match_item(text, names)
    if m[0] == "unknown":
        print("I don't know that object. I can fetch: " + ", ".join(sorted({t for t, _ in names.values()})))
        return "UNKNOWN"
    if m[0] == "ambiguous":
        print(f"Did you mean {m[1][0]} or {m[1][1]}? Say it again more clearly.")
        return "AMBIGUOUS_SPEECH"
    _, target, category = m
    print(f"-> {target}")
    status = run_pick(target, category, a, pick_main)
    print(SAY.get(str(status), f"Result: {status}"))
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description="speak to fetch")
    ap.add_argument("--items", default=os.path.join(HERE, "items_v1.json"))
    ap.add_argument("--vlm", choices=["none", "ollama", "claude"], default="ollama")
    ap.add_argument("--arm-url", default="http://localhost:8890")
    ap.add_argument("--bowl", nargs=2, type=float, metavar=("X", "Y"), help="put the object down here (cm)")
    ap.add_argument("--auto", action="store_true", help="grab without asking 'Press Enter'")
    ap.add_argument("--dry-run", action="store_true", help="never move the arm")
    ap.add_argument("--text", help="skip the microphone: use this sentence once")
    ap.add_argument("--whisper-model", default="small.en", help="e.g. base.en (faster), small.en, medium.en")
    ap.add_argument("--lang", default="en")
    a = ap.parse_args(argv)
    names = load_items(a.items)
    if a.text:
        handle(a.text, names, a)
        return
    ear = Listener(a.whisper_model, a.lang, names)
    print("Ready. Ctrl+C to quit.")
    while True:
        try:
            text = ear.hear()
        except KeyboardInterrupt:
            break
        if text:
            handle(text, names, a)
        else:
            print("(heard nothing)")


if __name__ == "__main__":
    main()
