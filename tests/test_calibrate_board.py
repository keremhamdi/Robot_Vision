"""calibrate_board.py against a synthetic camera with a KNOWN pixel<->table mapping."""
import json, os, sys, tempfile, unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import cv2
    import calibrate_board as CB
except ImportError:
    cv2 = None

COLS, ROWS, SQ = 9, 6, 2.5
W, H = 1280, 720
# true camera: table cm -> pixel (a tilted view)
H_TRUE = np.array([[22.0, 3.0, 640.0], [1.5, 14.0, 330.0], [0.0004, 0.012, 1.0]])


def to_px(xy):
    p = np.c_[np.atleast_2d(xy), np.ones(len(np.atleast_2d(xy)))] @ H_TRUE.T
    return p[:, :2] / p[:, 2:3]


def render(x0, y0, theta_deg, mirror, path, rng):
    """Board placed on the table at (x0, y0), rotated, optionally mirrored; returns the arm xy of each corner."""
    px, m = 120, 120
    CB.make_board(COLS, ROWS, os.path.join(tempfile.gettempdir(), "b.png"), px)
    board = cv2.imread(os.path.join(tempfile.gettempdir(), "b.png"), 0)
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M = np.diag([-1.0, 1.0]) if mirror else np.eye(2)

    def board_to_table(c, r):
        return R @ (M @ np.array([c * SQ, r * SQ])) + [x0, y0]

    # board image pixel -> board cm -> table cm -> camera pixel (a homography through 4 points)
    src = np.float32([[m + px, m + px], [m + 7 * px, m + px], [m + px, m + 5 * px], [m + 7 * px, m + 5 * px]])
    cm = [((u - (m + px)) / px * SQ, (v - (m + px)) / px * SQ) for u, v in src]
    dst = np.float32([to_px(R @ (M @ np.array(c)) + [x0, y0])[0] for c in cm])
    Hm = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(board, Hm, (W, H), borderValue=200)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img.astype(float) + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return board_to_table


@unittest.skipIf(cv2 is None, "opencv not installed")
class Calibrate(unittest.TestCase):
    def run_case(self, x0, y0, theta, mirror, noise=0.1, append=None, out=None):
        rng = np.random.default_rng(1)
        d = tempfile.mkdtemp()
        img = os.path.join(d, "t.png")
        b2t = render(x0, y0, theta, mirror, img, rng)
        m = self.touch(img, rng, noise)
        out = out or os.path.join(d, "camera.json")
        held = CB.calibrate(cv2.imread(img), COLS, ROWS, SQ, m, out, os.path.join(d, "v.jpg"),
                            os.path.join(d, "c.jpg"), append)
        return out, held

    def touch(self, img, rng, noise=0.1, scale=1.0):
        """What the arm tip reads when touching the corners the script LABELS A, B, C in the picture
        (the script may number the corners from either end of the board): the true table position
        under that pixel, plus touching noise."""
        corners = CB.detect(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY), COLS, ROWS)
        Hinv = np.linalg.inv(H_TRUE)
        out = {}
        for n, (c, r) in {"A": (0, 0), "B": (COLS - 1, 0), "C": (0, ROWS - 1), "D": (COLS - 1, ROWS - 1)}.items():
            u, v = corners[r * COLS + c]
            p = Hinv @ [u, v, 1.0]
            out[n] = tuple(p[:2] / p[2] + rng.normal(0, noise, 2))
        return out

    def check_mapping(self, out, tol=0.3):
        with open(out) as fh:
            d = json.load(fh)
        pairs = np.array(d["pairs"])
        H = CB.fit_h(pairs[:, :2] / d["size"], pairs[:, 2:])
        # compare with the TRUE mapping at points on the table
        pts = np.array([[x, y] for x in range(-8, 9, 4) for y in range(26, 38, 4)], float)
        got = CB.apply_h(H, to_px(pts) / d["size"])
        err = np.linalg.norm(got - pts, axis=1)
        self.assertLess(err.max(), tol, f"worst error {err.max():.2f} cm")

    def test_normal_board(self):
        out, held = self.run_case(-6, 26, 12, False)
        self.assertLess(held.max(), 0.3)
        self.check_mapping(out)

    def test_mirrored_axes_and_rotation(self):
        out, held = self.run_case(0, 26, 200, True)
        self.assertLess(held.max(), 0.3)
        self.check_mapping(out)

    def test_two_positions_append(self):
        out, _ = self.run_case(-8, 24, 0, False)
        out, held = self.run_case(2, 31, 30, False, append=out, out=out)
        with open(out) as fh:
            self.assertEqual(len(json.load(fh)["pairs"]), 2 * COLS * ROWS)
        self.check_mapping(out)

    def bad_touch(self, mutate, force=False):
        import io, contextlib
        rng = np.random.default_rng(2)
        d = tempfile.mkdtemp(); img = os.path.join(d, "t.png"); out = os.path.join(d, "o.json")
        render(-6, 26, 0, False, img, rng)
        m = mutate(self.touch(img, rng, 0.0))
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                CB.calibrate(cv2.imread(img), COLS, ROWS, SQ, m, out, os.path.join(d, "v.jpg"),
                             os.path.join(d, "c.jpg"), force=force)
            exited = False
        except SystemExit:
            exited = True
        return exited, os.path.exists(out), buf.getvalue()

    def test_swapped_corners_refused(self):
        exited, wrote, text = self.bad_touch(lambda m: dict(m, B=m["C"], C=m["B"]))
        self.assertTrue(exited); self.assertFalse(wrote); self.assertIn("PROBLEM", text)

    def test_typo_refused(self):
        exited, wrote, _ = self.bad_touch(lambda m: dict(m, B=(m["B"][0] + 3.0, m["B"][1])))
        self.assertTrue(exited); self.assertFalse(wrote)

    def test_wrong_square_size_refused(self):
        import io, contextlib
        rng = np.random.default_rng(2)
        d = tempfile.mkdtemp(); img = os.path.join(d, "t.png"); out = os.path.join(d, "o.json")
        render(-6, 26, 0, False, img, rng)
        m = self.touch(img, rng, 0.0)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            CB.calibrate(cv2.imread(img), COLS, ROWS, SQ * 1.3, m, out, os.path.join(d, "v.jpg"), os.path.join(d, "c.jpg"))
        self.assertFalse(os.path.exists(out))

    def test_force_writes_anyway(self):
        exited, wrote, text = self.bad_touch(lambda m: dict(m, B=m["C"], C=m["B"]), force=True)
        self.assertFalse(exited); self.assertTrue(wrote); self.assertIn("Do not trust", text)

    def test_careless_but_honest_touches_pass(self):
        out, held = self.run_case(-6, 26, 12, False, noise=0.25)     # +-2.5 mm touching error is normal
        self.assertLess(held.max(), 0.5)

    def test_adjacent_corner_touched_by_mistake_is_refused(self):
        # tip on the neighbouring corner of B (one square = 2.5 cm away) instead of B
        exited, wrote, _ = self.bad_touch(lambda m: dict(m, B=(m["B"][0] + SQ, m["B"][1])))
        self.assertTrue(exited); self.assertFalse(wrote)

    def test_three_corners_are_enough(self):
        import io, contextlib
        rng = np.random.default_rng(3)
        d = tempfile.mkdtemp(); img = os.path.join(d, "t.png"); out = os.path.join(d, "o.json")
        render(-6, 26, 12, False, img, rng)
        m = self.touch(img, rng, 0.1); m.pop("D")
        with contextlib.redirect_stdout(io.StringIO()):
            CB.calibrate(cv2.imread(img), COLS, ROWS, SQ, m, out, os.path.join(d, "v.jpg"), os.path.join(d, "c.jpg"))
        self.check_mapping(out, tol=0.6)

    def test_realistic_touching_error_is_not_refused(self):
        import io, contextlib
        rng = np.random.default_rng(7)
        d = tempfile.mkdtemp(); img = os.path.join(d, "t.png")
        render(-6, 26, 12, False, img, rng)
        base = self.touch(img, rng, 0.0)
        refused = 0
        for _ in range(60):
            m = {k: (v[0] + rng.normal(0, 0.25), v[1] + rng.normal(0, 0.25)) for k, v in base.items()}   # 2.5 mm
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    CB.calibrate(cv2.imread(img), COLS, ROWS, SQ, m, os.path.join(d, "o.json"),
                                 os.path.join(d, "v.jpg"), os.path.join(d, "c.jpg"))
            except SystemExit:
                refused += 1
        self.assertEqual(refused, 0)

    def test_live_camera_loop_with_small_board(self):
        """SPACE is only accepted once the board is FOUND, even when the board looks small in the frame."""
        rng = np.random.default_rng(1)
        d = tempfile.mkdtemp(); img = os.path.join(d, "t.png")
        render(-6, 26, 12, False, img, rng)
        board_frame = cv2.imread(img)
        blank = np.full((H, W, 3), 130, np.uint8)
        seq = {"n": 0}

        class FakeCap:
            def __init__(self, i): pass
            def isOpened(self): return True
            def read(self):
                seq["n"] += 1
                return True, (blank if seq["n"] < 4 else board_frame)
            def release(self): pass

        keys = iter([ord(" "), ord(" "), ord(" "), ord(" ")] + [ord(" ")] * 10)   # SPACE pressed on every frame
        saved = (cv2.VideoCapture, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows)
        cv2.VideoCapture, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = (
            FakeCap, lambda n, f: None, lambda ms: next(keys), lambda: None)
        try:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                frame = CB.grab_camera(0, COLS, ROWS)
        finally:
            cv2.VideoCapture, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = saved
        self.assertEqual(seq["n"], 4)            # blank frames were skipped; the first frame with the board was taken
        self.assertTrue((frame == board_frame).all())

    def test_missing_board_exits(self):
        blank = np.full((H, W, 3), 128, np.uint8)
        with self.assertRaises(SystemExit):
            CB.calibrate(blank, COLS, ROWS, SQ, {}, "/tmp/x.json", "/tmp/v.jpg", "/tmp/c.jpg")


if __name__ == "__main__":
    unittest.main()
