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
        for n, (c, r) in {"A": (0, 0), "B": (COLS - 1, 0), "C": (0, ROWS - 1)}.items():
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

    def test_wrong_square_size_is_flagged(self):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rng = np.random.default_rng(2)
            d = tempfile.mkdtemp(); img = os.path.join(d, "t.png")
            render(-6, 26, 0, False, img, rng)
            m = self.touch(img, rng, 0.0)
            CB.calibrate(cv2.imread(img), COLS, ROWS, SQ * 1.3, m, os.path.join(d, "o.json"),
                         os.path.join(d, "v.jpg"), os.path.join(d, "c.jpg"))
        self.assertIn("WARNING", buf.getvalue())

    def test_missing_board_exits(self):
        blank = np.full((H, W, 3), 128, np.uint8)
        with self.assertRaises(SystemExit):
            CB.calibrate(blank, COLS, ROWS, SQ, {}, "/tmp/x.json", "/tmp/v.jpg", "/tmp/c.jpg")


if __name__ == "__main__":
    unittest.main()
