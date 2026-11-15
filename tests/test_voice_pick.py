import os, sys, unittest, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import voice_pick as V

NAMES = V.load_items(os.path.join(ROOT, "items_v1.json"))


class Match(unittest.TestCase):
    def ok(self, text, target):
        m = V.match_item(text, NAMES)
        self.assertEqual(m[:2], ("ok", target), text)

    def test_hits(self):
        self.ok("please bring me the water bottle", "water bottle")
        self.ok("can you grab the energy drink can", "energy drink can")
        self.ok("bring my medicine box", "blue pill box")
        self.ok("get the pillbox", "blue pill box")
        self.ok("Feetech servo box", "Feetech servo box")
        self.ok("the foam ring please", "foam ring")
        self.ok("pick the white ring", "foam ring")
        self.ok("pick the white box", "white box")

    def test_refuses_unknown_and_close_but_wrong(self):
        for t in ("bring me the red soda can", "hello there", "", "the blue box", "pick up the keys"):
            self.assertEqual(V.match_item(t, NAMES)[0], "unknown", t)


class Handle(unittest.TestCase):
    def args(self, **kw):
        d = dict(vlm="none", arm_url="http://x", items="items_v1.json", dry_run=False, auto=False, bowl=None)
        d.update(kw)
        return types.SimpleNamespace(**d)

    def test_passes_target_and_flags(self):
        seen = []
        out = V.handle("grab the water bottle", NAMES, self.args(auto=True, bowl=(-10, 25)),
                       pick_main=lambda argv: seen.append(argv) or "PLACED")
        self.assertEqual(out, "PLACED")
        argv = seen[0]
        self.assertEqual(argv[argv.index("--target") + 1], "water bottle")
        self.assertEqual(argv[argv.index("--category") + 1], "bottle")
        self.assertIn("--go", argv); self.assertIn("-y", argv); self.assertIn("--place", argv)

    def test_image_mode_never_goes_even_with_auto(self):
        seen = []
        V.handle("pick the white ring", NAMES, self.args(image="wrist1.jpeg", calib="camera.json", out="/tmp/v.jpg",
                 auto=True, bowl=(1, 2)), pick_main=lambda argv: seen.append(argv) or "DRY_RUN")
        argv = seen[0]
        self.assertEqual(argv[argv.index("--target") + 1], "foam ring")
        self.assertIn("--image", argv)
        for bad in ("--go", "-y", "--place", "--no-show"):
            self.assertNotIn(bad, argv)

    def test_dry_run_never_goes(self):
        seen = []
        V.handle("water bottle", NAMES, self.args(dry_run=True, auto=True),
                 pick_main=lambda argv: seen.append(argv) or "DRY_RUN")
        self.assertNotIn("--go", seen[0]); self.assertNotIn("-y", seen[0])

    def test_unknown_object_never_calls_arm(self):
        seen = []
        out = V.handle("bring me the keys", NAMES, self.args(), pick_main=lambda argv: seen.append(argv))
        self.assertEqual(out, "UNKNOWN"); self.assertEqual(seen, [])

    def test_pick_exit_is_reported_not_raised(self):
        def stop(argv): sys.exit("arm page not reachable")
        out = V.handle("water bottle", NAMES, self.args(), pick_main=stop)
        self.assertTrue(out.startswith("STOPPED"))


if __name__ == "__main__":
    unittest.main()
