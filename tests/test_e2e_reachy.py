"""End to end, no hardware: Reachy's MedicineBuddyService -> real HttpArmClient -> arm_service (pick faked).
Needs a LosPollosForTheElders checkout WITH the fetch_object changes (reachy_patch/fetch_object.patch)."""
import os, sys, tempfile, unittest, threading
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LOSPOLLOS = os.environ.get("LOSPOLLOS_DIR", os.path.join(ROOT, "..", "tommasosommaruga", "lospollosfortheelders"))
sys.path.insert(0, os.path.join(LOSPOLLOS, "apps", "medicine_buddy"))
import arm_service as S

try:
    from medicine_buddy.api import MedicineBuddyService
    from medicine_buddy.config import AppConfig
    ok = hasattr(MedicineBuddyService, "fetch_object")
except Exception:  # noqa: BLE001
    ok = False


@unittest.skipUnless(ok, "patched LosPollosForTheElders checkout not importable")
class E2E(unittest.TestCase):
    def setUp(self):
        self.picks = []
        def runner(argv):
            self.picks.append(argv); return "PLACED"
        svc = S.Service(os.path.join(ROOT, "items_v1.json"), "http://x", "none", (-10, 25), runner=runner,
                        handover=(0, 30), out_dir=tempfile.mkdtemp())
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(svc))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        self.srv.shutdown()

    def service(self, dry):
        return MedicineBuddyService(self.tmp / "e2e.db", AppConfig(dry_run=dry, arm_url=self.url))

    def test_fetch_water_bottle(self):
        svc = self.service(dry=False)
        self.assertEqual(svc.fetch_object({"object": "water bottle", "confirmed": False})["status"], "needs_confirm")
        self.assertEqual(self.picks, [])                       # nothing moves before the yes
        r = svc.fetch_object({"object": "water bottle", "confirmed": True})
        self.assertEqual(r["status"], "ok"); self.assertEqual(r["final_state"], "placed")
        a = self.picks[0]
        self.assertEqual(a[a.index("--target") + 1], "water bottle"); self.assertIn("--go", a)
        self.assertEqual(a[a.index("--place") + 1:][:2], ["0", "30"])
        svc.close()

    def test_fetch_in_dry_run_does_not_move(self):
        svc = self.service(dry=True)
        svc.fetch_object({"object": "water bottle", "confirmed": False})
        svc.fetch_object({"object": "water bottle", "confirmed": True})
        self.assertTrue(self.picks and "--go" not in self.picks[0])
        svc.close()

    def test_unknown_object_fails_politely(self):
        svc = self.service(dry=False)
        svc.fetch_object({"object": "car keys", "confirmed": False})
        r = svc.fetch_object({"object": "car keys", "confirmed": True})
        self.assertEqual(r["status"], "failed"); self.assertEqual(self.picks, [])
        self.assertIn("Sorry", r["speak_hint"])
        svc.close()

    def test_arm_pc_down_is_a_spoken_error(self):
        svc = MedicineBuddyService(self.tmp / "d.db", AppConfig(dry_run=False, arm_url="http://127.0.0.1:9"))
        svc.fetch_object({"object": "water bottle", "confirmed": False})
        r = svc.fetch_object({"object": "water bottle", "confirmed": True})
        self.assertEqual(r["status"], "error"); self.assertIn("arm computer", r["speak_hint"])
        svc.close()


if __name__ == "__main__":
    unittest.main()
