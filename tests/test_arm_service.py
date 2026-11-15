"""Wiring tests for arm_service.py. Uses stdlib unittest (python -m unittest discover tests).

If the LosPollosForTheElders checkout is found (env LOSPOLLOS_DIR, default ../tommasosommaruga/lospollosfortheelders),
the round-trip test drives the service with Reachy's real HttpArmClient."""
import os, sys, threading, unittest, json, urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import arm_service as S

ITEMS = os.path.join(ROOT, "items_v1.json")
LOSPOLLOS = os.environ.get("LOSPOLLOS_DIR", os.path.join(ROOT, "..", "tommasosommaruga", "lospollosfortheelders"))
sys.path.insert(0, os.path.join(LOSPOLLOS, "apps", "medicine_buddy"))


def serve(svc):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(svc))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


class Resolve(unittest.TestCase):
    def test_alias_and_name(self):
        self.assertEqual(S.resolve_item(ITEMS, "Pill Box"), ("blue pill box", "box"))
        self.assertEqual(S.resolve_item(ITEMS, "energy drink can"), ("energy drink can", "can"))
        self.assertIsNone(S.resolve_item(ITEMS, "Aspirin"))

    def test_extract_json(self):
        txt = 'Calibration: 5 marks {not json}\n{\n "status": "GO", "x_cm": 1.5\n}\nsaved pick.jpg'
        self.assertEqual(S.extract_json(txt)["x_cm"], 1.5)


class Dispense(unittest.TestCase):
    def post(self, base, path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=10))

    def run_with(self, status, body, bowl=(-10, 25)):
        calls = []
        def runner(argv):
            calls.append(argv)
            print('{"status": "GO", "x_cm": 3.0, "y_cm": 4.0}')
            return status
        srv, base = serve(S.Service(ITEMS, "http://x", "none", bowl, runner=runner, out_dir="/tmp/arm_service_test"))
        try:
            return self.post(base, "/v1/dispense", body), calls
        finally:
            srv.shutdown()

    def req(self, name="pill box", dry=False):
        return {"schema": S.SCHEMA, "request_id": "r1", "medicine": {"id": 1, "name": name}, "dry_run": dry}

    def test_placed_moves_arm(self):
        r, calls = self.run_with("PLACED", self.req())
        self.assertTrue(r["ok"]); self.assertEqual(r["final_state"], "placed"); self.assertTrue(r["place_verified"])
        self.assertIn("--go", calls[0]); self.assertEqual(calls[0][calls[0].index("--place") + 1:][:2], ["-10", "25"])

    def test_dry_run_never_moves(self):
        r, calls = self.run_with("DRY_RUN", self.req(dry=True))
        self.assertTrue(r["ok"]); self.assertFalse(r["place_verified"])
        self.assertNotIn("--go", calls[0])

    def test_failures_are_not_ok(self):
        for st in ("NOT_FOUND", "AMBIGUOUS", "EMPTY", "ARM_ERROR", "HOLDING", "NO_GRASP"):
            r, _ = self.run_with(st, self.req())
            self.assertFalse(r["ok"], st); self.assertEqual(r["final_state"], "failed"); self.assertTrue(r["error"])

    def test_unknown_medicine_and_missing_bowl(self):
        r, calls = self.run_with("PLACED", self.req("Aspirin"))
        self.assertFalse(r["ok"]); self.assertEqual(calls, [])
        r, calls = self.run_with("PLACED", self.req(), bowl=None)
        self.assertFalse(r["ok"]); self.assertEqual(calls, [])

    def test_pick_crash_is_a_failure_not_a_500(self):
        def boom(argv): raise RuntimeError("cuda exploded")
        srv, base = serve(S.Service(ITEMS, "http://x", "none", (0, 0), runner=boom, out_dir="/tmp/arm_service_test"))
        try:
            r = self.post(base, "/v1/dispense", self.req())
        finally:
            srv.shutdown()
        self.assertFalse(r["ok"]); self.assertIn("cuda exploded", r["error"])

    def test_check_taken_is_inconclusive(self):
        srv, base = serve(S.Service(ITEMS, "http://x", "none", None, fake=True))
        try:
            r = self.post(base, "/v1/check_taken", {"request_id": "r2"})
        finally:
            srv.shutdown()
        self.assertIsNone(r["taken"])


@unittest.skipUnless(os.path.isdir(LOSPOLLOS), "LosPollosForTheElders checkout not found")
class RoundTripWithReachyClient(unittest.TestCase):
    def test_real_client(self):
        from medicine_buddy.http_arm import HttpArmClient
        svc = S.Service(ITEMS, "http://x", "none", None, fake=True)
        srv, base = serve(svc)
        try:
            c = HttpArmClient(base, timeout_s=10)
            d = c.dispense(medicine_id=1, medicine_name="Pill box", person="Mario")
            self.assertTrue(d.ok); self.assertEqual(d.final_state, "placed")
            miss = c.dispense(medicine_id=2, medicine_name="Aspirin")
            self.assertFalse(miss.ok)      # unknown medicine: a clear failure, not a guess
            self.assertIn("no object profile", miss.error)
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
