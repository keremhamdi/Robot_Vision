"""
place_snippet_v1.py — NOT a program to run. It is a small addition for your friend's
arm/click.py so the arm can put an object down at a table point given in cm.

Why x, y in cm and not a pixel: the drop-off point is a fixed place on the table (for
example the hand-over zone). Measure it once with the arm itself:
    python -m arm.reach where        (move the tip over the spot, read x and y)
That way placing does not depend on the camera at all.

WHERE IT GOES: in arm/click.py, inside `async def api(self, name, body)`, directly
above the final lines
        else:
            raise RuntimeError(f"Unknown action {name}")
paste this block (same indentation as the other `elif name == ...` blocks):

        elif name == "place":
            if self.busy:
                raise RuntimeError("Already moving.")
            x, y = float(body["x"]), float(body["y"])
            above, height = float(body.get("above", 12)), float(body.get("height", 4))
            self.busy = True
            self.log = f"Placing at x={x:.1f}, y={y:.1f}..."
            try:
                r = self.reach()
                await r.goto(x, y, above)                 # carry it over, staying high
                await r.goto(x, y, height, line=True)     # straight down
                await self.arm.move({GRIPPER_ID: 100}, 0.8)   # let go
                await r.goto(x, y, above, line=True)      # straight back up
                self.log = f"Placed at x={x:.1f}, y={y:.1f}."
            finally:
                self.busy = False

It only uses functions that already exist in his code (Reacher.goto, Arm.move), with the
same safety checks (nothing goes below the table, unreachable points are refused).
I could not run it here: it needs the real arm and Station. Test it first with an empty
gripper and a high `height`.

Then:  python pick_v2.py --target "blue pill box" --category box --vlm ollama --go --place -10 25
"""
