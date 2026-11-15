# Hackathon Arm: ElRobot household pick-up helper

Status as of 2026-11-15, checked against the repo (`keremhamdi/Robot_Vision`, one commit).
Nothing was executed here: the sandbox has no torch, cv2, arm or camera. Everything below comes from reading the code and the saved eval CSVs.

## Key finding
**This repo is the vision side only.** `pick_v4.py` talks over HTTP (`/status`, `/frame.jpg`, `/api/grab`) to an arm page (`arm/click.py`) that is **not in this repo**.

**Correction (2026-11-15):** `LosPollosForTheElders` is the **Reachy Mini** app (Medicine Buddy), not the arm code. It contains no `arm/click.py`. It talks to the arm only through the `medicine_buddy.arm.v1` HTTP contract (`POST /v1/dispense`, `/v1/check_taken`), and the arm side there is still fake. Which repo holds the `arm.click` page is still to confirm.

**Combining them:** `arm_service.py` (this repo) is the arm-side server for that contract. It maps the medicine name to an object in `items_v1.json`, runs `pick_v4`, and answers in the format Reachy expects. On Reachy only `MEDICINE_BUDDY_ARM_URL=http://<mac>:9090` changes, and the app's forced dry-run must be switched off. See `arm_service.py` for the details and `tests/test_arm_service.py` for the checks.

**Update:** `arm_service.py` also serves `/v1/fetch` ("give me the water bottle"). The matching Reachy-side change (a `fetch_object` voice tool, plus un-hard-coding dry-run) is `reachy_patch/fetch_object.patch`; see `reachy_patch/README.md`. `voice_pick.py` is the direct voice-to-arm path on the Mac without Reachy.

## What exists (verified in code)
| Part | State |
|---|---|
| `target_selector_v5.py` | SAM 3 (HF Transformers, `facebook/sam3`) proposes masks from a target prompt plus a category prompt. A VLM (`--vlm ollama` = qwen2.5vl:7b, `claude`, or `none`) checks each crop. Result: `GO` / `NOT_FOUND` / `AMBIGUOUS`. Device order is cuda, then **mps**, then cpu, with `PYTORCH_ENABLE_MPS_FALLBACK=1`, so the M3 Max works as written. |
| `grasp_planner_v3.py` | Top-down grasp (center and closing angle) from the mask. |
| `pick_v4.py` | The glue. Takes the frame from the arm page (`GET /frame.jpg`), picks the object, computes the grasp point, converts pixel to table x,y in cm, then sends `POST /api/grab {u, v, height, above}`. Dry run by default, `--go` moves the arm, `--place X Y` puts the object down after. Has `--check` for calibration accuracy and an offline mode (`--image` + `--calib`). |
| `items_v1.json` | Per-object profile: height, width, grasp style (side / top / auto). Has "blue pill box". |
| `place_snippet_v1.py` | A `place` action for his `arm/click.py`. **Untested**, per its own docstring, and not yet in his code. |
| `eval_v2.py`, `tests_v1.csv` | Eval over 14 saved photos (4 images, 7 objects). |
| Reachy Mini | **Nothing in the repo.** The only mentions are comments ("Reachy: I can't find it"). |

## Answers to the open questions
- **What the arm accepts:** a pixel `(u, v)` plus `height` and `above` in cm, via his page. The pixel is mapped to table x,y by his homography. `pick_v4` computes x,y itself and sends back the pixel his model maps to that point, so his code needs no change.
- **Calibration:** `camera.json` has **5 marks** at 1280 px. That is too few to measure honest accuracy, which needs 6 or more and ideally 8 or more spread over the pick area. Run `python pick_v4.py --check` on the Mac (needs cv2). The tool warns if worst error is above 1 cm.
- **Depth without a camera:** already handled. It uses the table-height assumption plus calibration, taking the point where the object touches the table and offsetting by half the object width. Heights come from `items_v1.json`. Visual servoing is not implemented and is not needed if calibration is good.
- **Demo objects (from `items_v1.json` / tests):** energy drink can, water bottle, Feetech servo box, white box, foam ring, blue pill box. The test photos also use a red soda can and black glasses case as "absent" or extra objects. The medication box is **detected like the others**, not fixed.

## Eval results (saved CSVs, 14 cases, device and machine unknown)
| Checker | Pass | Missed | Wrong GO (dangerous) | Time per query |
|---|---|---|---|---|
| `--vlm ollama` | 12 / 14 | 2 (glasses case gives AMBIGUOUS; servo box in wide2 gives NOT_FOUND) | 0 | 1.8 to 21 s, typically 5 to 15 s |
| `--vlm none` | 10 / 14 | 4 | 0 | about 1.6 s (looks cached) |

Both runs have **zero wrong GOs**: misses fail safe. Ollama is the better checker but is slow for a live demo. `--vlm none` is fast but misses the servo box and the can in the wide shot. Neither number is measured on the wrist camera at the final demo layout.

## Plan (revised)
1. **Calibrate properly.** Redo to 8+ marks spread over the pick area on his page, then `pick_v4.py --check` until worst error is under 1 cm. This is the biggest risk to the demo.
2. **Run the dry-run loop on the Mac:** `python pick_v4.py --target "blue pill box" --category box --vlm ollama` against the live arm page, check `pick.jpg`, then `--go`. Time SAM 3 + qwen on MPS: ollama at 5 to 21 s per query may be too slow for stage.
3. **Add the `place` action** from `place_snippet_v1.py` into his `arm/click.py`, test with an empty gripper and a high `height`.
4. **Trim the object list to the demo set** and add each to `tests_v1.csv` with wrist-camera photos at the real layout, then rerun `eval_v2.py`. Add the medication box profile (real dimensions) to `items_v1.json`.
5. **Reachy Mini trigger (not started):** a small HTTP endpoint wrapping `pick_v4.main()` (e.g. `POST /fetch {"target": "blue pill box"}`), which Reachy calls. `main()` returns a status string (`PLACED`, `HOLDING`, `NOT_FOUND`, `AMBIGUOUS`, `EMPTY`, `ARM_ERROR`, ...) that can be sent back for Reachy to speak. Needs the medication calendar logic from the team.
6. **Demo fallback:** a scripted pick at fixed table x,y for the fixed layout. `pick_v4` computes x,y but has no mode to send fixed coordinates; add a `--xy X Y` flag.

## Housekeeping
- Seven overlapping script versions (`pick_v2` to `v4`, `target_selector` plus `_v2`, `_v3`, `_v5` and `target_selector (1).py`, `eval_v1`/`v2`, `grasp_planner_v2`/`v3`) and many result JPGs are committed. Only `pick_v4`, `target_selector_v5`, `grasp_planner_v3`, `eval_v2`, `items_v1.json`, `tests_v1.csv` are in use. Safe to move the rest to an `old/` folder, and `__pycache__` should be git-ignored. There is no requirements file.
- Docs say `target_selector_v5` default VLM is `claude`, which needs an API key. For offline demo pass `--vlm ollama` or `none`.

## Open questions
- Is the arm page (`python -m arm.click`) on the same Mac, and is it already calibrated with 6+ marks?
- What does Reachy Mini already do (calendar, speech)? Nothing is in this repo.
- Real size of the medication box?
