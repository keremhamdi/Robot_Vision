# Reachy side: "give me the bottle" (fetch_object)

`fetch_object.patch` is a change for **LosPollosForTheElders** (Tommaso's repo; this session could only read it).
It was tested against that repo's `main` (`6756590`): applies cleanly, 4 new tests pass, and the existing suite
has the same results as before (13 pre-existing errors in `test_reminders.py`: needs `fastapi`, not installed here).

```bash
cd LosPollosForTheElders
git apply /path/to/Robot_Vision/reachy_patch/fetch_object.patch
```

## What it adds
| File | Change |
|---|---|
| `arm_protocol.py`, `http_arm.py` | `FetchCommand` / `FetchResult`, `HttpArmClient.fetch()` -> `POST /v1/fetch` |
| `api.py` | `POST /arm/fetch` (`fetch_object`): asks "Should I bring you the X?" first, only moves after a yes, same voice guard as dispense |
| `conversation/external_tools/fetch_object.py` | the voice tool Reachy's LLM calls |
| `profile.md` | tool enabled + rules (confirm first, no medicines via this tool, no dangerous items) |
| `main.py` | **dry-run was hard-coded on**; now `MEDICINE_BUDDY_DRY_RUN` decides (default is still dry-run) |
| `tests/test_fetch_object.py` | 4 tests |

## Whole system, how to run it
Mac (arm + vision):
```bash
python arm_service.py --bowl -10 25 --handover 0 30 --vlm ollama --arm-url http://localhost:8890
```
Reachy app environment:
```bash
MEDICINE_BUDDY_ARM_URL=http://<mac-ip>:9090
MEDICINE_BUDDY_DRY_RUN=0            # the arm really moves. Leave it at 1 to rehearse: the arm never moves
MEDICINE_BUDDY_ARM_TIMEOUT_S=120    # detection with the ollama checker can take ~20 s
```

## The two flows
- **8:00 medicine (already built on the Reachy side):** the schedule fires -> Reachy says "time for X, should I bring it?"
  -> yes -> `dispense_medicine` -> `/v1/dispense` -> arm finds the box, puts it in the bowl.
  The arm needs to know which physical box is which medicine: add the medicine name as an alias of the item in `items_v1.json`.
- **"Give me the water bottle":** Reachy asks to confirm -> yes -> `fetch_object` -> `/v1/fetch` -> arm finds the bottle,
  puts it at `--handover`.

## Not done / not tested with hardware
- Nothing here moved a real arm. Both sides are tested with the picking step faked.
- `check_taken` always answers "inconclusive", so Reachy asks by voice.
- Reachy's own reminder text must be checked on the robot (timezone is often UTC on the robot: set it on the Family page).
