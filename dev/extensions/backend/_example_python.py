# strata-hooks: before_request, after_response
#
# Example Python hook for Strata Dev. Files starting with "_" are not loaded: rename it to example_python.py to try it.
# The host runs it once per call: {"event": ..., "payload": ...} arrives as JSON on stdin; print one JSON object.
# {"payload": {...}} replaces the payload (before_request); print nothing to keep it. Lines on stderr go to the Logs tab.
import json
import sys

data = json.load(sys.stdin)
event, payload = data["event"], data["payload"]
if event == "before_request":
    print(f"python hook saw: {str(payload.get('message', ''))[:60]}", file=sys.stderr)
elif event == "after_response":
    print(f"python hook: {len(payload.get('text', ''))} characters", file=sys.stderr)
