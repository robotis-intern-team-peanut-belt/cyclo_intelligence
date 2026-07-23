"""Customizable label constants for the L2 scoring app.

Edit these to match your team's failure taxonomy. The frontend reads them
from ``GET /api/labels`` at load time, so changing a label here is enough —
no HTML edits required. Keyboard digits 0..N map to the codes in order.
"""

# Failure codes. Keys are the stored `failure_code` values (F0..F6); the text
# after the colon is only shown in the UI. Keep the F# keys stable once you
# start annotating, or old CSVs will disagree with new ones.
FAILURE_CODES = {
    "F0": "grasp miss (never grasped the object)",
    "F1": "grasp slip (grasped then dropped)",
    "F2": "wrong object / wrong target",
    "F3": "placement error (missed the goal region)",
    "F4": "collision / knocked something over",
    "F5": "stuck / no progress (timeout)",
    "F6": "other (see notes)",
}

# Vertical position of the init condition. Stored verbatim in `y_position`.
Y_POSITIONS = ["top", "mid", "bottom"]

# Result values stored in the `result` column.
RESULT_SUCCESS = "success"
RESULT_FAILURE = "failure"
