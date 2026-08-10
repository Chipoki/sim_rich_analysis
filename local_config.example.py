"""
Machine-local settings, deliberately kept out of version control.

Copy this file to `local_config.py` (untracked -- see .gitignore) and fill in
the value below. `config.py` imports it and falls back to None with a warning
if the file is absent.

`NO_LETTERS_MANUAL_EXCLUDE_PARTICIPANT_ID` is the Prolific id of the No-Letters
participant excluded on the technical-problems criterion documented at length in
config.py. It lives here rather than in config.py because a Prolific id is a
stable participant identifier, and publishing one next to the finding that this
person's data was unusable is participant-level information.

Set it to None to analyze the No-Letters condition at N=20 instead of the
reported N=19.
"""

NO_LETTERS_MANUAL_EXCLUDE_PARTICIPANT_ID = None
