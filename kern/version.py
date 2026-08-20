"""
Scoring-Versions-Stempel.

Nach jeder Änderung an der Scoring-Logik:
    git rev-parse --short HEAD
und den Hash in SCORING_COMMIT eintragen.
Ohne diesen Stempel ist --review unbrauchbar (Scores mischen Logiken).
"""
SCORING_COMMIT = "b2c4197"
SCORING_VERSION = "0.2.0"


def versionstempel() -> str:
    return f"v{SCORING_VERSION}@{SCORING_COMMIT}"
