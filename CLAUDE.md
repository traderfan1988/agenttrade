# Arbeitsregeln

- `python -m pytest` muss vor jedem Commit grün sein.
- Jeder gefundene Fehler wird ZUERST ein Test, dann behoben.
- Kein Agent gibt eine nackte Zahl zurück, nur `Befund` mit `Label`.
- Fehlende Daten → UNBESTIMMT, niemals PASSIERT.
- `bestanden=None` heißt "nicht bewertbar", nicht "nein".
- Schwellen gehören ins `SCHWELLEN`-Dict oben in der Datei, nie verstreut.
- Vergleichsbasis bei Zerlegung: Drawdown gegen Drawdown, nie Rendite im
  selben Fenster (Fenster ist am 52W-Hoch der Aktie verankert → Bias).
- Nach Änderungen an der Scoring-Logik: Commit-Hash wandert über
  kern/version.py ins Gedächtnis, sonst wird --review unbrauchbar.
