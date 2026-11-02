#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize.py RESULT.json")
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    errors = [record for record in records if record.get("error")]
    print(json.dumps({
        "status": data.get("status"),
        "decision": data.get("decision"),
        "records": len(records),
        "errors": len(errors),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

