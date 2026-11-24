"""Download the official tiny model and exercise real CPU inference."""
import sys
from pathlib import Path

from transcribe import transcribe

result, paths = transcribe(Path(sys.argv[1]), Path(sys.argv[2]), model="tiny", device="cpu")
text = paths["txt"].read_text(encoding="utf-8").casefold()
assert "brown fox" in text and "lazy dog" in text, text
assert result["complete"] and result["processed_seconds"] >= 3
assert any(segment["words"] for segment in result["segments"])
assert all(path.is_file() for path in paths.values())
