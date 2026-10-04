from pathlib import Path

def candidate_checkpoints(directory): return sorted(Path(directory).glob("*.zip"))
