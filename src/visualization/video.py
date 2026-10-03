"""Conservative file-only video composition; inputs must already be aligned."""
import json
from pathlib import Path
import subprocess


def probe(path: Path, ffprobe: str = "ffprobe") -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    p = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def compose(inputs: list[Path], output: Path, *, height: int = 360, fps: float = 25, ffmpeg: str = "ffmpeg") -> None:
    """Muted H.264 review, equal padded tiles. Never overwrite source media."""
    if len(inputs) < 2 or height < 32 or height % 2 or not 1 <= fps <= 120:
        raise ValueError("Need >=2 videos, an even height >=32, and fps in [1,120]")
    if output.suffix.lower() != ".mp4" or output.exists() or output.resolve() in {p.resolve() for p in inputs}:
        raise ValueError("Output must be a new MP4, distinct from all inputs")
    for path in inputs:
        if not any(s["codec_type"] == "video" for s in probe(path)["streams"]):
            raise ValueError(f"No video stream: {path.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-nostdin", "-n", "-v", "error", "-filter_complex_threads", "1"]
    for path in inputs:
        cmd += ["-threads", "2", "-i", str(path)]
    filters = [f"[{i}:v]setpts=PTS-STARTPTS,fps={fps},scale={height}:{height}:force_original_aspect_ratio=decrease,pad={height}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]" for i in range(len(inputs))]
    filters.append("".join(f"[v{i}]" for i in range(len(inputs))) + f"hstack=inputs={len(inputs)}:shortest=1[v]")
    cmd += ["-filter_complex", ";".join(filters), "-map", "[v]", "-an", "-map_metadata", "-1", "-c:v", "libx264", "-threads", "2", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    subprocess.run(cmd, check=True)
