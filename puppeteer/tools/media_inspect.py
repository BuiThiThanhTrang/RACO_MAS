from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import yaml

from tools.base.base_tool import Tool
from tools.base.register import global_tool_registry


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv", ".avi", ".webm"}

_VISION_EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "complete": {"type": "boolean"},
        "orientation": {"type": "string"},
        "observations": {"type": "array", "items": {"type": "string"}},
        "exact_text": {"type": "array", "items": {"type": "string"}},
        "spatial_layout": {"type": "array", "items": {"type": "string"}},
        "candidate_answer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "uncertain_items": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "complete",
        "orientation",
        "observations",
        "exact_text",
        "spatial_layout",
        "candidate_answer",
        "confidence",
        "uncertain_items",
    ],
    "additionalProperties": False,
}


class VisionOutputError(RuntimeError):
    """The vision backend returned evidence that is unsafe to route onward."""


def _parse_vision_payload(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
            if cleaned.casefold().startswith("json\n"):
                cleaned = cleaned[5:].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise VisionOutputError(
            "vision_output_invalid_json: the model response was incomplete or "
            "did not follow the structured evidence contract"
        ) from error
    if not isinstance(payload, dict):
        raise VisionOutputError("vision_output_invalid_json: expected one JSON object")

    missing = [key for key in _VISION_EVIDENCE_SCHEMA["required"] if key not in payload]
    if missing:
        raise VisionOutputError(
            "vision_output_incomplete: missing fields " + ", ".join(missing)
        )
    if payload.get("complete") is not True:
        detail = "; ".join(str(item) for item in payload.get("uncertain_items") or [])
        raise VisionOutputError(
            "vision_output_incomplete: model marked the inspection incomplete"
            + (f" ({detail})" if detail else "")
        )

    list_fields = ("observations", "exact_text", "spatial_layout", "uncertain_items")
    for key in list_fields:
        if not isinstance(payload.get(key), list):
            raise VisionOutputError(f"vision_output_invalid_json: {key} must be a list")
        payload[key] = [str(item).strip() for item in payload[key] if str(item).strip()]
    if not any(payload.get(key) for key in ("observations", "exact_text", "spatial_layout")):
        raise VisionOutputError("vision_output_incomplete: no visual evidence was extracted")

    payload["orientation"] = str(payload.get("orientation") or "").strip()
    payload["candidate_answer"] = str(payload.get("candidate_answer") or "").strip()
    try:
        payload["confidence"] = min(1.0, max(0.0, float(payload.get("confidence"))))
    except (TypeError, ValueError) as error:
        raise VisionOutputError("vision_output_invalid_json: confidence must be numeric") from error
    return payload


def _format_vision_payload(payload: dict[str, Any], finish_reason: str) -> str:
    routed_payload = dict(payload)
    routed_payload["finish_reason"] = finish_reason or "unknown"
    return "Structured visual evidence:\n" + json.dumps(
        routed_payload, ensure_ascii=False, indent=2
    )

def _load_media_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "config" / "global.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as source:
            return dict((yaml.safe_load(source) or {}).get("media_tools") or {})
    except (OSError, yaml.YAMLError):
        return {}


def _timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, remaining = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining:05.2f}"
    return f"{minutes:02d}:{remaining:05.2f}"


def transcribe_audio_file(file_path: str | os.PathLike[str]) -> str:
    """Transcribe audio in an isolated process and cache the structured result.

    On Windows the main benchmark process loads PyTorch's Intel OpenMP runtime,
    while faster-whisper/CTranslate2 loads LLVM OpenMP. Loading both DLLs in one
    process aborts with OMP Error #15, so ASR must stay out of this process.
    """

    config = _load_media_config()
    asr_config = dict(config.get("audio") or {})
    model_name = str(asr_config.get("model", "small"))
    device = str(asr_config.get("device", "cpu"))
    compute_type = str(
        asr_config.get("compute_type", "int8" if device == "cpu" else "float16")
    )
    source = Path(file_path).resolve()
    if not source.is_file():
        raise RuntimeError(f"Audio file does not exist: {source}")
    file_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache_key = json.dumps(
        {
            "file": file_digest,
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "beam_size": int(asr_config.get("beam_size", 5)),
            "vad_filter": bool(asr_config.get("vad_filter", True)),
        },
        sort_keys=True,
    )

    from diskcache import Cache

    cache_path = str(asr_config.get("cache_path", "cache/asr/faster_whisper"))
    with Cache(cache_path) as cache:
        payload = cache.get(cache_key)
        if payload is None:
            command = [
                sys.executable,
                "-m",
                "tools.local_asr_worker",
                "--file",
                str(source),
                "--model",
                model_name,
                "--device",
                device,
                "--compute-type",
                compute_type,
                "--beam-size",
                str(int(asr_config.get("beam_size", 5))),
                "--vad-filter",
                "true" if bool(asr_config.get("vad_filter", True)) else "false",
            ]
            download_root = str(asr_config.get("download_root") or "").strip()
            if download_root:
                command.extend(["--download-root", download_root])
            process_options: dict[str, Any] = {
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "timeout": int(asr_config.get("timeout_seconds", 900)),
                "check": False,
            }
            if os.name == "nt":
                process_options["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run(command, **process_options)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown error").strip()
                raise RuntimeError(
                    f"isolated faster-whisper worker failed with exit code "
                    f"{completed.returncode}: {detail[-2000:]}"
                )
            output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if not output_lines:
                raise RuntimeError("isolated faster-whisper worker returned no output")
            try:
                payload = json.loads(output_lines[-1])
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "isolated faster-whisper worker returned invalid JSON: "
                    + completed.stdout[-2000:]
                ) from error
            cache.set(cache_key, payload)

    lines = []
    for segment in payload.get("segments", []):
        text = str(segment.get("text") or "").strip()
        if text:
            lines.append(
                f"[{_timestamp(segment.get('start', 0.0))} - "
                f"{_timestamp(segment.get('end', 0.0))}] {text}"
            )
    language = payload.get("language") or "unknown"
    duration = float(payload.get("duration") or 0.0)
    transcript = "\n".join(lines) if lines else "[No speech detected]"
    return (
        f"Language: {language}\n"
        f"Duration seconds: {duration:.2f}\n"
        f"Transcript:\n{transcript}"
    )


def _data_uri(extension: str, payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        content_type = "image/png"
    elif payload.startswith(b"GIF8"):
        content_type = "image/gif"
    else:
        content_type = mimetypes.guess_type("attachment" + extension)[0] or "image/jpeg"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _vision_request(images: list[tuple[str, bytes]], question: str) -> str:
    config = _load_media_config()
    vision_config = dict(config.get("vision") or {})
    model = str(vision_config.get("model", "gemini-2.5-flash"))
    max_tokens = int(vision_config.get("max_tokens", 4096))
    reasoning_effort = str(vision_config.get("reasoning_effort", "none")).strip()
    structured_output = bool(vision_config.get("structured_output", True))

    from model import global_openai_client

    prompt = (
        "Inspect every supplied image as evidence for the task below. Extract all "
        "answer-relevant visible text, counts, labels, objects, and spatial "
        "relationships. Preserve exact wording, qualifiers, ordering, and coordinates. "
        "Distinguish direct observation from inference and never guess an obscured "
        "detail. For a board, grid, table, diagram, map, or worksheet, enumerate every "
        "occupied or relevant cell using the coordinate labels visible in the image; "
        "verify orientation from both axes before assigning coordinates and check that "
        "no relevant row, column, piece, symbol, or item was omitted. Set complete=true "
        "only after performing that completeness check. Give a concise candidate_answer "
        "when the image alone supports one; otherwise use an empty string and explain "
        "the unresolved details in uncertain_items. Return only the requested JSON "
        "object, with no Markdown or surrounding prose.\n\nTask:\n" + question
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for label, payload in images:
        content.append({"type": "text", "text": label})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _data_uri(".jpg", payload)},
            }
        )
    request_options: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        request_options["reasoning_effort"] = reasoning_effort
    if structured_output:
        request_options["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "vision_evidence",
                "strict": True,
                "schema": _VISION_EVIDENCE_SCHEMA,
            },
        }
    response = global_openai_client.chat.completions.create(
        **request_options
    )
    choice = response.choices[0]
    finish_reason = str(getattr(choice, "finish_reason", "") or "").strip().casefold()
    if finish_reason in {"length", "max_tokens", "max_output_tokens", "incomplete"}:
        raise VisionOutputError(
            f"vision_output_truncated: finish_reason={finish_reason}"
        )
    payload = _parse_vision_payload(str(choice.message.content or ""))
    return _format_vision_payload(payload, finish_reason)


def _inspect_image(path: Path, question: str) -> str:
    payload = path.read_bytes()
    description = _vision_request([("Image attachment", payload)], question)
    if not description:
        raise RuntimeError("The vision backend returned an empty image analysis")
    return "Media type: image\n\n" + description


def _sample_video_frames(path: Path, sample_count: int) -> tuple[list[tuple[str, bytes]], str]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path.name}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
        if frame_count <= 0:
            indices = [0]
        elif sample_count <= 1:
            indices = [frame_count // 2]
        else:
            indices = sorted(
                {
                    round(index * (frame_count - 1) / (sample_count - 1))
                    for index in range(sample_count)
                }
            )

        frames: list[tuple[str, bytes]] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            if width > 1024:
                scale = 1024.0 / width
                frame = cv2.resize(frame, (1024, max(1, round(height * scale))))
            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if ok:
                timestamp = frame_index / fps if fps > 0 else 0.0
                frames.append(
                    (f"Video frame at {_timestamp(timestamp)}", encoded.tobytes())
                )
        metadata = (
            f"Frame count: {frame_count}\nFPS: {fps:.3f}\n"
            f"Duration seconds: {duration:.2f}"
        )
        return frames, metadata
    finally:
        capture.release()


def _inspect_video(path: Path, question: str) -> str:
    config = _load_media_config()
    video_config = dict(config.get("video") or {})
    frames, metadata = _sample_video_frames(
        path, max(1, int(video_config.get("sample_frames", 8)))
    )
    if not frames:
        raise RuntimeError("No video frames could be decoded")
    description = _vision_request(frames, question)
    sections = ["Media type: video", metadata, "Visual evidence:\n" + description]
    if bool(video_config.get("transcribe_audio", True)):
        try:
            sections.append("Audio evidence:\n" + transcribe_audio_file(path))
        except Exception as error:
            sections.append(f"Audio evidence unavailable: {type(error).__name__}: {error}")
    return "\n\n".join(sections)


@global_tool_registry("inspect_media")
class MediaInspect(Tool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            description="inspect a local image, audio, or video attachment",
            execute_function=self.execute,
        )

    def execute(self, *args, **kwargs):
        file_path = Path(str(kwargs.get("file_path") or "")).resolve()
        if not file_path.is_file():
            return False, f"Media file does not exist: {file_path}"
        extension = str(kwargs.get("file_extension") or file_path.suffix).lower()
        question = str(kwargs.get("question") or "Describe the answer-relevant evidence.")
        try:
            if extension in IMAGE_EXTENSIONS:
                return True, _inspect_image(file_path, question)
            if extension in AUDIO_EXTENSIONS:
                return True, "Media type: audio\n\n" + transcribe_audio_file(file_path)
            if extension in VIDEO_EXTENSIONS:
                return True, _inspect_video(file_path, question)
            return False, f"Unsupported media format: {extension or '[unknown]'}"
        except Exception as error:
            return False, f"Media inspection failed ({type(error).__name__}): {error}"
