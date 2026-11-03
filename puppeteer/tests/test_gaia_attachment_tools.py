import os
import tempfile
import unittest
import zipfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openpyxl import Workbook

from tools.code_interpreter import PythonInterpreter
from tools.file_read import FileRead
from tools.media_inspect import MediaInspect
from tools.spreadsheet_inspect import SpreadsheetInspect


class GaiaAttachmentToolTests(unittest.TestCase):
    def test_spreadsheet_inspector_preserves_formula_and_fill(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "styled.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Evidence"
            sheet["A1"] = 2
            sheet["B1"] = 3
            sheet["C1"] = "=A1+B1"
            sheet["A2"] = "START"
            sheet["A2"].fill.fgColor.rgb = "FF00FF00"
            workbook.save(path)

            ok, output = SpreadsheetInspect("spreadsheet-test").execute(
                file_path=str(path), file_extension=".xlsx"
            )

        self.assertTrue(ok)
        self.assertIn("## Sheet: Evidence", output)
        self.assertIn("C1='=A1+B1'", output)
        self.assertIn("A2='START'", output)
        self.assertIn("fill=FF00FF00", output)

    def test_media_inspector_routes_audio_to_local_transcription(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice.mp3"
            path.write_bytes(b"not-real-audio")
            with patch(
                "tools.media_inspect.transcribe_audio_file",
                return_value="Language: en\nTranscript:\n[00:00 - 00:01] evidence",
            ):
                ok, output = MediaInspect("media-test").execute(
                    file_path=str(path), file_extension=".mp3", question="What was said?"
                )

        self.assertTrue(ok)
        self.assertIn("Media type: audio", output)
        self.assertIn("evidence", output)

    def test_image_inspector_returns_complete_structured_evidence(self):
        payload = {
            "complete": True,
            "orientation": "files h-a left-to-right; ranks 1-8 top-to-bottom",
            "observations": ["White king is on g1", "Black rook is on d8"],
            "exact_text": [],
            "spatial_layout": ["g1: white king", "d8: black rook"],
            "candidate_answer": "Rd5",
            "confidence": 0.91,
            "uncertain_items": [],
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps(payload)),
            )]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "board.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            with patch("model.global_openai_client") as client:
                client.chat.completions.create.return_value = response
                ok, output = MediaInspect("media-test").execute(
                    file_path=str(path), file_extension=".png", question="Best move?"
                )

        self.assertTrue(ok)
        self.assertIn('"complete": true', output)
        self.assertIn('"candidate_answer": "Rd5"', output)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertEqual(request["max_tokens"], 8192)
        self.assertEqual(request["response_format"]["type"], "json_schema")

    def test_image_inspector_rejects_truncated_model_output(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content='{"complete": true'),
            )]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "board.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            with patch("model.global_openai_client") as client:
                client.chat.completions.create.return_value = response
                ok, output = MediaInspect("media-test").execute(
                    file_path=str(path), file_extension=".png", question="Best move?"
                )

        self.assertFalse(ok)
        self.assertIn("vision_output_truncated", output)

    def test_read_file_safely_expands_zip_members(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bundle.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("notes.txt", "decisive evidence")
            ok, output = FileRead("file-test").execute(
                file_path=str(path), file_extension=".zip"
            )

        self.assertTrue(ok)
        self.assertIn("Archive member: notes.txt", output)
        self.assertIn("decisive evidence", output)

    def test_python_tool_exposes_absolute_attachment_environment_variable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            attachment = root / "input.txt"
            attachment.write_text("attachment evidence", encoding="utf-8")
            work = root / "work"
            work.mkdir()
            code = (
                "import os\n"
                "path = os.environ['GAIA_ATTACHMENT']\n"
                "print(os.path.isabs(path))\n"
                "print(open(path, encoding='utf-8').read())\n"
            )
            ok, output = PythonInterpreter("python-test").execute(
                work_path=str(work),
                code=code,
                file_path=str(attachment),
                timeout_detected=True,
            )

        self.assertTrue(ok)
        self.assertIn("True", output)
        self.assertIn("attachment evidence", output)


if __name__ == "__main__":
    unittest.main()
