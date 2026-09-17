import subprocess
import json
import time
import torch
import numpy as np
import re
import os
import signal
import math

from model import query_gpt
from model.query_manager import query_manager
from model.embedding import OpenAIEmbedding
from utils.file_utils import read_code, read_text

FLOAT_TOLERANCE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

class BenchmarkEvaluator:
    @staticmethod
    def commongen_coverage(concepts, text_path):
        generated_text = read_text(text_path).lower()
        concepts = [concept.lower() for concept in concepts]
        if not concepts:
            return 0.0
        matched = sum(
            bool(re.search(rf"\b{re.escape(concept)}\b", generated_text, re.IGNORECASE))
            for concept in concepts
        )
        return matched / len(concepts)

    @staticmethod
    def commongen_gpt_score(concepts, text_path, judge_model="gemini-3.5-flash"):
        generated_text = read_text(text_path)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are the strict StoryMaster used by the Puppeteer CommonGen "
                    "evaluation. Score three dimensions as integers 1-4. "
                    "Grammar and Fluency: 1 is grammatically sound but stylistically "
                    "plain; 2 has strong grammar and coherent flow; 3 has refined, "
                    "engaging sentence craft; 4 is clear, elegant, creative, and "
                    "linguistically masterful. Context Relevance: 1 connects the "
                    "required elements only basically; 2 connects them clearly but "
                    "without depth; 3 interweaves them coherently with developed "
                    "content; 4 integrates every element profoundly and immersively. "
                    "Logic Consistency: 1 is structured with possible minor lapses; "
                    "2 is generally logical with clear progression; 3 is strongly "
                    "consistent and plausible; 4 has impeccable causal and internal "
                    "consistency. Return only JSON with integer keys grammar, "
                    "relevance, consistency."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Required concepts: {concepts}\nStory:\n{generated_text}\n"
                    "Return: {\"grammar\":1,\"relevance\":1,\"consistency\":1}"
                ),
            },
        ]
        response_text, _ = query_manager.query(judge_model, prompt)
        cleaned = response_text.strip()
        if cleaned.startswith(chr(96) * 3):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1])
        try:
            payload = json.loads(cleaned)
            raw_scores = [
                int(payload["grammar"]),
                int(payload["relevance"]),
                int(payload["consistency"]),
            ]
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            raw_scores = [int(item) for item in re.findall(r"\b[1-4]\b", cleaned)[:3]]
        raw_scores = (raw_scores + [0, 0, 0])[:3]
        return [score / 4.0 for score in raw_scores]

    @staticmethod
    def check_commongen(concepts, text_path):
        coverage = torch.tensor(
            BenchmarkEvaluator.commongen_coverage(concepts, text_path),
            dtype=torch.float32,
            device=DEVICE,
        )
        scores = BenchmarkEvaluator.commongen_gpt_score(concepts, text_path)
        grammar, relevance, consistency = [
            torch.tensor(score, dtype=torch.float32, device=DEVICE) for score in scores
        ]
        metrics = {
            "grammar": grammar,
            "relevance": relevance,
            "consistency": consistency,
            "coverage": coverage,
        }
        mean_score = torch.tensor(sum(scores) / 3, dtype=torch.float32, device=DEVICE)
        return coverage * mean_score, metrics

    @staticmethod
    def commongen_binary_success(metrics):
        return (
            float(metrics["coverage"]) >= 1.0
            and float(metrics["grammar"]) >= 0.75
            and float(metrics["relevance"]) >= 0.75
            and float(metrics["consistency"]) >= 0.75
        )

    @staticmethod
    def srdd_binary_success(metrics):
        return (
            float(metrics["executability"]) >= 1.0
            and float(metrics["completeness"]) >= 1.0
            and float(metrics["consistency"]) >= 0.70
        )

    @staticmethod
    def metrics_to_json(metrics):
        return {
            key: float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in metrics.items()
        }

    @staticmethod
    def check_srdd(code_path, text):
        # Metric implementation inspired by ChatDev project:
        # https://github.com/OpenBMB/ChatDev
        path = code_path
        code = read_code(path)
        consistency = BenchmarkEvaluator.srdd_consistency(text, code)
        completeness = BenchmarkEvaluator.srdd_completeness(code)
        executability, _ = BenchmarkEvaluator.srdd_executability(path)
        executability = 1 if executability else 0
        executability = torch.tensor(executability, dtype=torch.float32, device=DEVICE)  
        consistency = torch.tensor(consistency, dtype=torch.float32, device=DEVICE)  
        completeness = torch.tensor(completeness, dtype=torch.float32, device=DEVICE)  
        metrics = {"consistency": consistency, "completeness": completeness, "executability": executability}
        if executability:
            alignment = consistency * completeness
            return alignment, metrics
        else:
            return -1.0, metrics
    
    @staticmethod
    def srdd_consistency(text, code):
        code = BenchmarkEvaluator.remove_comments(code)
        text = re.sub(r'^[^\n]*\n', '', text)
        text_embedding = OpenAIEmbedding.get_embedding(text)
        code_embedding = OpenAIEmbedding.get_embedding(code)
        similarity = BenchmarkEvaluator.get_cosine_similarity(text_embedding, code_embedding)
        return similarity

    @staticmethod
    def srdd_completeness(code):
        lines = code.split("\n")
        lines = [line for line in lines if
                "password" not in line.lower() and "passenger" not in line.lower() and "passed" not in line.lower() and "passes" not in line.lower()]
        lines = [line for line in lines if "pass" in line.lower() or "todo" in line.lower()]
        if len(lines) > 0:
            return 0.0
        return 1.0

    @staticmethod 
    def srdd_executability(work_path):
        def robust_kill(process):
            """Robustly kill the process based on the OS."""
            if process.poll() is None:  # Check if the process is still running
                if os.name == 'nt':  # For Windows
                    os.kill(process.pid, signal.SIGTERM)
                    time.sleep(1)  
                    if process.poll() is None:  
                        os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                else:  # For Linux/macOS
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)  
                    time.sleep(1)  
                    if process.poll() is None:  
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        try:
            if not os.path.exists(work_path):
                return False, "The file path does not exist."
            if os.name == 'nt':  
                command = f" python {work_path}"
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:  
                command = f"python3 {work_path}"
                process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)

            try:
                out, err = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                robust_kill(process)
                return True, "The process completes without encountering any errors."

            return_code = process.returncode
            output = out.decode('utf-8', errors='ignore')
            error_output = err.decode('utf-8', errors='ignore')

            # If the process is still running after the timeout
            if process.poll() is None:
                robust_kill(process)  
            return_code = process.returncode

            # Handle return code and output
            if return_code == 0:
                # Clean up file paths in the output for readability
                work_path = os.getcwd()
                output = output.replace(work_path, "")
                return True, output
            else:
                # Handle errors in the output
                if error_output:
                    work_path = os.getcwd()
                    if "Traceback".lower() in error_output.lower():
                        errs = error_output.replace(work_path + "/", "").replace(work_path, "")
                        return False, errs
                return False, error_output

        except subprocess.CalledProcessError as e:
            return False, f"CalledProcessError: {str(e)}"
        except Exception as ex:
            return False, f"An unexpected error occurred: {str(ex)}"


    @staticmethod
    def get_cosine_similarity(embeddingi, embeddingj):
        embeddingi = np.array(embeddingi)
        embeddingj = np.array(embeddingj).T
        cos_sim = embeddingi.dot(embeddingj) / (np.linalg.norm(embeddingi) * np.linalg.norm(embeddingj))
        return cos_sim
    
    @staticmethod
    def remove_comments(string):
        def remove_comments_by_regex(string, regex):
            lines = string.split("\n")
            lines = [line for line in lines if not line.strip().startswith("#")]
            string = "\n".join(lines)
            comments = []
            matches = re.finditer(regex, string, re.DOTALL)
            for match in matches:
                group1 = match.group(1)
                comments.append(group1)
            for comment in comments + ["''''''\n"]:
                string = string.replace(comment, "")
            return string

        string = remove_comments_by_regex(string, r"'''(.*?)'''")
        string = remove_comments_by_regex(string, r"\"\"\"(.*?)\"\"\"")
        return string

    
    @staticmethod
    def check_mmlu(final_ans, true_ans):
        if not final_ans or true_ans is None:
            return False
        prediction = BenchmarkEvaluator.extract_choice_answer(final_ans)
        gold = BenchmarkEvaluator.extract_letter(true_ans.strip()).upper()
        return prediction.upper() == gold

    @staticmethod
    def check_gsm8k(final_ans, true_ans):
        if final_ans is None or true_ans is None:   
            return False
        if isinstance(final_ans, str):
            final_num = BenchmarkEvaluator.extract_number(final_ans)
            if final_num is None:
                return False
        else:
            final_num = float(final_ans)
        true_num = float(true_ans)
        
        if not (math.isfinite(final_num) and math.isfinite(true_num)):
            return False  

        # Accuracy computation adapted from: https://github.com/reasoning-machines/pal/blob/main/scripts/gsm_eval.py
        is_correct = abs(float(final_num) - float(true_num)) < FLOAT_TOLERANCE 
        if not is_correct:
            is_correct = (round(float(final_num)) == round(float(true_num)))
            if is_correct:
                 return is_correct
            if abs(int(float(final_num))) > 100 and abs(int(float(true_num))) > 100:
                is_correct = (int(float(final_num)) == int(float(true_num)))
        return is_correct
    
    @staticmethod
    def extract_math_answer(text):
        if text is None:
            return text
        if isinstance(text, str):
            final_num = BenchmarkEvaluator.extract_number(text)
        else:
            final_num = float(text)
        return final_num
    
    @staticmethod
    def extract_choice_answer(text):
        if text is None:
            return text
        # Compare positions across supported formats, rather than returning
        # the first match from the first matching pattern.
        matches = list(re.finditer(
            r"(?i:\bis)\s+([A-Z])\b|\(([A-Za-z])\)", text
        ))
        if matches:
            match = matches[-1]
            return (match.group(1) or match.group(2)).upper()
        return text.strip()

    @staticmethod
    def normalize_string(s):
        return ''.join(s.split()).lower()

    @staticmethod
    def extract_number(text):
        matches = re.findall(r'-?\d+\.\d+|-?\d+', text)
        return float(matches[0]) if matches else None

    @staticmethod
    def extract_ground_truth(text):
        return text.split('####')[-1].strip()
    
    @staticmethod
    def extract_letter(text):
            pattern = r'\((\w)\)'
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()  
            return text.strip()  