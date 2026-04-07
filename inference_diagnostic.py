# Copyright (C) 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import json
import time
import torch
import sys
from pathlib import Path
from typing import Tuple, List
from string import Template

import soundfile as sf
import librosa

# Assuming the backbone is placed in the models/ directory
from models.mimo_audio.mimo_audio import MimoAudio

# ==========================================
# Note for Reviewers:
# Pre-trained checkpoints for TTS-PRISM will be publicly released on HuggingFace 
# upon paper acceptance. For local reproduction during the review process, 
# please configure the dummy paths below to point to your local model directories.
# ==========================================

# ====== Path Configuration ======
# Placeholder paths for blind review. Will be updated to HuggingFace repo ID after acceptance.
MODEL_PATH = "./checkpoints/TTS-PRISM-7B" 
TOKENIZER_PATH = "./checkpoints/MiMo-Audio-Tokenizer"

# SCP_FILE Format Explanation:
# The .scp file contains the list of audio files and their corresponding reference texts.
# Each line should be formatted as: `<audio_path>\t<reference_text>` (separated by a Tab).
# Example: 
# /path/to/audio1.wav    This is the reference text for audio one.
# /path/to/audio2.wav    This is another text.
SCP_FILE = "./data/test_1.6k.scp"
OUTPUT_DIR = "./results/tts_prism_eval"
LOG_FILE_NAME = "inference_log.txt"

# ====== Hyperparameters ======
TARGET_SR = 24000
MAX_RETRIES = 10
SEED = 1234

# ====== Prompts ======
# Note: The prompt remains in Mandarin as the model is designed to evaluate Mandarin TTS,
# and the schema-driven instruction tuning was conducted in Mandarin.
OVERALL_PROMPT_TPL = Template("""
请根据参考文本和音频，对语音进行多维度客观评测，并按照 JSON 格式输出各维度的打分理由和分数。

参考文本：$reference_text
""")

# ====== Logging Utility ======
class DualLogger:
    """
    Redirects standard output to both the terminal and a log file simultaneously.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure real-time writing

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ====== Basic Utilities ======
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def read_scp_file(scp_path: str) -> List[Tuple[Path, str]]:
    """
    Reads the .scp file containing audio paths and reference texts.
    """
    entries = []
    with open(scp_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line, maxsplit=1)
            if len(parts) < 2:
                print(f"WARNING: Invalid format at line {line_num} in SCP file.")
                continue
            audio_path, reference_text = parts
            audio_path = Path(audio_path)
            if not audio_path.exists():
                print(f"WARNING: Audio file not found: {audio_path}")
                continue
            entries.append((audio_path, reference_text))
    return entries

def call_model(model: MimoAudio, audio_path: Path, prompt: str) -> str:
    try:
        return str(
            model.audio_understanding_sft(
                input_speech=str(audio_path),
                input_text=prompt,
                thinking=False
            )
        )
    except Exception as e:
        return f"ERROR: {e}"

# ====== Audio Resampling ======
def resample_audio(audio_path: Path) -> Tuple[Path | None, float | None]:
    """
    Resamples the input audio to the target sample rate (24kHz) required by the tokenizer.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        if sr == TARGET_SR:
            return audio_path, librosa.get_duration(y=audio, sr=sr)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        out = audio_path.with_name(audio_path.stem + "_resampled.wav")
        sf.write(out, audio, TARGET_SR)
        return out, librosa.get_duration(y=audio, sr=TARGET_SR)
    except Exception as e:
        print(f"Resampling failed for {audio_path}: {e}")
        return None, None

# ====== JSON Extraction =====
def extract_first_complete_json(text: str) -> dict | None:
    if not text:
        return None

    s = text.strip()
    start = s.find("{")
    if start < 0:
        return None

    in_str = False
    esc = False
    depth = 0

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None

def remove_duplicate_fields(data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    out = {}
    for k, v in data.items():
        if k not in out:
            out[k] = remove_duplicate_fields(v)
    return out

# ====== Main Pipeline ======
def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.manual_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    # Setup log redirection
    log_path = Path(OUTPUT_DIR) / LOG_FILE_NAME
    sys.stdout = DualLogger(log_path)
    print(f"Logs will be saved to: {log_path}")

    print("Initializing TTS-PRISM model...")
    # NOTE: Model weights required. Ensure paths are correctly configured.
    try:
        model = MimoAudio(
            model_path=MODEL_PATH,
            mimo_audio_tokenizer_path=TOKENIZER_PATH
        )
    except Exception as e:
        print(f"Failed to initialize model. Please check the checkpoint paths. Error: {e}")
        return

    audio_entries = read_scp_file(SCP_FILE)
    if not audio_entries:
        print("❌ No valid audio entries found in the SCP file.")
        return

    total_time = 0.0
    processed_count = 0  

    for idx, (audio_path, reference_text) in enumerate(audio_entries, 1):
        orig_audio_path = audio_path
        
        # Check if result already exists (Resume capability)
        expected_json_path = Path(OUTPUT_DIR) / f"{orig_audio_path.stem}.json"
        
        if expected_json_path.exists():
            print(f"[{idx}/{len(audio_entries)}] ⏩ Skipping existing file: {orig_audio_path.stem}")
            continue

        t0 = time.time()
        print(f"\n[{idx}/{len(audio_entries)}] Processing: {audio_path.name}")
        
        resampled_file = None 

        audio_path, dur = resample_audio(audio_path)
        if audio_path is None:
            continue
        
        if audio_path != orig_audio_path:
            resampled_file = audio_path 

        prompt = OVERALL_PROMPT_TPL.substitute(reference_text=reference_text)

        parsed = None
        raw = ""

        # Retry mechanism for robust JSON extraction
        for attempt in range(1, MAX_RETRIES + 1):
            raw = call_model(model, audio_path, prompt)
            parsed = extract_first_complete_json(raw)
            if parsed is not None:
                break
            print(f"  Attempt {attempt}: Failed to extract complete JSON")

        result = {
            "file": str(orig_audio_path),
            "reference_text": reference_text,
            "dimensions": {}
        }

        if parsed is not None:
            dims = parsed.get("dimensions", parsed)
            result["dimensions"] = remove_duplicate_fields(dims)
        else:
            result["dimensions"] = {
                "error": "Model failed to output parseable JSON after maximum retries.",
                "raw_response": raw[:2000]
            }

        # Save results
        with open(expected_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Clean up temporary resampled files
        if resampled_file and resampled_file.exists():
            try:
                os.remove(resampled_file)
                print(f"  🗑️ Cleaned up temporary resampled file: {resampled_file.name}")
            except Exception as e:
                print(f"  ⚠️ Failed to clean up resampled file: {resampled_file.name} - {str(e)[:50]}")

        cost = time.time() - t0
        total_time += cost
        processed_count += 1
        print(f"  ✅ Saved: {expected_json_path.name} ({cost:.2f}s)")

    print("\n===== Evaluation Statistics =====")
    print(f"Total samples: {len(audio_entries)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already exists): {len(audio_entries) - processed_count}")
    if processed_count > 0:
        print(f"Total inference time: {total_time:.1f}s")
        print(f"Average time per sample: {total_time / processed_count:.2f}s")

if __name__ == "__main__":
    main()