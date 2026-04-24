'''
llm tools for autoresearch
'''
import os
import subprocess
import sys
from pathlib import Path
from litellm import completion

# Base paths
PROJECT_ROOT = Path(os.path.abspath(__file__)).resolve().parent.parent.parent
ANDREW_EMB_DIR = PROJECT_ROOT / "andrew-embedding"
AUTORESEARCH_DIR = ANDREW_EMB_DIR / "autoresearch"
EXPERIMENTS_FILE = AUTORESEARCH_DIR / "EXPERIMENTS.md"

def read_file(path: str) -> str:
    """Reads a file from the given path without restrictions."""
    try:
        target_path = Path(path).resolve()
        with open(target_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: str, content: str) -> str:
    """Writes content to a file. Strictly limited to train.py in andrew-embedding."""
    target_path = Path(path).resolve()
    allowed_paths = [(ANDREW_EMB_DIR / "train.py").resolve(), (AUTORESEARCH_DIR / "EXPERIMENTS.md").resolve()]
    
    if target_path not in allowed_paths:
        return f"Error: write_file is sandboxed. You are only allowed to modify {allowed_paths}. Refusing to write to {target_path}."
    
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: {target_path.name} has been updated."
    except Exception as e:
        return f"Error writing file: {e}"

def _run_and_stream(cmd, cwd):
    output_lines = []
    process = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='replace'
    )
    for line in process.stdout:
        try:
            sys.stdout.write(line)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or 'ascii'
            sys.stdout.write(line.encode(enc, errors='replace').decode(enc))
        sys.stdout.flush()
        output_lines.append(line)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, output="".join(output_lines))
    return "".join(output_lines)

def run_experiment(entrypoint: str = "") -> str:
    """Executes train.py, then embed.py, then eval.py, and returns the aggregated output."""
    output = []
    
    # 1. Run train.py
    try:
        print("\n=== Running train.py ===")
        output.append("=== Running train.py ===")
        out_str = _run_and_stream(
            [sys.executable, "train.py", "--autoresearch", "True"], 
            cwd=str(ANDREW_EMB_DIR)
        )
        output.append(out_str)
    except subprocess.CalledProcessError as e:
        return f"Error running train.py:\n{e.output}"

    # 2. Run embed.py
    try:
        print("\n=== Running embed.py ===")
        output.append("\n=== Running embed.py ===")
        out_str = _run_and_stream(
            [sys.executable, "embed.py", "--autoresearch", "True"], 
            cwd=str(ANDREW_EMB_DIR)
        )
        output.append(out_str)
    except subprocess.CalledProcessError as e:
        return f"Error running embed.py:\n{e.output}"

    # 3. Run eval.py
    try:
        print("\n=== Running eval.py ===")
        output.append("\n=== Running eval.py ===")
        eval_script = str(PROJECT_ROOT / "evaluation layer" / "v2" / "eval.py")
        out_str = _run_and_stream(
            [sys.executable, eval_script, "--autoresearch", "True"], 
            cwd=str(PROJECT_ROOT)
        )
        output.append(out_str)
    except subprocess.CalledProcessError as e:
        return f"Error running eval.py:\n{e.output}"

    return "\n".join(output)

def create_report(results: str, next_steps: str) -> str:
    """Prompts the model for improvements on this iteration and dumps the results into EXPERIMENTS.md"""
    # Read past experiments for context
    past_experiments = ""
    if EXPERIMENTS_FILE.exists():
        with open(EXPERIMENTS_FILE, "r", encoding="utf-8") as f:
            past_experiments = f.read()

    system_prompt = (
        "You are an autoresearch AI agent. You have just run an experiment on a "
        "graph neural network embedding model. Analyze the following results "
        "from the experiment (train, embed, eval) and the defined next steps. "
        "Generate a concise markdown block detailing the changes made, "
        "the final evaluation metrics, and any insights for improvements in the next iteration."
    )
    
    user_prompt = f"Results:\n{results}\n\nNext Steps:\n{next_steps}\n\nFormulate the update for EXPERIMENTS.md."
    
    model = os.getenv("LITELLM_MODEL", "gpt-4o")
    
    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        report = response.choices[0].message.content
        
        # Dump to EXPERIMENTS.md
        with open(EXPERIMENTS_FILE, "a", encoding="utf-8") as f:
            # Enforce some spacing before appending
            f.write("\n\n" + report.strip() + "\n")
            
        return "Report successfully generated and appended to EXPERIMENTS.md"
    except Exception as e:
        return f"Error generating report: {e}"


if __name__ == "__main__":
    print(PROJECT_ROOT)
    print(ANDREW_EMB_DIR)
    print(AUTORESEARCH_DIR)
    print(EXPERIMENTS_FILE)