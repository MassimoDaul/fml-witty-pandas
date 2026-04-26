'''
llm tools for autoresearch
'''
import os
import subprocess
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
    allowed_path = (ANDREW_EMB_DIR / "train.py").resolve()
    
    if target_path != allowed_path:
        return f"Error: write_file is sandboxed. You are only allowed to modify {allowed_path}. Refusing to write to {target_path}."
    
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: {target_path.name} has been updated."
    except Exception as e:
        return f"Error writing file: {e}"

def run_experiment(entrypoint: str = "") -> str:
    """Executes train.py, then embed.py, then eval.py, and returns the aggregated output."""
    output = []
    
    # 1. Run train.py
    try:
        output.append("=== Running train.py ===")
        res_train = subprocess.run(
            ["python", "train.py"], 
            cwd=str(ANDREW_EMB_DIR), 
            capture_output=True, text=True, check=True
        )
        output.append(res_train.stdout)
    except subprocess.CalledProcessError as e:
        return f"Error running train.py:\n{e.stdout}\n{e.stderr}"

    # 2. Run embed.py
    try:
        output.append("\n=== Running embed.py ===")
        res_embed = subprocess.run(
            ["python", "embed.py"], 
            cwd=str(ANDREW_EMB_DIR), 
            capture_output=True, text=True, check=True
        )
        output.append(res_embed.stdout)
    except subprocess.CalledProcessError as e:
        return f"Error running embed.py:\n{e.stdout}\n{e.stderr}"

    # 3. Run eval.py
    try:
        output.append("\n=== Running eval.py ===")
        eval_script = str(PROJECT_ROOT / "evaluation" / "eval.py")
        res_eval = subprocess.run(
            ["python", eval_script], 
            cwd=str(PROJECT_ROOT), 
            capture_output=True, text=True, check=True
        )
        output.append(res_eval.stdout)
    except subprocess.CalledProcessError as e:
        return f"Error running eval.py:\n{e.stdout}\n{e.stderr}"

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
