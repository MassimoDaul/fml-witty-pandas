"""
Autoresearch orchestrator loop.
"""
import os
import json
from litellm import completion
from tools import read_file, write_file, run_experiment, create_report, ANDREW_EMB_DIR

def execute_tool_call(tool_call):
    function_name = tool_call.function.name
    
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return f"Error: Invalid JSON arguments for {function_name}"
        
    if function_name == "read_file":
        return read_file(**arguments)
    elif function_name == "write_file":
        return write_file(**arguments)
    elif function_name == "run_experiment":
        return run_experiment(**arguments)
    elif function_name == "create_report":
        return create_report(**arguments)
    else:
        return f"Error: Tool {function_name} not found."

# Define explicit tool schemas for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the current contents of a file. Use this to understand train.py before modifying it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Writes the FULL content of a script. Restricted to 'train.py' in andrew-embedding. "
                "CRITICAL WARNING: Do NOT abuse this tool. You must ONLY call this after carefully thinking through your experiment. "
                "You MUST write the complete, valid, executable python code in one go. "
                "DO NOT use placeholders like '# ... existing code ...'. DO NOT omit any imports or logic. "
                "If you write incomplete code, the pipeline will break."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path. Valid values: e.g. path/to/train.py"},
                    "content": {"type": "string", "description": "The complete, fully-functioning python source code"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_experiment",
            "description": "Executes train.py, embed.py, and eval.py sequentially. Call this ONLY AFTER calling write_file to evaluate your newly written experiment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entrypoint": {"type": "string", "description": "Optional script name"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_report",
            "description": "Generates a final assessment report comparing your eval results with previous best. Use this as the FINAL step of the iteration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {"type": "string", "description": "Raw string output from run_experiment"},
                    "next_steps": {"type": "string", "description": "Hypothesis for the NEXT iteration based on these results"}
                },
                "required": ["results", "next_steps"]
            }
        }
    }
]

system_prompt = f"""You are an autonomous AI research scientist optimizing a graph neural network embedding model.
Your workflow for EACH iteration is strict:
1. Examine the current code using `read_file` (typically `{str(ANDREW_EMB_DIR / 'train.py')}`).
2. Formulate a hypothesis for improving the composite evaluation score.
3. Use `write_file` exactly ONCE to overwrite `train.py` with your FULL, COMPLETE code. Avoid minor formatting edits; focus on real ML hyperparameters (e.g. learning rate, GNN layers, negative sampling, contrastive margins). We consider `write_file` heavily rate-limited, so use it wisely.
4. Use `run_experiment` to evaluate your change.
5. Use `create_report` to document the results, which will terminate your iteration.

Remember that you are strictly sandboxed. Repeatedly attempting to write files outside the sandbox or using placeholders in your code will result in immediate termination."""

def main():
    # Example starting loop
    model = os.getenv("LITELLM_MODEL", "gpt-4o")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please start a new experiment loop to improve the model composite score. First, inspect train.py to devise an experiment, then run the cycle."}
    ]
    
    print("Starting autoresearch loop...")
    MAX_STEPS = 10
    
    for step in range(MAX_STEPS):
        print(f"\n--- Step {step+1}/{MAX_STEPS} ---")
        try:
            response = completion(model=model, messages=messages, tools=tools)
        except Exception as e:
            print(f"LLM API Error: {e}")
            break
            
        message = response.choices[0].message
        messages.append(message)
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                print(f"[LLM] Called Tool: {func_name}")
                
                # Execute tool
                result = execute_tool_call(tool_call)
                
                # Log outcome appropriately
                if func_name == "write_file":
                    print(f"[Tool] write_file execution complete. ({len(result)} chars log)")
                elif func_name == "read_file":
                    print(f"[Tool] read_file read {len(str(result))} characters.")
                else:
                    print(f"[Tool] Output: {str(result)[:200]}...")
                    
                # Append tool result back to message history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": str(result)
                })
                
                # If report is created, loop iteration is finished
                if func_name == "create_report":
                    print("\nAutoresearch iteration complete.")
                    return
        else:
            print(f"[LLM] Content: {message.content}")
            if "complete" in (message.content or "").lower():
                break

if __name__ == "__main__":
    main()
