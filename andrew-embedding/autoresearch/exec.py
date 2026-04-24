"""
Autoresearch orchestrator loop.
"""
import os
import sys
import json
import time
from litellm import completion
from tools import read_file, write_file, run_experiment, create_report, ANDREW_EMB_DIR, PROJECT_ROOT
from dotenv import load_dotenv

_ = load_dotenv(PROJECT_ROOT / ".env")

model = os.getenv("LITELLM_MODEL")
api_key = os.getenv("LITELLM_API_KEY")

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
            "description": "Executes train.py, embed.py, and eval.py sequentially. Call this to evaluate your changes.",
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
            "description": "Generates a final assessment report comparing your eval results with previous best. Use this to conclude an iteration.",
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

MAX_STEPS = 20
system_prompt = f"""You are an autonomous AI research scientist optimizing a graph neural network embedding model.
You have flexibility to explore, read files, write files (limited to andrew-embedding/train.py and andrew-embedding/autoresearch/EXPERIMENTS.md), and run experiments as needed to improve the composite evaluation score.

Your workflow involves:
1. Examining the current code using `read_file` (e.g., `{str(ANDREW_EMB_DIR / 'train.py')}`).
2. Formulating a hypothesis for improving the metrics.
3. Modifying code using `write_file` (ensure complete code without placeholders).
4. Evaluating changes using `run_experiment`.
5. Documenting findings using `create_report`, which concludes the iteration.

You have a budget of {MAX_STEPS} tool calls to complete your objectives. Once you have finalized your experiment and run the evaluations, invoke `create_report`."""

def main():
    from tools import EXPERIMENTS_FILE
    past_experiments = ""
    if EXPERIMENTS_FILE.exists():
        with open(EXPERIMENTS_FILE, "r", encoding="utf-8") as f:
            past_experiments = f.read().strip()
            
    sys_prompt = system_prompt
    if past_experiments:
        sys_prompt += f"\n\nPast Experiments:\n{past_experiments}"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "Please start a new experiment loop to improve the model composite score. First, inspect train.py to devise an experiment, then run the cycle."}
    ]
    
    print("Starting autoresearch loop...")
    
    for step in range(MAX_STEPS):
        print(f"\n--- Step {step+1}/{MAX_STEPS} ---")
        
        max_retries = 5
        base_delay = 2
        complete_content = ""
        tool_calls_dict = {}
        success = False
        
        for attempt in range(max_retries):
            try:
                response = completion(model=model, api_key=api_key, messages=messages, tools=tools, stream=True)
                
                print("\n[LLM] ", end="")
                sys.stdout.flush()
                
                complete_content = ""
                tool_calls_dict = {}
                
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    
                    if hasattr(delta, 'content') and delta.content is not None:
                        complete_content += delta.content
                        sys.stdout.write(delta.content)
                        sys.stdout.flush()
                        
                    if hasattr(delta, 'tool_calls') and delta.tool_calls is not None:
                        for tc in delta.tool_calls:
                            index = tc.index
                            if index not in tool_calls_dict:
                                tool_calls_dict[index] = {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name or "",
                                        "arguments": ""
                                    }
                                }
                                func_name = tc.function.name or ""
                                sys.stdout.write(f"\n[LLM] Calling Tool: {func_name}\n[Tool Args] ")
                                sys.stdout.flush()
                                
                            if hasattr(tc.function, 'arguments') and tc.function.arguments is not None:
                                sys.stdout.write(tc.function.arguments)
                                sys.stdout.flush()
                                tool_calls_dict[index]["function"]["arguments"] += tc.function.arguments
                
                success = True
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "503" in err_str or "RateLimitError" in err_str or "ServiceUnavailableError" in err_str or "MidStreamFallbackError" in err_str:
                    if attempt == max_retries - 1:
                        print(f"\nLLM API Error: Max retries reached. {e}")
                        break
                    sleep_time = base_delay * (2 ** attempt)
                    print(f"\nLLM API Error (429/503): {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"\nLLM API Error: {e}")
                    break
        
        if not success:
            print("\nError: Failed to get valid response from LLM.")
            break
            
        print() # Newline after response completes
        
        message_dict = {
            "role": "assistant",
            "content": complete_content or None
        }
        
        if tool_calls_dict:
            message_dict["tool_calls"] = [tool_calls_dict[idx] for idx in sorted(tool_calls_dict.keys())]
            
        messages.append(message_dict)
        
        if tool_calls_dict:
            for idx in sorted(tool_calls_dict.keys()):
                tc_dict = tool_calls_dict[idx]
                func_name = tc_dict["function"]["name"]
                
                # Mock a tool_call object for execute_tool_call
                class MockFunction:
                    def __init__(self, name, arguments):
                        self.name = name
                        self.arguments = arguments
                class MockToolCall:
                    def __init__(self, function):
                        self.function = function
                
                mock_tool_call = MockToolCall(MockFunction(func_name, tc_dict["function"]["arguments"]))
                
                # Execute tool
                result = execute_tool_call(mock_tool_call)
                
                # Log outcome appropriately
                if func_name == "write_file":
                    print(f"[Tool] write_file execution complete. ({len(str(result))} chars log)")
                elif func_name == "read_file":
                    print(f"[Tool] read_file read {len(str(result))} characters.")
                else:
                    print(f"[Tool] Output: {str(result)[:200]}...")
                    
                # Append tool result back to message history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_dict["id"],
                    "name": func_name,
                    "content": str(result)
                })
                
                # If report is created, loop iteration is finished
                if func_name == "create_report":
                    print("\nAutoresearch iteration complete.")
                    return
        else:
            if "complete" in (complete_content or "").lower():
                break

if __name__ == "__main__":
    # response = completion(
    #     model=model,
    #     api_key=api_key,
    #     messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
    # )
    # print(response)
    main()
