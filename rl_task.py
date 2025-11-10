import anthropic
import os
import tempfile
import subprocess
from sklearn.datasets import make_blobs, make_moons, make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import onnxruntime as ort

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

num_successes = 0


# === Define tools: Write File and Run Python ===
tools = [
    {
        "name": "write_file",
        "description": "Write contents to a file at the given path",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_python",
        "description": "Run a Python script and return its stdout and stderr",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
]


def run_tool(name, input_data, cwd):
    """
    Runs the tool based on it's name. Requires an input data
    """
    if name == "write_file":
        try: 
            path = os.path.join(cwd, input_data["path"])
            with open(path, "w") as f:
                f.write(input_data["content"])
            return {"success": True, "message": f"Wrote to {path}"}
        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    elif name == "run_python":
        path = os.path.join(cwd, input_data["path"])
        try:
            result = subprocess.run(
                ["python3", path],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {"stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Error: Script execution timed out after 60 seconds"
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Error: {str(e)}"
            }


# === Dataset generation ===
def generate_dataset(tmpdir):
    X, y = make_classification(n_samples=400, n_features=10, n_informative=4, 
                               n_redundant=3, n_repeated=0, n_classes=2, 
                               n_clusters_per_class=2, weights=None, flip_y=0.01, 
                               class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, 
                               shuffle=True, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    train = np.column_stack((X_train, y_train))
    test = np.column_stack((X_test, y_test))
    
    train_df = pd.DataFrame(train, columns=[f"x{i+1}" for i in range(X.shape[1])] + ["label"])
    test_df = pd.DataFrame(test, columns=[f"x{i+1}" for i in range(X.shape[1])] + ["label"])
    
    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)
    
    train_df.to_csv(f"{tmpdir}/train.csv", index=False)
    test_df.to_csv(f"{tmpdir}/test.csv", index=False)




def gen_prompt(tmpdir, calls_left):
    prompt = f"""
You have {calls_left} tool calls left.
You are given two CSV files in the directory: train.csv and test.csv.

Your task:
1. Write Python code to train a neural network in PyTorch on train.csv.
2. After training, save the trained model in **ONNX format** to a file called `model.onnx`.
3. Print both the train and test accuracy in your Python script.
4. Your goal is to reach at least 0.85 test accuracy.

Once you see test accuracy >= 0.85, respond with "TASK COMPLETE".

Warning: the python code times out at 60 seconds. So periodically save the best model.

You can call `write_file` to create files and `run_python` to execute them.
Working directory: {tmpdir}
"""
    return prompt



# === Conversation loop ===
def run_task(tmpdir):
    max_calls = 2
    calls_made = 0
    user_prompt = gen_prompt(tmpdir, max_calls)
    messages = [{"role": "user", "content": user_prompt}]
    i = 0
    
    while calls_made < max_calls:  # Add max iterations to prevent infinite loops
        i += 1
        print(f">>> LLM Iteration: {i}. Calls made: {calls_made}/{max_calls}")
        messages[0]["content"] = gen_prompt(tmpdir, max_calls-calls_made)
        
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            tools=tools,
            messages=messages,
        )
        
        print(f"Stop reason: {resp.stop_reason}")
        
        # Add assistant's response to messages
        messages.append({"role": "assistant", "content": resp.content})
        
        # Check if we need to process tool calls
        if resp.stop_reason == "tool_use":
            calls_made += 1
            # Process all tool uses in the response
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    print(f"Calling tool: {block.name}")
                    tool_result = run_tool(block.name, block.input, tmpdir)
                    print(f"Result: {tool_result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(tool_result)
                    })
            
            # Add tool results as a user message
            messages.append({"role": "user", "content": tool_results})
            
        elif resp.stop_reason == "end_turn":
            # Print final response
            for block in resp.content:
                if hasattr(block, "text"):
                    print("Final response:", block.text)
            break
        else:
            print(f"Unexpected stop reason: {resp.stop_reason}")
            break
    print(f"Finished task. Calls made: {calls_made}/{max_calls}")


def grade_task(tmpdir):
    print("=============== GRADING ===================")
    global num_successes
    
    try:
        # Load test data
        test_path = os.path.join(tmpdir, 'test.csv')
        test_data = pd.read_csv(test_path)
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values
        
        # Load ONNX model
        model_path = os.path.join(tmpdir, 'model.onnx')
        if not os.path.exists(model_path):
            print("✗ Task failed - model.onnx not found")
            return
        
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        # Run inference
        y_pred = session.run(None, {input_name: X_test.astype(np.float32)})[0]
        
        # Convert probabilities/logits to 0/1 if needed
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            y_pred_labels = (y_pred.flatten() > 0.5).astype(int)
        
        # Calculate accuracy
        correct = (y_pred_labels == y_test).sum()
        accuracy = correct / len(y_test)
        
        print(f"Test accuracy: {accuracy:.4f}")
        
        if accuracy >= 0.85:
            num_successes += 1
            print("✓ Task succeeded!")
        else:
            print("✗ Task failed - accuracy below 0.85")
            
    except Exception as e:
        print(f"✗ Task failed with error: {e}")

if __name__ == "__main__":
    # tmpdir = '/home/brandodecu/AIChampTest/tests'
    
    
    # Create directory if it doesn't exist
    # os.makedirs(tmpdir, exist_ok=True)
    num_tasks = 10
    

    for i in range(10):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(tmpdir)
            print(f"\n{'='*50}")
            print(f"Running task {i+1}/{num_tasks}")
            print(f"{'='*50}\n")
            run_task(tmpdir)
            grade_task(tmpdir)
        
    print(f"\n{'='*50}")
    print(f"Completed {num_tasks} tasks with success rate of: {num_successes/num_tasks*100}%")
    print(f"{'='*50}")