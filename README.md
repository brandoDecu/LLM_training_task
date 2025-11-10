# 🧠 RL Task: Train and Export a Neural Network to ONNX

This repository contains a **Reinforcement Learning (RL) training task** designed to teach large language models (LLMs) how to **train a PyTorch neural network**, **export it to ONNX**, and **evaluate model performance** — a realistic ML engineering workflow.

---

## 📘 Overview

The task challenges an LLM to:

1. Write Python code to **train a neural network in PyTorch** using a provided dataset (`train.csv` and `test.csv`).
2. **Export the trained model** to ONNX format (`model.onnx`).
3. **Print the train and test accuracy** during training.
4. **Achieve a minimum test accuracy of 0.875**.
5. Stop once the model reports success by outputting `"TASK COMPLETE"`.

The task tests the LLM’s ability to reason about:

* Model architecture selection
* Training loops
* Dataset handling in pandas and PyTorch
* Model export to ONNX
* Performance evaluation

---

## ⚙️ How It Works

Each iteration of the RL loop simulates an **interactive coding session** between the model and a set of predefined tools.

### Tools

The model is allowed to use two tools:

* **`write_file`** — create or modify Python files.
* **`run_python`** — execute Python scripts and return their stdout/stderr.

### Task Flow

1. A random classification dataset is generated with 10 features using `sklearn.make_classification`.
2. The model receives the dataset and instructions via the prompt.
3. It can make up to **four tool calls** to complete the task, giving it a chance to correct errors.
4. After training, the grader loads `model.onnx`, runs inference with `onnxruntime`, and computes test accuracy.
5. Success is recorded if accuracy ≥ 0.875.

---

## 🧩 File Structure

```
.
├── rl_task.py         # Main task definition and grading script
├── README.md          # This file
```

---

## 🧪 Running the Task

### Requirements

* Python 3.9+
* Dependencies:
anthropic numpy pandas scikit-learn torch onnx onnxruntime

* Anthropic API key (free tier supported)

### Run the Task

```bash
export ANTHROPIC_API_KEY=your_api_key_here
python rl_task.py
```

This will:

* Generate datasets
* Run the model-building task 10 times
* Print the number of successful completions

---

## 🧮 Grading Criteria

| Criterion                | Description                                                 |
| ------------------------ | ----------------------------------------------------------- |
| ✅ **Model Export**       | File `model.onnx` must exist                                |
| ✅ **Performance**        | Test accuracy ≥ 0.875                                       |
| ✅ **Correct Evaluation** | Predictions are compared correctly (binary classification)  |
| ✅ **Prompt Alignment**   | All requirements in the prompt are verifiable by the grader |

A task is **successful** if the ONNX model achieves the target accuracy.

---

## 🎯 Purpose

This task evaluates and trains LLMs on:

* Realistic ML engineering workflows
* Correct use of ONNX for model deployment
* Writing robust, efficient PyTorch training code under time constraints

The **pass rate target** is between **10% and 40%**, ensuring the task is challenging but learnable.

---

## 📈 Example Output

```
==================================================
Running task 1/10
==================================================
>>> LLM Iteration: 1. Calls made: 0/2
Stop reason: tool_use
Calling tool: write_file
Result: {'success': True, 'message': 'Wrote to /tmp/tmpabcd1234/train_model.py'}

>>> LLM Iteration: 2. Calls made: 1/2
Stop reason: tool_use
Calling tool: run_python
Result: { 'stdout': 'Train acc: 0.95 | Test acc: 0.88\nTASK COMPLETE' }

=============== GRADING ===================
Test accuracy: 0.8800
✓ Task succeeded!

Completed 10 tasks with success rate of: 30%
```