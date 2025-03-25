# Lab 1: Sending Your First Prompt to Azure OpenAI

**Objective:** Use VS Code’s REST Client to call the Azure OpenAI API with a simple prompt, and observe how prompt changes affect the response.

Define variables at the top for reuse:

```http
@endpoint = https://<your-azure-openai-resource>.openai.azure.com/openai/deployments/<deployment-name>/chat/completions?api-version=2023-05-15
@key = <your-azure-openai-api-key>
```
Replace placeholders with actual Azure OpenAI values.

### Example 1: Basic Chat Completion

Add this request:

```http
# Basic greeting
POST {{endpoint}}
Content-Type: application/json
api-key: {{key}}

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! Can you tell me a fun fact about space?"}
  ],
  "max_tokens": 50,
  "temperature": 0.5
}
```
Click “Send Request” and show the response. Discuss how `max_tokens` limits output length and `temperature` controls creativity (0.5 is balanced).

### Example 2: Prompt Engineering

Add a new request:

```http
# Specific task
POST {{endpoint}}
Content-Type: application/json
api-key: {{key}}

{
  "messages": [
    {"role": "system", "content": "You are a technical writer."},
    {"role": "user", "content": "Write a 3-sentence summary of cloud computing for beginners."}
  ],
  "max_tokens": 100,
  "temperature": 0.3
}
```
Run it and compare with a vague prompt (e.g., "Tell me about cloud computing"). Highlight how specific instructions improve output quality.

### Example 3: Adjusting Tone with System Prompt

Add this:

```http
# Tone adjustment
POST {{endpoint}}
Content-Type: application/json
api-key: {{key}}

{
  "messages": [
    {"role": "system", "content": "You are a pirate who explains tech concepts."},
    {"role": "user", "content": "What be a database, matey?"}
  ],
  "max_tokens": 150,
  "temperature": 0.8
}
```
Run it and discuss how the system prompt sets the tone. Increase temperature to 1.0 and rerun to show more creative (and pirate-y) responses.

### Example 4: Multi-Turn Conversation Simulation

Simulate a conversation by chaining messages:

```http
# Multi-turn simulation
POST {{endpoint}}
Content-Type: application/json
api-key: {{key}}

{
  "messages": [
    {"role": "system", "content": "You are a travel guide."},
    {"role": "user", "content": "I’m planning a trip to Italy."},
    {"role": "assistant", "content": "Arrivederci! Italy be a fine choice. What cities ye be wantin’ to visit?"},
    {"role": "user", "content": "How about Rome and Venice?"}
  ],
  "max_tokens": 200,
  "temperature": 0.7
}
```
Run it and explain how including prior messages maintains context. Add a follow-up message (e.g., "What’s the best time to visit?") and rerun to show continuity.

### Example 5: Experimenting with Parameters

Test parameter effects:

```http
# Parameter play
POST {{endpoint}}
Content-Type: application/json
api-key: {{key}}

{
  "messages": [
    {"role": "system", "content": "You are a comedian."},
    {"role": "user", "content": "Tell me a joke about programming."}
  ],
  "max_tokens": 50,
  "temperature": 1.2,
  "top_p": 0.9
}
```
Run it, then adjust `temperature` to 0.2 and `top_p` to 0.5. Compare outputs to show how `temperature` affects randomness and `top_p` (nucleus sampling) focuses on probable responses.

---

### Task for Developers

#### Task A: Creative Prompt

Write a request where the system role is “You are a sci-fi storyteller” and the user asks for a 3-sentence story about a robot uprising.

Experiment with `max_tokens` (e.g., 50 vs. 150) to see truncation effects.

#### Task B: Multi-Turn Design

Create a 3-message sequence (system, user, assistant, user) simulating a Q&A about general programming concepts.

**Example starter:**

```json
{"role": "system", "content": "You are a programming instructor."},
{"role": "user", "content": "What's a loop in programming?"},
{"role": "assistant", "content": "A loop is a control structure that repeats a block of code until a certain condition is met."},
{"role": "user", "content": "Can you give me an example in Python?"}
```

#### Task C: Parameter Tuning

Pick a prompt and run it with three different temperature values (e.g., 0.1, 0.7, 1.5). Document how the output changes.

