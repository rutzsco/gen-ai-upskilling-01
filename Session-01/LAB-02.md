# Lab 2: Prompt Engineering with Azure Open AI Chat Completions API

### 1. Instruction Prompting
**Goal**: Use clear, direct language to instruct the model to complete a task.  
**Explanation**: Instruction prompting involves explicitly telling the model what to do without extra context or examples. The key is clarity and specificity.

**Example Request**:
```http
POST {{endpoint}}
Content-Type: application/json
api-key: {{api-key}}

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize the following article in 3 bullet points: 'Artificial intelligence (AI) is transforming industries worldwide. From healthcare to finance, AI systems are improving efficiency and decision-making. Recent advancements in natural language processing have enabled more human-like interactions with machines.'"}
  ],
  "max_tokens": 100
}
```

**Expected Response**:
- AI is revolutionizing industries globally.  
- It enhances efficiency and decision-making in sectors like healthcare and finance.  
- Advances in NLP allow machines to interact more naturally with humans.

**Exercise**:
- Modify the prompt to summarize a different short text (e.g., a paragraph about climate change).
- Run the request and observe the output.

---

### 2. Few-shot Prompting
**Goal**: Guide the model with examples to improve its performance on a task.  
**Explanation**: Few-shot prompting provides the model with a few examples within the prompt to demonstrate the desired behavior, helping it generalize to new inputs.

**Example Request**:
```http
POST {{endpoint}}
Content-Type: application/json
api-key: {{api-key}}

{
  "messages": [
    {"role": "system", "content": "You are a translator."},
    {"role": "user", "content": "Translate the following sentences from English to Spanish. Here are examples:\n- English: Hello → Spanish: Hola\n- English: Goodbye → Spanish: Adiós\nNow translate: 'Thank you'"}
  ],
  "max_tokens": 50
}
```

**Expected Response**: `"Gracias"`

**Exercise**:
- Add two more examples (e.g., "Yes → Sí" and "No → No").
- Ask the model to translate "Please".
- Run the request and verify the output.

---

### 3. Zero-shot Prompting
**Goal**: Ask the model to perform a task without any examples.  
**Explanation**: Zero-shot prompting relies on the model's pre-trained knowledge to handle tasks without prior guidance. It’s useful when you want quick results and don’t have examples handy.

**Example Request**:
```http
POST {{endpoint}}
Content-Type: application/json
api-key: {{api-key}}

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Classify the sentiment of this sentence as positive, negative, or neutral: 'The weather is beautiful today.'" }
  ],
  "max_tokens": 50
}
```

**Expected Response**: `"Positive"`

**Exercise**:
- Change the sentence to `"I’m so tired of this rain."`
- Run the request and check if the sentiment classification changes to `"Negative"`.

---

### 4. Role Prompting
**Goal**: Assign a specific role to the model to shape its responses.  
**Explanation**: By defining a role, you can steer the model’s tone, expertise, or perspective to suit your needs.

**Example Request**:
```http
POST {{endpoint}}
Content-Type: application/json
api-key: {{api-key}}

{
  "messages": [
    {"role": "system", "content": "You are a helpful technical support agent."},
    {"role": "user", "content": "My printer isn’t working. What should I do?"}
  ],
  "max_tokens": 150
}
```

**Expected Response**:  
A step-by-step troubleshooting guide, e.g.,  
_"First, check if the printer is powered on. Then, ensure it’s connected to your computer..."_

**Exercise**:
- Change the role to `"You are a friendly chef."`
- Ask, `"How do I make a quick breakfast?"`
- Run the request and note how the tone and content shift.

---

### 5. Formatting Guidance
**Goal**: Specify the output format (e.g., tables, bullet points, JSON) for structured responses.  
**Explanation**: Adding formatting instructions ensures the model returns data in a usable, organized way.

**Example Request**:
```http
POST {{endpoint}}
Content-Type: application/json
api-key: {{api-key}}

{
  "messages": [
    {"role": "system", "content": "You are a data analyst."},
    {"role": "user", "content": "List 3 benefits of AI in healthcare. Return the result as a JSON object."}
  ],
  "max_tokens": 100
}
```

**Expected Response**:
```json
{
  "benefits": [
    "Improved diagnostic accuracy",
    "Faster patient data analysis",
    "Personalized treatment plans"
  ]
}
```

**Exercise**:
- Modify the prompt to request the output as a bullet-point list instead of JSON.
- Run the request and confirm the output format changes.

---

## Putting It All Together

### Final Challenge
**Goal**: Combine multiple techniques into one prompt.

**Scenario**:
- Use **role prompting**: `"You are a history teacher"`
- Apply **few-shot prompting**: Provide examples of historical event summaries
- Include **formatting guidance**: `"Return the result as a table"`

**Sample Request**:
```http
POST {{endpoint}}
Content-Type: application/json
api-key: {{api-key}}

{
  "messages": [
    {"role": "system", "content": "You are a history teacher."},
    {"role": "user", "content": "Summarize historical events. Examples:\n- Event: Battle of Waterloo → Summary: Napoleon was defeated in 1815.\n- Event: Signing of the Magna Carta → Summary: Limited royal power in 1215.\nNow summarize 'The Industrial Revolution' and return the result as a table."}
  ],
  "max_tokens": 150
}
```

**Expected Response**:

| Event                    | Summary                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| The Industrial Revolution | Began in the 18th century, transformed economies with machines and factories. |

---

## Wrap-Up

**What you’ve learned**:
- Instruction Prompting  
- Few-shot Prompting  
- Zero-shot Prompting  
- Role Prompting  
- Formatting Guidance

**Next steps**:
- Experiment with more complex prompts  
- Adjust `max_tokens` or `temperature` parameters  
- Explore additional Azure Open AI features
