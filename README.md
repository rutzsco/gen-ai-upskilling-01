# Lesson Plan: Building GenAI Applications and Semantic Kernel
**Course Overview:** This hands-on course is divided into three 3-hour sessions. It will introduce participants to developing Generative AI (GenAI) applications using Python and Microsoft’s Semantic Kernel. Across the sessions, learners will progressively build confidence and skills in prompt engineering, retrieval-augmented generation (RAG), agent-based AI, and function calling. Each session includes a mix of lecture content, guided lab exercises, and quizzes to reinforce learning.

**Prerequisites:** Participants should have the following ready before starting:
- Visual Studio Code (VS Code) with the [REST Client extension](https://marketplace.visualstudio.com/items?itemName=humao.rest-client) installed.
- Python 3.11 (or above) installed and configured.
- An Azure OpenAI resource with a deployment (e.g., GPT-3.5-Turbo or GPT-4) and a valid API key.
- Basic familiarity with Python programming (beginner level) and JSON data format.

**Learning Objectives:** By the end of the course, participants will:
- Understand what *prompt engineering* is and apply best practices to craft effective prompts.
- Build GenAI applications that use **Retrieval-Augmented Generation (RAG)** to ground LLM responses with external data.
- Differentiate between *static RAG* (pre-fetched context) and *agentic RAG* (dynamic tool use).
- Utilize the **Semantic Kernel (SK)** SDK in Python to integrate LLMs into applications.
- Implement basic AI “agents” that can call functions (tools) using OpenAI’s function calling capability.
- Gain confidence through labs in combining prompts, code, and external data to build intelligent applications.


## Agenda

### Session 1: Introduction to GenAI and Prompt Engineering
- **Presentation**
  - **Generative AI and Azure OpenAI Basics:** What are large language models and how we can use them via Azure OpenAI. Setting up the development environment and Semantic Kernel overview.
  - **Semantic Kernel Overview:** Understanding Semantic Kernel’s purpose and how it connects AI models with code ([Semantic Kernel](https://github.com/microsoft/semantic-kernel)).
  - **Prompt Engineering Fundamentals:** What is prompt engineering and why it matters ([Prompt engineering techniques](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)). Crafting effective prompts (dos and don’ts, best practices).
- **Labs**
  - **Lab 1:** Using the REST Client to send a prompt to Azure OpenAI and observe the output.
  - **Lab 2:** Prompt Engineering Best Practices: Tips to refine prompts for better results (specificity, context, examples, etc.).
  - **Lab 3:** Creating a simple Python script with Semantic Kernel to call an LLM and experimenting with prompt improvements.
- **Discussion and Knowledge Check:** QnA, Discussion, Knowledge checks.

### Session 2: Retrieval-Augmented Generation (RAG) – Grounding the AI with Data
- **Presentation**
  - **Recap & Motivation:** Brief review of Session 1 and discuss LLM limitations (hallucinations, knowledge cut-off) that motivate Retrieval-Augmented Generation.
  - **Introduction to RAG:** Concept and benefits of RAG – using external knowledge to ground LLM responses. Real-world examples (e.g., Bing Chat using search, or Q&A bots on company docs).
  - **Static vs Dynamic Retrieval:** Explain the difference between always providing pre-fetched context (static RAG) versus retrieving “on the fly” when needed (dynamic or agentic RAG).
  - **Semantic Search and Embeddings:** How semantic search works (embeddings & similarity) vs. keyword search, in simple terms.
- **Lab 1:** Create a small knowledge base and generate embeddings for its content (using Azure OpenAI embeddings).
- **Lab 2:** Implement a question-answering pipeline that uses the knowledge base: retrieve relevant info and feed it into an LLM prompt (static RAG).
- **Discussion and Knowledge Check:** Check understanding of RAG and embeddings.

### Session 3: Agentic AI – Dynamic Retrieval and Function Calling

- **Presentation**
  - **Recap:** Quick recap of static RAG pipeline from Session 2. Identify its limitations (requires developer to anticipate and fetch info). Introduce the need for more autonomous AI agents.
  - **What is an AI Agent?:** Understanding agentic behavior in LLMs – an agent can iteratively plan, use tools (functions), and incorporate results into its reasoning.
  - **Function Calling Mechanism:** Learn about OpenAI’s function calling feature that allows an LLM to call external functions through JSON requests. How Semantic Kernel plugins relate to this (functions as skills).
  - **Agentic RAG:** Combining retrieval with function calling – enabling the AI itself to decide when to retrieve information (dynamic RAG) rather than always being given context.
- **Lab 1:** Implement an AI “agent” that uses a function call to perform a knowledge base lookup (search) when needed (turning our RAG pipeline into a tool the AI can invoke).
- **Lab 2:** Extend function calling to another use-case (e.g., calling a calculator or a weather API) to illustrate agents executing different functions.
- **Discussion and Knowledge Check:** Final quiz on agents and function calling.

### Reference

- https://learn.microsoft.com/en-us/azure/developer/ai/introduction-build-generative-ai-solutions
- https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Chat/README.md
- https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview
- https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide
- [Retrieve data from plugins for RAG | Microsoft Learn](https://learn.microsoft.com/en-us/semantic-kernel/concepts/plugins/using-data-retrieval-functions-for-rag#:~:text=Pre)) ([Retrieve data from plugins for RAG | Microsoft Learn](https://learn.microsoft.com/en-us/semantic-kernel/concepts/plugins/using-data-retrieval-functions-for-rag#:~:text=Dynamic%20data%20retrieval)
- [GitHub - microsoft/semantic-kernel: Integrate cutting-edge LLM technology quickly and easily into your apps](https://github.com/microsoft/semantic-kernel#:~:text=Semantic%20Kernel%20is%20an%20SDK,a%20few%20lines%20of%20code)
- [Function calling and other API updates | OpenAI](https://openai.com/index/function-calling-and-other-api-updates/#:~:text=Developers%20can%20now%20describe%20functions,with%20external%20tools%20and%20APIs)
  

