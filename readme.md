# QA Summarization Application using LangGraph, FastAPI, and MongoDB

## Overview

This application processes a given paragraph of text containing questions. It extracts these questions, fetches answers using OpenAI's general knowledge, Brave Search, and Tavily Search for real-time information, and then synthesizes a final summary based on the original paragraph and the gathered answers.

The application uses:
- **LangGraph** to define and execute the workflow as a state machine.
- **LangChain** to interact with OpenAI models and Search APIs (Brave via custom tool, Tavily via built-in tool).
- **FastAPI** to expose the workflow as a web API endpoint.
- **MongoDB** (managed via Docker) for persisting the workflow state using LangGraph's checkpointer mechanism, allowing for resilience and potential resumption.
- **LangSmith** (optional) for tracing and observability.
- **Twilio** (optional) for WhatsApp integration.

## Features

- Extracts questions directly from the input text.
- Fetches answers from multiple sources:
    - OpenAI (GPT models) - runs in parallel with search branch.
    - Brave Search API (for current information) - runs first in the search branch.
    - Tavily Search API (for comprehensive search results) - runs after Brave Search.
- Limits OpenAI answer length (currently 200 tokens).
- Adds a delay between Brave Search API calls (currently 1 second) to respect free tier limits.
- Truncates Tavily Search results to approximately 200 tokens (800 characters).
- Combines the original text and gathered answers into a final summary using OpenAI.
- Exposes functionality via a FastAPI endpoint (`/process`).
- Uses MongoDB for persistent state management of the workflow runs.
- Integrates with LangSmith for tracing (requires configuration).
- Includes a webhook endpoint for Twilio WhatsApp integration.

## Technology Stack

- Python 3.x
- LangGraph
- LangChain (Core, OpenAI, Community)
- FastAPI
- Uvicorn (ASGI Server)
- Pydantic (for API models)
- Requests (for Brave Search tool)
- tavily-python (for Tavily Search tool)
- python-dotenv (for environment variables)
- MongoDB (via Docker)
- pymongo (MongoDB driver)
- motor (Async MongoDB driver for checkpointer)
- twilio (for WhatsApp integration)
- Docker & Docker Compose

## Setup Instructions

1.  **Clone/Download:** Get the project files (`app.py`, `docker-compose.yml`, etc.).
2.  **Create `.env` File:** Create a file named `.env` in the project root and add your API keys:
    ```env
    # --- Core APIs ---
    OPENAI_API_KEY="your_openai_api_key"
    BRAVE_SEARCH_API_KEY="your_brave_search_api_key"
    TAVILY_API_KEY="your_tavily_api_key" # Add your Tavily key

    # --- Twilio (for WhatsApp Integration) ---
    TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    TWILIO_AUTH_TOKEN="your_auth_token"
    TWILIO_WHATSAPP_NUMBER="whatsapp:+14155238886" # Your Twilio WhatsApp number

    # --- MongoDB Configuration ---
    MONGO_URI="mongodb://localhost:27017/"
    MONGO_DB_NAME="langgraph_db"
    MONGO_COLLECTION_NAME="qa_workflows"

    # --- LangSmith Tracing (Optional) ---
    LANGCHAIN_TRACING_V2="true"
    LANGCHAIN_API_KEY="your_langsmith_api_key" # Create key at langsmith.com
    LANGCHAIN_PROJECT="QA Summarization App"   # Choose a project name
    # LANGCHAIN_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)" # Optional: Defaults to this
    ```
3.  **Start MongoDB:** Ensure Docker Desktop is running. Open a terminal in the project directory and run:
    ```bash
    docker-compose up -d
    ```
    This will start a MongoDB container in the background, accessible at `mongodb://localhost:27017`.

4.  **Create Python Virtual Environment:**
    ```bash
    # Navigate to project directory
    cd path/to/your/project
    # Create environment (use python3 if needed)
    python -m venv venv
    # Activate environment
    # macOS/Linux:
    source venv/bin/activate
    # Windows (cmd):
    # venv\Scripts\activate.bat
    # Windows (PowerShell):
    # .\venv\Scripts\Activate.ps1
    ```

5.  **Install Dependencies:** Create a `requirements.txt` file with the following content (or ensure these are present):
    ```txt
    # requirements.txt
    langchain
    langchain-core
    langchain-openai
    langchain-community
    langgraph
    pymongo # MongoDB driver
    motor # Async MongoDB driver (used by MongoDBSaver)
    tavily-python>=0.3.0 # Tavily client
    # Ensure langgraph[mongodb] or similar is installed if needed for MongoDBSaver
    pydantic>=2.0,<3.0 # Check FastAPI version compatibility if needed
    python-dotenv>=1.0.0
    requests>=2.20
    fastapi
    uvicorn[standard]
    twilio # For WhatsApp integration
    ```
    Then, install them (make sure your virtual environment is active):
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure the `langgraph.checkpoint.mongodb.MongoDBSaver` class is available. If you encounter import errors, you might need a specific version or extra install for langgraph related to MongoDB support.)*

## Running the Application

1.  **Activate Virtual Environment:** `source venv/bin/activate` (or equivalent).
2.  **Start FastAPI Server:** Run the following command in your terminal from the project directory:
    ```bash
    uvicorn app:api --reload --host 0.0.0.0 --port 8000
    ```
    The server will start, likely on `http://0.0.0.0:8000`.

## API Endpoint

- **URL:** `/process`
- **Method:** `POST`
- **Request Body (JSON):**
    ```json
    {
      "paragraph": "Your text containing questions here...",
      "thread_id": "optional_existing_thread_id"
    }
    ```
    - `paragraph` (string, required): The text to process.
    - `thread_id` (string, optional): If provided, the workflow attempts to resume or use the state associated with this ID (useful for multi-turn interactions if the graph supported them). If omitted, a new unique ID is generated.
- **Success Response (JSON, Status 200):**
    ```json
    {
      "thread_id": "generated_or_provided_thread_id",
      "summary": "The generated summary text...",
      "error": null
    }
    ```
- **Error Response (JSON, Status 500):**
    ```json
    {
      "detail": {
          "message": "Error description...",
          "thread_id": "thread_id_being_processed",
          "last_known_state": { ... } // Optional: Last state if retrievable
      }
    }
    ```
- **Testing:** You can easily test this endpoint using the interactive Swagger UI provided by FastAPI at `http://localhost:8000/docs` in your browser.

## WhatsApp Integration (Twilio)

- **Webhook Endpoint:** `/twilio/webhook` (Handles `POST` requests from Twilio)
- **Setup:**
    1.  Configure your Twilio WhatsApp number's "WHEN A MESSAGE COMES IN" webhook to point to your deployed application's `/twilio/webhook` URL (use ngrok for local testing). Method must be `HTTP POST`.
    2.  Ensure `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `TWILIO_WHATSAPP_NUMBER` are set in your `.env` file.
- **Functionality:** Receives incoming WhatsApp messages, triggers the `/process` endpoint in the background, and sends the summary/error back to the user via Twilio (truncating long messages).

## LangGraph Workflow

The application's logic is orchestrated using a LangGraph `StateGraph`.

**State (`GraphState`):**

The graph maintains the following state information during execution:
- `paragraph`: The initial input text.
- `questions`: A list of questions extracted from the paragraph. (Annotated with `operator.add`)
- `openai_answers`: A dictionary mapping extracted questions to answers generated by OpenAI.
- `search_answers`: A dictionary mapping extracted questions to snippets found by Brave Search.
- `tavily_answers`: A dictionary mapping extracted questions to snippets found by Tavily Search.
- `final_summary`: The final synthesized summary text.
- `error`: An optional string containing error messages if any node fails.

**Nodes:**

1.  **`generate_questions` (Entry Point):** Extracts questions from the `paragraph`. Updates `questions`.
2.  **`fetch_openai_answers`:** Gets OpenAI answers for `questions`. Updates `openai_answers`. (Runs in parallel with search branch)
3.  **`fetch_brave_search_answers`:** Gets Brave Search snippets for `questions`. Updates `search_answers`. (First step in search branch)
4.  **`fetch_tavily_answers`:** Gets Tavily Search snippets for `questions`. Updates `tavily_answers`. (Runs *after* Brave Search)
5.  **`combine_and_summarize`:** Combines inputs (OpenAI, Brave, Tavily) and generates a summary using OpenAI. Updates `final_summary`.

**Workflow Diagram (Mermaid):**

```mermaid
graph TD
    A[Start] --> B(generate_questions);
    B --> C{Parallel Fork};
    C --> D[fetch_openai_answers];
    C --> E[fetch_brave_search_answers];
    E --> F[fetch_tavily_answers];
    D --> G{Join};
    F --> G;
    G --> H(combine_and_summarize);
    H --> I[End];

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style G fill:#ccf,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5


Edges (Flow Explained):
Entry: The graph starts at generate_questions.
Parallel Fork: After generate_questions, the workflow splits into two main branches that run concurrently:
Branch 1: Executes fetch_openai_answers.
Branch 2: Executes fetch_brave_search_answers.
Sequential Search: Within Branch 2, after fetch_brave_search_answers completes, fetch_tavily_answers is executed.
Join: The graph waits until both Branch 1 (fetch_openai_answers) and the end of Branch 2 (fetch_tavily_answers) have finished.
Summarize: Once both branches are complete, the combine_and_summarize node runs.
End: After combine_and_summarize, the graph reaches its END state.
Checkpointer Configuration (MongoDB)
The application attempts to use langgraph.checkpoint.mongodb.MongoDBSaver for persistence.
Requires the pymongo library.
Connects to MongoDB using MONGO_URI, MONGO_DB_NAME, and MONGO_COLLECTION_NAME from .env.
Initializes the saver using a synchronous pymongo.MongoClient.
Ensure your MongoDB server (e.g., Docker container) is running.
If initialization fails, a warning is printed, and persistence is disabled.
LangSmith Tracing Configuration
To enable tracing, set the following environment variables in your .env file:
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="Your Project Name"
Sign up at langsmith.com to get an API key.
No code changes are required; tracing is automatic when variables are set.
View traces in your project on the LangSmith dashboard.
Potential Improvements
More Robust Error Handling: Implement more specific error handling within nodes and potentially add dedicated error-handling paths in the graph.
Conditional Logic: Add conditional edges based on the state (e.g., skip summarization if no answers were found).
Streaming Response: Modify the FastAPI endpoint and LangGraph invocation to stream intermediate results or the final summary back to the client.
Async Checkpointer: If MongoDBSaver fully supports async initialization and usage with motor, switch back to async def process_paragraph and use ainvoke/aget_state for potentially better performance under load.
Async Tool Calls: If an async Brave Search client/wrapper becomes available, update the BraveSearchTool._arun method for true async I/O. (Tavily tool might already support async).
Batching: For a very large number of questions, consider batching requests to OpenAI or the search tool if their APIs support it.
Configuration: Move model names, delays, token limits, etc., to a configuration file or environment variables for easier management.
WhatsApp UX: Send immediate "Processing..." message via WhatsApp. Handle non-text messages gracefully. Split very long summaries into multiple WhatsApp messages instead of truncating.
