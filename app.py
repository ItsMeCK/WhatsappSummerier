# app.py
import os
import json
import uuid
import time # Import time module for sleep
import requests # For Brave Search API call
import traceback # Import traceback module for detailed error logging
from dotenv import load_dotenv
from typing import List, Dict, TypedDict, Annotated, Sequence, Optional, Any
import operator # Needed for the Annotated state definition
from concurrent.futures import ThreadPoolExecutor

# --- FastAPI Imports ---
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Form, BackgroundTasks, Response
from pydantic import BaseModel as FastApiBaseModel # Alias to avoid Pydantic v1/v2 confusion

# --- Twilio Imports ---
# *** Correct import for TwilioClient and MessagingResponse ***
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse # Ensure this import is present

# LangChain components
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field # LangChain uses Pydantic v1
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import BaseTool # To create the custom Brave Search tool
# *** Import Tavily Search Tool ***
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph components
from langgraph.graph import StateGraph, END
# *** Import Official MongoDB Checkpointer ***
# Install required packages: pip install motor pymongo langgraph tavily-python twilio
try:
    # Assuming MongoDBSaver is available within langgraph's checkpoint module
    from langgraph.checkpoint.mongodb import MongoDBSaver
    MONGO_SAVER_AVAILABLE = True
except ImportError:
    # This might happen if MongoDBSaver isn't included directly or requires a specific extra install
    print("--------------------------------------------------------------------")
    print("WARNING: MongoDBSaver could not be imported from langgraph.checkpoint.mongodb.")
    print("Ensure 'motor' and 'pymongo' are installed. You might need a specific langgraph install like 'langgraph[mongodb]' or similar.")
    print("Checkpointer will not be used. State will not be persisted.")
    print("--------------------------------------------------------------------")
    MongoDBSaver = None # Define as None if import fails
    MONGO_SAVER_AVAILABLE = False

# from langgraph.checkpoint.sqlite import SqliteSaver # Comment out SQLite
from langgraph.checkpoint.base import BaseCheckpointSaver # Needed for type hinting

# --- MongoDB Client (Using synchronous pymongo for checkpointer init) ---
from pymongo import MongoClient
import motor.motor_asyncio # Import the async driver for potential async checkpointer usage if needed later

# --- Configuration & API Keys ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # Load Tavily Key
# *** MongoDB Configuration ***
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/") # Default for local Docker setup
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "langgraph_db")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "qa_workflows") # MongoDBSaver might need db/collection name during init

# *** Twilio Configuration ***
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER") # e.g., "whatsapp:+14155238886"


if not all([OPENAI_API_KEY, BRAVE_API_KEY, TAVILY_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER]):
    raise ValueError("Missing required environment variables for APIs (OpenAI, Brave, Tavily, Twilio)")

# --- Initialize Twilio Client ---
# Check if credentials exist before initializing
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    print("Twilio Client Initialized.")
else:
    twilio_client = None
    print("WARNING: Twilio credentials not found in environment variables. WhatsApp reply functionality disabled.")


# --- Model Definitions ---
# Use environment variables or specify models directly
QUESTION_MODEL_NAME = os.getenv("OPENAI_QUESTION_MODEL", "gpt-3.5-turbo")
ANSWER_MODEL_NAME = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4-turbo-preview")
SUMMARY_MODEL_NAME = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4-turbo-preview")

llm_question_gen = ChatOpenAI(model=QUESTION_MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
# *** Add max_tokens parameter to limit answer length ***
llm_answer = ChatOpenAI(model=ANSWER_MODEL_NAME, temperature=0.2, api_key=OPENAI_API_KEY, max_tokens=200)
llm_summarize = ChatOpenAI(model=SUMMARY_MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY, max_tokens=1    00)

# --- Custom Brave Search Tool ---
class BraveSearchTool(BaseTool):
    name: str = "brave_search"
    description: str = (
        "A wrapper around Brave Search API. "
        "Useful for when you need to answer questions about current events or real-time information. "
        "Input should be a search query."
    )
    api_key: str = Field(default=BRAVE_API_KEY)
    base_url: str = "https://api.search.brave.com/res/v1/web/search"
    results_count: int = 3

    def _run(self, query: str, run_manager: Optional[Any] = None) -> str:
        brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not brave_key: return "Error: BRAVE_SEARCH_API_KEY not set."
        headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": brave_key}
        params = {"q": query, "count": self.results_count}
        try:
            response = requests.get(self.base_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            snippets = [res.get("description", res.get("title", "")) for res in data.get("web", {}).get("results", []) if res.get("description") or res.get("title")]
            return "\n".join(snippets) if snippets else "No relevant information found by Brave Search."
        except requests.exceptions.RequestException as e: print(f"Error calling Brave Search API: {e}"); return f"Error: Could not retrieve search results ({e})"
        except Exception as e: print(f"Error processing Brave Search results: {e}"); return f"Error: Could not process results ({e})"

    async def _arun(self, query: str, run_manager: Optional[Any] = None) -> str:
        import asyncio
        await asyncio.sleep(1) # Keep delay for Brave free tier
        return await asyncio.to_thread(self._run, query, run_manager)

# Instantiate the Brave tool
brave_search_tool = BraveSearchTool()

# --- Instantiate Tavily Search Tool ---
# Uses TAVILY_API_KEY environment variable automatically
tavily_search_tool = TavilySearchResults(max_results=3)


# --- Output Parsers & Data Structures ---
# LangChain Pydantic V1 model for structured output
class Questions(LangChainBaseModel):
    """Schema for the list of questions generated from the paragraph."""
    questions: List[str] = Field(description="A list of distinct questions extracted directly from the input paragraph.")

# --- Graph State Definition ---
class GraphState(TypedDict):
    """
    Represents the state of our graph using Annotated fields
    where required by LangGraph for branching.
    """
    # Input field
    paragraph: str

    # Annotating 'questions' satisfies LangGraph's requirement for multiple outgoing edges.
    questions: Annotated[Optional[List[str]], operator.add]

    # Answer fields from different sources
    openai_answers: Optional[Dict[str, str]]
    search_answers: Optional[Dict[str, str]] # Brave Search results
    tavily_answers: Optional[Dict[str, str]] # *** Add Tavily answers key ***
    final_summary: Optional[str]
    error: Optional[str]


# --- Node Functions ---

def generate_questions_node(state: Dict) -> Dict[str, Any]:
    """
    Extracts sentences formatted as questions directly from the input paragraph.
    """
    print("--- Node: Extracting Questions ---")
    paragraph = state.get('paragraph')
    if not paragraph: return {"error": "Input paragraph is missing."}
    prompt = ChatPromptTemplate.from_messages([ SystemMessage(content="You are an expert text analysis tool... Extract questions exactly as they appear..."), HumanMessage(content=f"Here is the paragraph:\n\n{paragraph}\n\nPlease extract any questions present in this text.")])
    structured_llm_question_gen = llm_question_gen.with_structured_output(Questions)
    question_generation_chain = prompt | structured_llm_question_gen
    try:
        result = question_generation_chain.invoke({"paragraph": paragraph})
        questions = getattr(result, 'questions', []) if result else []
        print(f"Extracted Questions: {questions}")
        return {"questions": questions or [], "error": None} # Ensure list is returned
    except Exception as e: print(f"Error extracting questions: {e}"); return {"error": f"Failed to extract questions: {e}", "questions": None}


def fetch_openai_answers_node(state: Dict) -> Dict[str, Any]:
    """Fetches answers to questions using an OpenAI model in parallel (limited to 200 tokens)."""
    print("--- Node: Fetching OpenAI Answers ---")
    questions = state.get('questions')
    if not questions: return {"openai_answers": {}}
    prompt_template = PromptTemplate.from_template("Based on general knowledge, please answer the following question concisely:\nQuestion: {question}\nAnswer:")
    answer_chain = prompt_template | llm_answer
    results = {}
    def get_answer(question: str) -> str:
        try: return answer_chain.invoke({"question": question}).content
        except Exception as e: print(f"Error getting OpenAI answer for '{question}': {e}"); return f"Error: Could not get answer ({e})"
    with ThreadPoolExecutor(max_workers=5) as executor: answers = list(executor.map(get_answer, questions))
    results = dict(zip(questions, answers))
    print(f"OpenAI Answers fetched: {len(results)} answers.")
    return {"openai_answers": results}

def fetch_brave_search_answers_node(state: Dict) -> Dict[str, Any]:
    """Fetches answers/snippets for questions using the Brave Search tool in parallel with delay."""
    print("--- Node: Fetching Brave Search Answers ---")
    questions = state.get('questions')
    if not questions: return {"search_answers": {}}
    results = {}
    def get_search_result(question: str) -> str:
        try:
            print(f"Brave Node: Sleeping 1 second before searching for '{question[:50]}...'")
            time.sleep(1)
            return brave_search_tool.invoke(question) # Use the instantiated tool
        except Exception as e: print(f"Error getting Brave Search results for '{question}': {e}"); return f"Error: Could not get search results ({e})"
    with ThreadPoolExecutor(max_workers=5) as executor: search_data = list(executor.map(get_search_result, questions))
    results = dict(zip(questions, search_data))
    print(f"Brave Search Answers fetched: {len(results)} results.")
    return {"search_answers": results}

# *** Node for Tavily Search ***
def fetch_tavily_answers_node(state: Dict) -> Dict[str, Any]:
    """Fetches answers/snippets using Tavily Search and truncates results."""
    print("--- Node: Fetching Tavily Search Answers ---")
    questions = state.get('questions')
    if not questions: return {"tavily_answers": {}} # Return empty dict if no questions

    # Define character limit for truncation (approx. 200 tokens)
    CHARACTER_LIMIT = 800

    results = {}
    def get_tavily_result(question: str) -> str:
        try:
            # Invoke the Tavily tool
            search_results = tavily_search_tool.invoke(question)

            # Combine search snippets or just take the answer if available
            combined_result_str = ""
            if isinstance(search_results, list) and search_results:
                 snippets = [res.get('content', '') for res in search_results if res.get('content')]
                 combined_result_str = "\n".join(snippets) if snippets else "No content found by Tavily."
            elif isinstance(search_results, str): # Sometimes it might return a direct answer string
                combined_result_str = search_results
            else:
                combined_result_str = "No results found by Tavily."

            # *** Truncate the result string ***
            if len(combined_result_str) > CHARACTER_LIMIT:
                print(f"Tavily result for '{question[:30]}...' truncated from {len(combined_result_str)} to {CHARACTER_LIMIT} chars.")
                # Truncate and add ellipsis
                return combined_result_str[:CHARACTER_LIMIT] + "..."
            else:
                return combined_result_str

        except Exception as e:
            print(f"Error getting Tavily Search results for '{question}': {e}")
            return f"Error: Could not get Tavily search results ({e})"

    # Use ThreadPoolExecutor for parallel execution within the node
    with ThreadPoolExecutor(max_workers=5) as executor: # Limit workers if needed
        tavily_data = list(executor.map(get_tavily_result, questions))

    results = dict(zip(questions, tavily_data))

    print(f"Tavily Search Answers fetched: {len(results)} results.")
    return {"tavily_answers": results}


def combine_and_summarize_node(state: Dict) -> Dict[str, Any]:
    """Combines answers from OpenAI, Brave, and Tavily, then generates a final summary."""
    print("--- Node: Combining and Summarizing ---")
    questions = state.get('questions')
    openai_answers = state.get('openai_answers')
    search_answers = state.get('search_answers') # Brave results
    tavily_answers = state.get('tavily_answers') # *** Get Tavily results ***
    paragraph = state.get('paragraph')

    # Check if questions list exists and is not empty
    if not questions: # Handles None or empty list
         err_msg = "Cannot summarize: No questions were extracted or available."
         print(f"Warning: {err_msg}")
         return {"final_summary": err_msg, "error": None}

    # Handle cases where answer dictionaries might be None
    openai_answers = openai_answers or {}
    search_answers = search_answers or {}
    tavily_answers = tavily_answers or {} # *** Initialize if None ***

    # Format the combined information for the summarizer LLM
    combined_info = f"Original Paragraph:\n{paragraph}\n\n"
    combined_info += "Based on the paragraph, the following questions were extracted and answered using AI and Search Engines:\n\n" # Updated description

    for q in questions:
        combined_info += f"Question: {q}\n"
        combined_info += f"  - Answer from AI: {openai_answers.get(q, 'N/A')}\n"
        combined_info += f"  - Information from Brave Search: {search_answers.get(q, 'N/A')}\n"
        combined_info += f"  - Information from Tavily Search: {tavily_answers.get(q, 'N/A')}\n\n" # *** Add Tavily info ***

    # *** Use f-string for HumanMessage content ***
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an expert summarizer. Synthesize the information provided below, which includes an original paragraph, questions extracted from it, answers generated by an AI, and relevant snippets from Brave and Tavily web searches. Create a concise and comprehensive summary based on ALL the provided context. Focus on integrating the findings accurately."), # Updated description
        HumanMessage(content=f"Please summarize the following information:\n\n{combined_info}") # Use f-string here
    ])

    summarization_chain = prompt | llm_summarize

    try:
        # The variable 'combined_information' is passed via invoke's dictionary
        summary_response = summarization_chain.invoke({"combined_information": combined_info})
        final_summary = summary_response.content
        print(f"Final Summary generated.")
        # Return the summary string which will replace 'final_summary' in the state
        return {"final_summary": final_summary, "error": None}
    except Exception as e:
        print(f"Error during summarization: {e}")
        return {"error": f"Failed to summarize: {e}", "final_summary": None}


# --- Build the Graph ---

# Define the checkpointer using MongoDBSaver with synchronous pymongo client
checkpointer: Optional[BaseCheckpointSaver] = None
if MONGO_SAVER_AVAILABLE and MongoDBSaver: # Check if class was imported
    try:
        sync_mongo_client = MongoClient(MONGO_URI)
        sync_mongo_client.admin.command('ping')
        print(f"Successfully connected to MongoDB at {MONGO_URI}")
        # Initialize MongoDBSaver with the synchronous client
        checkpointer = MongoDBSaver(sync_mongo_client)
        print("Using MongoDB checkpointer (MongoDBSaver with sync client).")
    except Exception as e: print(f"Error initializing MongoDBSaver or connecting to MongoDB: {e}"); print("Checkpointer will not be used.")
else: print("Proceeding without a checkpointer.")


# Initialize the state graph using the Annotated GraphState
workflow = StateGraph(GraphState)

# --- Define Nodes ---
workflow.add_node("generate_questions", generate_questions_node)
workflow.add_node("fetch_openai_answers", fetch_openai_answers_node)
workflow.add_node("fetch_brave_search_answers", fetch_brave_search_answers_node)
workflow.add_node("fetch_tavily_answers", fetch_tavily_answers_node) # *** Add Tavily node ***
workflow.add_node("combine_and_summarize", combine_and_summarize_node)


# --- Define Edges ---

# 1. Set the entry point
workflow.set_entry_point("generate_questions")

# 2. *** Define edges for the two initial parallel branches ***
workflow.add_edge("generate_questions", "fetch_openai_answers")
workflow.add_edge("generate_questions", "fetch_brave_search_answers")

# 3. *** Define the sequential edge from Brave to Tavily ***
workflow.add_edge("fetch_brave_search_answers", "fetch_tavily_answers")

# 4. *** Define the join point (waits for OpenAI and Tavily) ***
workflow.add_edge("fetch_openai_answers", "combine_and_summarize")
workflow.add_edge("fetch_tavily_answers", "combine_and_summarize") # *** Edge from Tavily to Summarize ***

# 5. Define the final transition to the END state
workflow.add_edge("combine_and_summarize", END)

# Compile the graph into a runnable application
# Pass the checkpointer (which might be None)
compiled_app = workflow.compile(checkpointer=checkpointer)


# --- FastAPI Setup ---
api = FastAPI(title="LangGraph QA Summarization API", version="1.0.0")

class ProcessRequest(FastApiBaseModel):
    paragraph: str
    thread_id: Optional[str] = None

class ProcessResponse(FastApiBaseModel):
    thread_id: str
    summary: Optional[str] = None
    error: Optional[str] = None

# --- Existing /process endpoint (remains synchronous) ---
@api.post("/process", response_model=ProcessResponse)
def process_paragraph(request: ProcessRequest):
    print("-" * 30); print(f"Received request for paragraph: {request.paragraph[:100]}...")
    thread_id = request.thread_id if request.thread_id else str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    print(f"Processing with Thread ID: {thread_id}")
    final_state = None
    try:
        final_state = compiled_app.invoke({"paragraph": request.paragraph}, config=config)
    except Exception as e:
        print(f"\n--- An error occurred during graph execution (invoke) for thread {thread_id} ---"); print(f"Error: {e}"); traceback.print_exc()
        last_state = None
        if checkpointer:
            try: print("\nAttempting to retrieve last known state..."); last_state = compiled_app.get_state(config); print("Successfully retrieved last known state.")
            except Exception as state_e: print(f"Could not retrieve final state from checkpointer: {state_e}")
        raise HTTPException(status_code=500, detail={"message": f"An error occurred: {type(e).__name__}: {e}", "thread_id": thread_id, "last_known_state": last_state})
    if final_state:
        print(f"--- Final State for thread {thread_id} ---")
        response_error = final_state.get("error")
        response_summary = final_state.get("final_summary")
        if not response_error and not response_summary:
             if final_state.get("questions") is None: response_error = "Question extraction failed."
             elif not final_state.get("questions"): response_error = "No questions extracted."
             elif not final_state.get("openai_answers") and not final_state.get("search_answers") and not final_state.get("tavily_answers"): response_error = "All answer fetching steps failed." # Updated check
             else: response_error = "Processing finished, but no summary generated."
             print(f"Warning for thread {thread_id}: {response_error}")
        return ProcessResponse(thread_id=thread_id, summary=response_summary, error=response_error)
    else: print(f"Error: Final state was unexpectedly None for thread {thread_id}"); raise HTTPException(status_code=500, detail={"message": "Processing completed but no final state was returned.", "thread_id": thread_id})

# --- Background Task Function for WhatsApp Reply ---
def process_and_reply_whatsapp(user_number: str, message_body: str):
    """
    Calls the /process API and sends the result back via Twilio WhatsApp.
    Designed to be run as a background task.
    """
    # Check if Twilio client was initialized
    if not twilio_client:
        print("Error: Twilio client not initialized. Cannot send WhatsApp reply.")
        return

    print(f"Background task started for user: {user_number}")
    process_api_url = "http://localhost:8000/process" # Assuming API runs locally
    reply_message = "Sorry, something went wrong while processing your request." # Default error reply

    try:
        # Call the internal /process API
        payload = {"paragraph": message_body}
        response = requests.post(process_api_url, json=payload, timeout=300) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses

        api_response_data = response.json()
        summary = api_response_data.get("summary")
        error = api_response_data.get("error")

        if error:
            reply_message = f"Error processing your request: {error}"
        elif summary:
            reply_message = summary
        else:
            reply_message = "Processing finished, but couldn't generate a summary." # Fallback

    except requests.exceptions.RequestException as http_err:
        print(f"Error calling /process API: {http_err}")
        reply_message = "Sorry, there was an error connecting to the processing service."
    except Exception as e:
        print(f"Unexpected error during background processing: {e}")
        traceback.print_exc()
        # Keep the default error message

    # Send the reply via Twilio
    try:
        print(f"Sending reply to {user_number}: {reply_message[:100]}...")
        message = twilio_client.messages.create(
                              body=reply_message,
                              from_=TWILIO_WHATSAPP_NUMBER, # Your Twilio WhatsApp number
                              to=user_number # User's number in "whatsapp:+..." format
                          )
        print(f"Twilio message sent, SID: {message.sid}")
    except Exception as e:
        print(f"Error sending Twilio message to {user_number}: {e}")

# --- Twilio Webhook Endpoint ---
@api.post("/twilio/webhook")
async def twilio_webhook(background_tasks: BackgroundTasks, request: Request):
    """
    Handles incoming WhatsApp messages from Twilio.
    Acknowledges receipt immediately and processes the message in the background.
    """
    try:
        # Twilio sends data as form-encoded
        form_data = await request.form()
        message_body = form_data.get("Body")
        user_number = form_data.get("From") # This will be like "whatsapp:+1234567890"

        print("-" * 30)
        print(f"Received Twilio webhook from: {user_number}")
        print(f"Message Body: {message_body}")

        if not message_body or not user_number:
            print("Webhook received without Body or From field. Ignoring.")
            # Still return empty TwiML to acknowledge
            twiml_response = MessagingResponse()
            # *** Use Response object from FastAPI ***
            return Response(content=str(twiml_response), media_type="application/xml")

        # Add the processing and reply logic to background tasks
        background_tasks.add_task(process_and_reply_whatsapp, user_number, message_body)

        # Respond immediately to Twilio to acknowledge receipt
        twiml_response = MessagingResponse()
        # Optionally send a quick ack message back immediately (costs $)
        # twiml_response.message("Processing your request...")
        # For now, just send empty response to prevent timeout
        # *** Use Response object from FastAPI ***
        return Response(content=str(twiml_response), media_type="application/xml")

    except Exception as e:
        print(f"Error processing Twilio webhook: {e}")
        traceback.print_exc()
        # Try to return an empty response even on error to satisfy Twilio
        twiml_response = MessagingResponse()
        # *** Use Response object from FastAPI ***
        return Response(content=str(twiml_response), media_type="application/xml")


# --- Run the Application (using uvicorn) ---
if __name__ == "__main__":
    print("--- Starting FastAPI Server ---")
    print("Graph compiled. Ready to accept requests.")
    print("Ensure MongoDB is running and accessible at:", MONGO_URI)
    if not MONGO_SAVER_AVAILABLE or checkpointer is None:
        print("WARNING: MongoDB checkpointer is NOT configured or failed to initialize. State will not persist.")
    else:
        print("MongoDB checkpointer configured successfully.")
    print("Run with: uvicorn app:api --reload --host 0.0.0.0 --port 8000")
    pass

