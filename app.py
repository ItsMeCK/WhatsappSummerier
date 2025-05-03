# app.py
import os
import json
import uuid
import time # Import time module for sleep
import requests # For Brave Search API call
from dotenv import load_dotenv
from typing import List, Dict, TypedDict, Annotated, Sequence, Optional, Any
import operator # Needed for the Annotated state definition
from concurrent.futures import ThreadPoolExecutor

# LangChain components
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import BaseTool # To create the custom Brave Search tool

# LangGraph components
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.mongodb import MongoDBSaver # Ideal target - check for availability or implement custom
from langgraph.checkpoint.sqlite import SqliteSaver # Using SQLite as a runnable default/fallback
from langgraph.checkpoint.base import BaseCheckpointSaver # Needed for custom saver type hinting

# --- Configuration & API Keys ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY") # Ensure this is named correctly in your .env
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/") # Default for local Docker setup
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "langgraph_db")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "qa_workflows")

# Optional: Add MONGO_USER and MONGO_PASSWORD if you set them in docker-compose.yml
# MONGO_USER = os.getenv("MONGO_USER")
# MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
# if MONGO_USER and MONGO_PASSWORD:
#     MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@localhost:27017/"


if not OPENAI_API_KEY or not BRAVE_API_KEY:
    raise ValueError("API keys for OpenAI (OPENAI_API_KEY) and Brave Search (BRAVE_SEARCH_API_KEY) must be set in .env file")

# --- Model Definitions ---
# Use environment variables or specify models directly
QUESTION_MODEL_NAME = os.getenv("OPENAI_QUESTION_MODEL", "gpt-3.5-turbo")
ANSWER_MODEL_NAME = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4-turbo-preview")
SUMMARY_MODEL_NAME = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4-turbo-preview")

llm_question_gen = ChatOpenAI(model=QUESTION_MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
# *** Add max_tokens parameter to limit answer length ***
llm_answer = ChatOpenAI(model=ANSWER_MODEL_NAME, temperature=0.2, api_key=OPENAI_API_KEY, max_tokens=200)
llm_summarize = ChatOpenAI(model=SUMMARY_MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY, max_tokens=200)

# --- Custom Brave Search Tool ---
class BraveSearchTool(BaseTool):
    """Tool that queries the Brave Search API."""
    name: str = "brave_search"
    description: str = (
        "A wrapper around Brave Search API. "
        "Useful for when you need to answer questions about current events or real-time information. "
        "Input should be a search query."
    )
    api_key: str = Field(default=BRAVE_API_KEY)
    base_url: str = "https://api.search.brave.com/res/v1/web/search"
    results_count: int = 3 # Number of results to fetch

    def _run(
        self, query: str, run_manager: Optional[Any] = None # Compatibility with LangChain run managers
    ) -> str:
        """Use the tool."""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {"q": query, "count": self.results_count}

        try:
            # *** Add time.sleep(1) before the API call ***
            # print(f"Brave Tool: Sleeping for 1 second before searching for '{query[:50]}...'") # Optional debug print
            # time.sleep(1) # Moved the sleep inside the node function for better control with parallel calls

            response = requests.get(self.base_url, headers=headers, params=params, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            # Extract relevant information (e.g., snippets or titles/links)
            # Adjust based on the actual Brave API response structure
            snippets = []
            if "web" in data and "results" in data["web"]:
                for result in data["web"]["results"]:
                    snippet = result.get("description", result.get("title", ""))
                    if snippet:
                        snippets.append(snippet)

            if not snippets:
                return "No relevant information found by Brave Search."

            return "\n".join(snippets) # Combine snippets into a single string

        except requests.exceptions.RequestException as e:
            print(f"Error calling Brave Search API: {e}")
            return f"Error: Could not retrieve search results from Brave Search ({e})"
        except Exception as e:
            print(f"Error processing Brave Search results: {e}")
            return f"Error: Could not process search results ({e})"

    async def _arun(
        self, query: str, run_manager: Optional[Any] = None
    ) -> str:
        # Brave Search API doesn't have an official async client, so we run sync in thread
        # For a production system, consider using httpx for async requests
        import asyncio
        # *** Add sleep here too for async calls ***
        # print(f"Brave Tool (async): Sleeping for 1 second before searching for '{query[:50]}...'") # Optional debug print
        # await asyncio.sleep(1) # Moved the sleep inside the node function
        return await asyncio.to_thread(self._run, query, run_manager)

# Instantiate the tool
search_tool = BraveSearchTool()

# --- Output Parsers & Data Structures ---
class Questions(BaseModel):
    """Schema for the list of questions generated from the paragraph."""
    questions: List[str] = Field(description="A list of distinct questions extracted directly from the input paragraph.")

# --- Graph State Definition ---
# *** Use Annotated state as confirmed working ***
class GraphState(TypedDict):
    """
    Represents the state of our graph using Annotated fields
    where required by LangGraph for branching.
    """
    # Input field
    paragraph: str

    # Annotating 'questions' satisfies LangGraph's requirement for multiple outgoing edges.
    # operator.add is used here, though default replacement (Any) might also work if
    # this was the *only* annotated field needed just to satisfy the check.
    questions: Annotated[Optional[List[str]], operator.add]

    # Other fields updated by graph nodes - explicit annotation not strictly needed
    # for their update logic in this graph, but can be added for consistency if preferred.
    openai_answers: Optional[Dict[str, str]]
    search_answers: Optional[Dict[str, str]]
    final_summary: Optional[str]
    error: Optional[str]


# --- Node Functions ---
# Node functions accept a standard dictionary (representing the state)
# and return a dictionary containing updates.

def generate_questions_node(state: Dict) -> Dict[str, Any]:
    """
    Extracts sentences formatted as questions directly from the input paragraph.
    """
    print("--- Node: Extracting Questions ---") # Renamed print statement for clarity
    paragraph = state.get('paragraph')
    if not paragraph:
        print("Error: Input paragraph is missing.")
        return {"error": "Input paragraph is missing."}

    # *** Use f-string for HumanMessage content ***
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an expert text analysis tool. Your task is to carefully read the provided paragraph and identify any sentences that are phrased as questions (e.g., ending with a question mark '?'). Extract these questions exactly as they appear in the text. If no questions are found in the paragraph, return an empty list. Output *only* the list of extracted questions in the specified format."),
        HumanMessage(content=f"Here is the paragraph:\n\n{paragraph}\n\nPlease extract any questions present in this text.")
    ])

    # Using structured output to ensure we get a list
    structured_llm_question_gen = llm_question_gen.with_structured_output(Questions)
    question_generation_chain = prompt | structured_llm_question_gen

    try:
        result = question_generation_chain.invoke({"paragraph": paragraph})
        # Ensure questions is always a list, even if result is None or questions attribute is missing
        questions = getattr(result, 'questions', []) if result else []

        if not questions:
             print("Warning: No questions found in the paragraph.")
             # Return empty list for 'questions'
             return {"questions": [], "error": None}
        print(f"Extracted Questions: {questions}")
        # Return the list which will be handled by operator.add
        return {"questions": questions, "error": None}
    except Exception as e:
        print(f"Error extracting questions: {e}")
        # Ensure questions is set to None if an error occurs
        return {"error": f"Failed to extract questions: {e}", "questions": None}


def fetch_openai_answers_node(state: Dict) -> Dict[str, Any]:
    """Fetches answers to questions using an OpenAI model in parallel (limited to 200 tokens)."""
    print("--- Node: Fetching OpenAI Answers ---")
    questions = state.get('questions')
    # Check if questions is None or empty list
    if not questions:
        print("Skipping OpenAI answers: No questions available.")
        return {"openai_answers": {}} # Return empty dict if no questions

    # Note: This prompt template doesn't use HumanMessage directly, so no f-string needed here
    # The max_tokens limit is set on the llm_answer instance itself.
    prompt_template = PromptTemplate.from_template("Based on general knowledge, please answer the following question concisely:\nQuestion: {question}\nAnswer:")
    answer_chain = prompt_template | llm_answer

    results = {}
    def get_answer(question: str) -> str:
        try:
            # The variable 'question' is passed via invoke's dictionary, not directly in message content
            response = answer_chain.invoke({"question": question})
            return response.content
        except Exception as e:
            print(f"Error getting OpenAI answer for '{question}': {e}")
            return f"Error: Could not get answer ({e})"

    # Use ThreadPoolExecutor for parallel execution within the node
    with ThreadPoolExecutor(max_workers=5) as executor: # Limit workers if needed
        answers = list(executor.map(get_answer, questions))

    results = dict(zip(questions, answers))

    print(f"OpenAI Answers fetched: {len(results)} answers.")
    # Return the dict which will replace the existing 'openai_answers' in the state
    return {"openai_answers": results}

def fetch_brave_search_answers_node(state: Dict) -> Dict[str, Any]:
    """Fetches answers/snippets for questions using the Brave Search tool in parallel with delay."""
    print("--- Node: Fetching Brave Search Answers ---")
    questions = state.get('questions')
    # Check if questions is None or empty list
    if not questions:
        print("Skipping Brave Search answers: No questions available.")
        return {"search_answers": {}} # Return empty dict if no questions

    results = {}
    def get_search_result(question: str) -> str:
        try:
            # *** Add 1-second sleep before each Brave Search call ***
            print(f"Brave Node: Sleeping 1 second before searching for '{question[:50]}...'") # Optional debug print
            time.sleep(1)
            # Invoke the custom Brave Search tool
            search_results_str = search_tool.invoke(question)
            return search_results_str
        except Exception as e:
            print(f"Error getting Brave Search results for '{question}': {e}")
            return f"Error: Could not get search results ({e})"

    # Use ThreadPoolExecutor for parallel execution within the node
    # The sleep happens *inside* each thread before its API call.
    with ThreadPoolExecutor(max_workers=5) as executor: # Limit workers if needed
        search_data = list(executor.map(get_search_result, questions))

    results = dict(zip(questions, search_data))

    print(f"Brave Search Answers fetched: {len(results)} results.")
    # Return the dict which will replace the existing 'search_answers' in the state
    return {"search_answers": results}


def combine_and_summarize_node(state: Dict) -> Dict[str, Any]:
    """Combines answers and generates a final summary."""
    print("--- Node: Combining and Summarizing ---")
    questions = state.get('questions')
    openai_answers = state.get('openai_answers')
    search_answers = state.get('search_answers')
    paragraph = state.get('paragraph')

    # Check if questions list exists and is not empty
    if not questions: # Handles None or empty list
         err_msg = "Cannot summarize: No questions were extracted or available."
         print(f"Warning: {err_msg}")
         return {"final_summary": err_msg, "error": None}

    # Handle cases where answer dictionaries might be None (if preceding nodes skipped/errored)
    if openai_answers is None:
        print("Warning: OpenAI answers missing for summarization.")
        openai_answers = {} # Use empty dict to avoid errors later
    if search_answers is None:
        print("Warning: Brave Search answers missing for summarization.")
        search_answers = {} # Use empty dict

    # Format the combined information for the summarizer LLM
    combined_info = f"Original Paragraph:\n{paragraph}\n\n"
    combined_info += "Based on the paragraph, the following questions were extracted and answered using AI and Brave Search:\n\n" # Updated description

    for q in questions:
        combined_info += f"Question: {q}\n"
        combined_info += f"  - Answer from AI: {openai_answers.get(q, 'N/A')}\n"
        combined_info += f"  - Information from Brave Search: {search_answers.get(q, 'N/A')}\n\n"

    # *** Use f-string for HumanMessage content ***
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an expert summarizer. Synthesize the information provided below, which includes an original paragraph, questions extracted from it, answers generated by an AI, and relevant snippets from Brave web search. Create a concise and comprehensive summary based on ALL the provided context. Focus on integrating the findings accurately."), # Updated description
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

# Define the checkpointer
checkpointer = SqliteSaver.from_conn_string("qa_summarizer_checkpoints.sqlite")
# (MongoDB checkpointer setup remains commented out)
# ...

# Initialize the state graph using the Annotated GraphState
workflow = StateGraph(GraphState)

# --- Define Nodes ---
# Nodes are added with their names and corresponding functions
workflow.add_node("generate_questions", generate_questions_node) # Function name unchanged, but behavior is now extraction
workflow.add_node("fetch_openai_answers", fetch_openai_answers_node)
workflow.add_node("fetch_brave_search_answers", fetch_brave_search_answers_node)
workflow.add_node("combine_and_summarize", combine_and_summarize_node)


# --- Define Edges ---
# Using the standard parallel execution definition, which works with the Annotated state

# 1. Set the entry point
workflow.set_entry_point("generate_questions")

# 2. Define edges for parallel branches FROM the starting node
# LangGraph uses the annotation(s) on GraphState to handle these multiple outgoing edges.
workflow.add_edge("generate_questions", "fetch_openai_answers")
workflow.add_edge("generate_questions", "fetch_brave_search_answers")

# 3. Define the join point
# Edges from EACH parallel branch to the joining node.
# LangGraph waits for both predecessors before executing 'combine_and_summarize'.
workflow.add_edge("fetch_openai_answers", "combine_and_summarize")
workflow.add_edge("fetch_brave_search_answers", "combine_and_summarize")

# 4. Define the final transition to the END state
workflow.add_edge("combine_and_summarize", END)

# Compile the graph into a runnable application
app = workflow.compile(checkpointer=checkpointer)


# --- Run the Application ---

if __name__ == "__main__":
    print("--- QA Summarization Application ---")
    print("Powered by LangGraph, OpenAI, and Brave Search")
    print("State managed with checkpointer (see configuration above).")
    print("-" * 30)

    user_paragraph = input("Please enter the paragraph you want to process:\n> ")

    if not user_paragraph.strip():
        print("Error: No paragraph provided. Exiting.")
    else:
        # Create a unique ID for this conversation/run thread.
        thread_id = str(uuid.uuid4())
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        print(f"\nProcessing paragraph with Thread ID: {thread_id}")
        print("Checkpoints will be saved using the configured saver.")
        print("-" * 30)

        final_state = None
        try:
            # Invoke the graph.
            # The final_state returned by invoke will be a dictionary.
            final_state = app.invoke({"paragraph": user_paragraph}, config=config)

        except Exception as e:
            print(f"\n--- An error occurred during graph execution ---")
            print(f"Error: {e}")
            # Attempt to retrieve the last known state from the checkpointer
            try:
                 print("\nAttempting to retrieve last known state...")
                 # get_state returns the internal state dictionary
                 retrieved_state_dict = app.get_state(config)
                 print("--- Last Known State (from checkpointer) ---")
                 print(json.dumps(retrieved_state_dict, indent=2, default=str))
            except Exception as state_e:
                 print(f"Could not retrieve final state from checkpointer: {state_e}")


        if final_state:
            print("\n" + "-" * 30)
            print("--- Final Result ---")
            # Access final_state as a dictionary
            if final_state.get("error"):
                print(f"Processing finished with an error: {final_state['error']}")
            elif final_state.get("final_summary"):
                print("\nSummary:")
                print(final_state["final_summary"])
            else:
                 # Check specific conditions for why summary might be missing
                if final_state.get("questions") is None:
                     print("Processing finished. Question extraction likely failed.")
                elif not final_state.get("questions"): # Check for empty list
                     print("Processing finished. No questions were extracted from the paragraph.")
                elif not final_state.get("openai_answers") and not final_state.get("search_answers"):
                     print("Processing finished. Both answer fetching steps seem to have failed or returned no results.")
                else:
                     print("Processing finished, but no final summary was generated (check logs/state).")


            # Optionally print the full final state for debugging
            # print("\n--- Full Final State ---")
            # print(json.dumps(final_state, indent=2, default=str))

        print("\n" + "-" * 30)
        print("Application finished.")
