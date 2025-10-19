# %%
import asyncio
import json
import re
import time
from enum import Enum
from typing import Any, Dict, List, TypedDict

import pandas as pd
from IPython.display import Image, display
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langgraph.graph import END, START, StateGraph

# %%
query_improver_prompt = PromptTemplate.from_template("""
You are an expert in refining and proofreading user queries to clearly express the user's intent.

Task:
Given the following user query:
{query}

Rewrite the query so that it:
- Is grammatically correct and unambiguous.
- Accurately reflects the user's original intent.
- Uses concise, natural language suitable for search or retrieval systems.
- Does not add or remove meaning beyond what is provided.

Return only the improved query text.
""")

query_translator_prompt = PromptTemplate.from_template("""
You are an expert in interpreting user intent and converting natural language queries into structured JSON.

Task:
Given the user query:
{query}

Generate a JSON object capturing the intent:
{{
  "category": "",
  "use_case": "",
  "key_attributes": ""
}}

Rules:
- Return ONLY valid JSON.
- All fields must be strings.
- Leave unclear fields as empty strings.
""")

translator_prompt = PromptTemplate.from_template("""
You are an expert in converting user queries into concise marketplace search queries.

Task:
Given the user query:
{query}

Generate a **direct, compact search string** suitable for a URL query. The format should be:
- Include only the main category and key attributes.
- Use '+' to separate words and numbers.
- Remove filler words, prepositions, or extra punctuation.
- Make it short, direct, and search-engine friendly.

Output Example:
laptop+Core+i7+16GB+RTX+5070+16GB+VRAM

Rules:
- Return ONLY the compact search string.
- Do NOT output JSON or explanations.
""")


query_ranker_prompt = PromptTemplate.from_template("""
You are an expert AI agent for ranking improved queries based on a primary user query.

Task:
- Primary user query: {primary_query}
- Improved query: {improved_query}
- Model: {model_name}

Instructions:
1. Evaluate alignment with the primary query.
2. Consider correctness, clarity, and relevance.
3. Assign a decimal score from 1.0 to 5.0.
4. Do NOT modify the improved query.

Return JSON:
{{
    "{{model_name}}": {{"improved_query": "{{improved_query}}", "score": <decimal score>}}
}}
""")


# %%
class ModelID(Enum):
    """Supported AI model IDs."""

    gemma: str = "gemma3:latest"
    deepseek: str = "deepseek-r1:latest"
    mistral: str = "mistral:7b"
    llama: str = "llama3:latest"
    qwen3: str = "qwen3:latest"

    @classmethod
    def as_dict(cls) -> Dict:
        """Return model IDs as a dictionary."""
        return {m.name: m.value for m in cls}


# %%
str_query = (
    "I need to find a speakers that are high quality priced as low as 3000 rands."
)


def clean_response(text: str) -> str:
    """Clean LLM response using regex."""
    text = str(text)
    text = re.sub(r'^[\'"]|[\'"]$', "", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def model_chain(model_id: str):
    """Create a model chain with prompt instructions."""
    llm = OllamaLLM(model=model_id, temperature=0.1)
    return query_improver_prompt | llm


async def run_model(name: str, model_id: str, query: str) -> tuple[str, str]:
    """Run a model asynchronously and return cleaned output."""
    print(f"\n--- Running model: {name} ({model_id}) ---")
    try:
        chain = model_chain(model_id)
        response = await asyncio.to_thread(chain.invoke, {"query": query})
        return name, clean_response(response)
    except Exception as e:
        print(f"Error running {name}: {e}")
        return name, f"Error: {e}"


async def create_optimise_query(query: str) -> Dict[str, Dict[str, str]]:
    """Run multiple AI models concurrently and record execution time."""

    async def timed_run(name, model_id, query):
        start = time.perf_counter()
        name, result = await run_model(name, model_id, query)
        duration = time.perf_counter() - start
        return name, {"improved_query": result, "time": f"{duration:.2f}s"}

    tasks = [
        timed_run(name, model_id, query) for name, model_id in ModelID.as_dict().items()
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)


# %%
try:
    result = await create_optimise_query(str_query)
except RuntimeError:
    result = asyncio.run(create_optimise_query(str_query))


# %%
def model_output_response(output: str) -> Dict:
    """Clean markdown and parse JSON response."""
    cleaned = re.sub(r"^```json\s*", "", output)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    return json.loads(cleaned)


llm_supervisor = OllamaLLM(model=ModelID.deepseek.value, temperature=0.1)
chain = query_ranker_prompt | llm_supervisor


async def rank_model(name: str, improved_query: str) -> tuple[str, float]:
    """Run the supervisor LLM to rank a single improved query."""
    print(f"Running model: [{name}]")
    response = await asyncio.to_thread(
        chain.invoke,
        {
            "primary_query": str_query,
            "improved_query": improved_query,
            "model_name": name,
        },
    )
    json_response = model_output_response(response)
    score = json_response[name]["score"]
    return name, score


async def rank_all_models(result: Dict) -> Dict:
    """Rank all models concurrently and update the result with scores."""
    tasks = [rank_model(name, data["improved_query"]) for name, data in result.items()]
    ranked = await asyncio.gather(*tasks)
    for name, score in ranked:
        result[name]["score"] = score
    return result


# %%
# Run the async ranking
try:
    final_result = await rank_all_models(result)
except RuntimeError:
    final_result = asyncio.run(rank_all_models(result))

final_result

# %%

print(json.dumps(final_result, indent=3))

# %%
df = pd.DataFrame.from_dict(final_result, orient="index")
df["time"] = df["time"].str.replace("s", "").astype(float)
df_sorted = df.sort_values(by=["score", "time"], ascending=[False, True])

df_sorted = df_sorted.reset_index().rename(columns={"index": "model"})

display(df_sorted)
highest_ranked = df_sorted.iloc[0]
# display(highest_ranked)

# %%
model_id = ModelID.deepseek.value
improved_query = highest_ranked.improved_query

llm_translator = OllamaLLM(model=model_id, temperature=0.1)
# chain = query_translator_prompt | llm_translator
chain = translator_prompt | llm_translator

# Invoke the chain
response = chain.invoke({"query": improved_query})

# Parse the JSON safely
# try:
#     translated_query = json.loads(response)
# except json.JSONDecodeError:
#     print("Failed to parse JSON:", response)
#     translated_query = {}

print(response)


# %%
# %%
class MarketState(TypedDict):
    """Represents the state of the marketplace query and response process."""

    query: str
    polished_query: str
    search_query: Dict[Any, str]
    web_url: str
    results: List[str]
    retries: int  # Tracks how many times the crawler has retried


# %%
def query_builder(state: MarketState) -> MarketState:
    """Proofreads and clarifies the user's query to determine its primary intent."""
    print("Building user query...")

    return state


def translate_user_query(state: MarketState) -> MarketState:
    """Translates the user's query into a JSON format compatible with marketplace search queries."""
    print("Translating user query...")
    return state


def product_crawler(state: MarketState) -> MarketState:
    """Crawls or scrapes products based on the user's intent derived from the translated query state."""
    print("Crawling marketplace...")

    if not state.get("results"):
        print("No products were found.")
    else:
        print(f"Found {len(state['results'])} products.")

    # Increment retry count
    state["retries"] = state.get("retries", 0) + 1
    return state


def crawler_decision(state: MarketState) -> str:
    """
    Determines the next step after crawling.

    Returns:
        "repeat"      -> Re-crawl if no search query is found.
        "retry_query" -> Go back to query building if no results were found.
        "stop"        -> End process when crawling is successful or retry limit reached.
    """
    print(f"Retry attempt: {state.get('retries', 0)}")

    # Stop after 3 retries to prevent infinite looping
    if state.get("retries", 0) >= 3:
        print("Maximum retry limit reached. Stopping process.")
        return "stop"

    if not state.get("search_query"):
        print("Missing search query. Repeating crawl...")
        return "repeat"

    if not state.get("results"):
        print("No results found. Retrying from query builder...")
        return "retry_query"

    print("Successful crawl. Ending process.")
    return "stop"


# %%


# Build and compile the state graph for marketplace query processing
graph_builder = StateGraph(MarketState)

# Add nodes representing the main processing stages
graph_builder.add_node("query_builder", query_builder)
graph_builder.add_node("translate_query", translate_user_query)
graph_builder.add_node("crawler", product_crawler)

# Define the workflow sequence through directed edges
graph_builder.add_edge(START, "query_builder")
graph_builder.add_edge("query_builder", "translate_query")
graph_builder.add_edge("translate_query", "crawler")

# Add conditional looping logic with retry handling
graph_builder.add_conditional_edges(
    "crawler",
    crawler_decision,
    {
        "repeat": "crawler",  # Retry crawling directly
        "retry_query": "query_builder",  # Rebuild query if no results
        "stop": END,  # End process safely
    },
)

# Compile the graph to produce the final executable workflow
graph = graph_builder.compile()

# Visualise the graph

display(Image(graph.get_graph().draw_mermaid_png()))

# %%
str_query = (
    "find a laptop which is core i7 and 16gb ram and a gpu of 5070 RTX 16gb vram."
)

graph.invoke({"query": str_query, "web_url": "https://www.evetech.co.za"})

# %%
# %%
