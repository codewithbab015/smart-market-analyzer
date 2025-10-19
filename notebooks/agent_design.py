# %%
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Tuple, TypedDict
from urllib.parse import urlencode

import pandas as pd
from IPython.display import Image, display
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langgraph.graph import END, START, StateGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# %%
REFINE_QUERY_PROMPT = PromptTemplate.from_template("""
You are an expert query refiner that improves user queries for clarity and precision.

Task:
Given the user query:
{query}

Rewrite it so that it:
- Is grammatically correct and unambiguous.
- Accurately reflects the user's original intent.
- Uses concise, natural language suitable for search or retrieval systems.
- Does not add or remove meaning beyond what is provided.

Output rules:
- Return ONLY the improved query as plain text.
- Do NOT include explanations, quotes, or additional formatting.
""")

COMPACT_SEARCH_PROMPT = PromptTemplate.from_template("""
You are an expert in converting any user query—whether a question, statement, or incomplete phrase—
into a concise, URL-friendly search string for marketplace platforms (e.g., Amazon, eBay, Newegg).

Your task is to translate the input into a direct, minimal search phrase suitable for a web search URL.

Input:
{query}

Output requirements:
1. Always convert the user input into a compact search phrase, even if it is a question, command, or statement.
   - Example: "Can you suggest a good gaming laptop?" → "gaming+laptop"
   - Example: "Looking for laptops with GPUs for AI." → "laptop+GPU+AI"
2. Use only information and keywords that appear in, or are clearly implied by, the input.
3. Do NOT add or invent new products, models, years, or specifications.
4. Include only the main category and relevant attributes.
5. Use '+' to separate words and numbers.
6. Remove filler words, prepositions, and punctuation.
7. Output must contain only the search string — no quotes, brackets, explanations, or commentary.

If the query is vague or incomplete:
- Keep only the most general category and main descriptive terms.
- Do not guess or infer extra information.

Example:
User query: "Find a laptop that can run AI models and high-end games."
Output: laptop+AI+model+training+gaming
""")


RANK_QUERY_PROMPT = PromptTemplate.from_template("""
You are an expert evaluator that ranks improved queries against the primary user query.

Inputs:
- Primary user query: {primary_query}
- Improved query: {improved_query}
- Model name: {model_name}

Instructions:
1. Assess how well the improved query aligns with the primary query.
2. Consider clarity, correctness, and contextual relevance.
3. Assign a decimal score between 1.0 and 5.0 (higher is better).
4. Do NOT alter the improved query.

Return JSON in this format only:
{{
  "{model_name}": {{
    "improved_query": "{improved_query}",
    "score": <decimal score>
  }}
}}
""")


# %%
class ModelID(Enum):
    """Supported AI model IDs."""

    # gemma: str = "gemma3:latest"
    # deepseek: str = "deepseek-r1:latest"
    mistral: str = "mistral:7b"
    llama: str = "llama3:latest"
    # qwen3: str = "qwen3:latest"

    @classmethod
    def as_dict(cls) -> Dict[str, str]:
        """Return model IDs as a dictionary."""
        return {m.name: m.value for m in cls}


@dataclass
class QueryBuilder:
    """
    A utility for improving and optimising user queries using multiple LLMs.

    Example:
        qb = QueryBuilder(query="Find high-quality speakers under 3000 rands.")
        result = await qb.create_optimise_query()
    """

    query: str

    def _clean_response(self, text: str) -> str:
        """Clean and normalise LLM response output."""
        if not text:
            return ""
        text = str(text)
        text = re.sub(r'^[\'"]|[\'"]$', "", text.strip())
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?-]", "", text)
        return text.strip()

    @staticmethod
    @lru_cache(maxsize=None)
    def _model_chain(model_id: str):
        """Cache and create a model chain with prompt instructions."""
        llm = OllamaLLM(model=model_id, temperature=0.1)
        return REFINE_QUERY_PROMPT | llm

    async def _run_model(self, name: str, model_id: str, query: str) -> Tuple[str, str]:
        """Run a model asynchronously and return cleaned output."""
        logger.info(f"Running model: {name} ({model_id})")
        try:
            chain = self._model_chain(model_id)
            response = await asyncio.to_thread(chain.invoke, {"query": query})
            return name, self._clean_response(response)
        except Exception as e:
            logger.error(f"Error running {name} ({model_id}): {type(e).__name__} - {e}")
            return name, f"Error: {type(e).__name__} - {e}"

    async def create_optimise_query(self) -> Dict[str, Dict[str, str]]:
        """Run multiple AI models concurrently, record execution time, and handle errors."""

        async def timed_run(name: str, model_id: str, query: str):
            start = time.perf_counter()
            name, result = await self._run_model(name, model_id, query)
            duration = time.perf_counter() - start
            logger.info(f"{name} completed in {duration:.2f}s")
            return name, {"improved_query": result, "time": f"{duration:.2f}s"}

        tasks = []
        for name, model_id in ModelID.as_dict().items():
            if not model_id:
                logger.warning(f"Skipping {name}: No model_id provided.")
                continue
            tasks.append(asyncio.create_task(timed_run(name, model_id, self.query)))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any model failures gracefully
        final_results = {}
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Model execution failed: {res}")
                continue
            name, data = res
            final_results[name] = data

        return final_results


# %%
@dataclass
class RankQuery:
    """
    Rank improved queries from multiple models using a supervisor LLM.
    """

    model_id: str = "deepseek-r1:latest"

    def __post_init__(self):
        # Initialize the supervisor LLM and chain at instance level
        self.llm_supervisor = OllamaLLM(model=self.model_id, temperature=0.1)
        self.chain = RANK_QUERY_PROMPT | self.llm_supervisor

    def _model_output_response(self, output: str) -> Dict[str, Any]:
        """Clean markdown formatting and parse JSON response safely."""
        cleaned = re.sub(r"^```json\s*", "", output)
        cleaned = re.sub(r"```$", "", cleaned.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM JSON output: %s", output)
            return {}

    async def _rank_model(
        self, name: str, primary_query: str, improved_query: str
    ) -> Tuple[str, float]:
        """Run the supervisor LLM to rank a single improved query."""
        logger.info(f"Running model: [{name}]")
        response = await asyncio.to_thread(
            self.chain.invoke,
            {
                "primary_query": primary_query,
                "improved_query": improved_query,
                "model_name": name,
            },
        )
        json_response = self._model_output_response(response)
        score = json_response.get(name, {}).get("score", 0.0)
        return name, score

    async def rank_all_models(
        self, result: Dict[str, Dict[str, Any]], primary_query: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Rank all models concurrently and update the result dictionary with scores.

        Args:
            result: Dictionary of model results, each containing 'improved_query'.
            primary_query: The original user query for reference.
        """
        tasks = [
            self._rank_model(name, primary_query, data["improved_query"])
            for name, data in result.items()
        ]
        ranked = await asyncio.gather(*tasks, return_exceptions=True)

        # Update results safely
        for item in ranked:
            if isinstance(item, Exception):
                logger.error("Ranking task failed: %s", item)
                continue
            name, score = item
            result[name]["score"] = score

        return result


def best_polished_query(query_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Select the best query by highest score and lowest time."""
    if not query_dict:
        raise ValueError("query_dict is empty.")

    df = pd.DataFrame.from_dict(query_dict, orient="index")
    if not {"score", "time", "improved_query"}.issubset(df.columns):
        raise ValueError(
            "Each entry must include 'score', 'time', and 'improved_query'."
        )

    df["time"] = df["time"].astype(str).str.replace("s", "", regex=False).astype(float)
    df_sorted = df.sort_values(
        by=["score", "time"], ascending=[False, True]
    ).reset_index()
    df_sorted.rename(columns={"index": "model"}, inplace=True)

    best = df_sorted.iloc[0]
    return {
        "model": best["model"],
        "query": best["improved_query"],
        "score": float(best["score"]),
        "time": float(best["time"]),
    }


# %%
@dataclass
class QueryTranslator:
    """
    Translates and refines a user query into a search-ready query
    suitable for e-commerce marketplaces like Amazon or Takealot.
    """

    model_id: str = "deepseek-r1:latest"
    temperature: float = 0.1

    def __post_init__(self):
        self.llm = OllamaLLM(model=self.model_id, temperature=self.temperature)
        self.chain = COMPACT_SEARCH_PROMPT | self.llm

    def show_translated_query(self, query: str) -> str:
        """Translate or refine the given user query."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        response: str = self.chain.invoke({"query": query})
        return response.strip()


# %%

# 1. Build polished query
user_query = "I am looking for a gaming laptop that is capable of running LLMs."
builder = QueryBuilder(query=user_query)
result = await builder.create_optimise_query()

# %%
result

# %%
# 2. Rank best query from multiple LLM agents
ranker = RankQuery()
ranked = await ranker.rank_all_models(result, primary_query=user_query)

ranked
# %%
# 3. Select highest-ranked query
best = best_polished_query(ranked)
print(best)

# %%
# 4. Translate query for e-commerce search
translator = QueryTranslator()
final_query = translator.show_translated_query(best["query"])

print(final_query)

# %%

# Given inputs
search_query = "gaming+laptop+llm+gpu"
web_url = "https://www.takealot.com/all?&qsearch="
search_query_url = f"{web_url}{urlencode({'': final_query})[1:]}"

print("Full Search URL:", search_query_url)


# %%
class MarketplaceState(TypedDict):
    """Represents the state of the marketplace query and response process."""

    query: str
    ranked_queries: Dict[str, str]
    web_url: str
    search_url: str
    results: List[str]
    retries: int


async def refine_marketplace_query(state: MarketplaceState) -> MarketplaceState:
    """
    Refines the user's raw query and ranks related candidate queries.
    """
    logger.info("Refining marketplace query...")

    try:
        builder = QueryBuilder(query=state["query"])
        result = await builder.create_optimise_query()

        ranker = RankQuery()
        ranked = await ranker.rank_all_models(result, primary_query=state["query"])

        state["ranked_queries"] = ranked
        state["retries"] = 0  # reset retries after successful query refinement
    except Exception as e:
        logger.error(f"Query refinement failed: {e}")
        state["retries"] = state.get("retries", 0) + 1

    return state


def generate_search_url(state: MarketplaceState) -> MarketplaceState:
    """
    Converts the top-ranked query into a marketplace search URL.
    """
    logger.info("Generating marketplace search URL...")

    try:
        best_query = best_polished_query(state["ranked_queries"])
        translator = QueryTranslator()
        translated_query = translator.show_translated_query(best_query["query"])

        search_query_url = f"{state['web_url']}{urlencode({'': translated_query})[1:]}"

        state["search_url"] = search_query_url

        logger.info(f"Search URL generated: {state['search_url']}")
    except Exception as e:
        logger.error(f"Search URL generation failed: {e}")
        state["retries"] = state.get("retries", 0) + 1

    return state


async def product_crawler(state: MarketplaceState) -> MarketplaceState:
    """
    Crawls marketplace products based on the translated query URL.
    """
    logger.info("Crawling marketplace...")

    try:
        search_url = state.get("search_url")
        if not search_url:
            raise ValueError("Missing search URL.")

        # TODO: 🧩 Placeholder for future async crawl logic
        # Example: results = await crawl_products(search_url)
        results = [f"Sample product from {search_url}"]

        state["results"] = results
        logger.info(f"Found {len(results)} products.")
    except Exception as e:
        logger.error(f"Crawling failed: {e}")
        state["retries"] = state.get("retries", 0) + 1

    return state


def crawler_decision(state: MarketplaceState) -> str:
    """
    Determines the next workflow step after crawling.
    """
    logger.info(f"Retry attempt: {state.get('retries', 0)}")

    if state.get("retries", 0) >= 3:
        logger.warning("Maximum retry limit reached. Stopping process.")
        return "stop"

    if not state.get("search_url"):
        logger.warning("Missing search URL. Repeating crawl...")
        return "repeat"

    if not state.get("results"):
        logger.warning("No results found. Retrying query build...")
        return "retry_query"

    logger.info("Successful crawl. Ending process.")
    return "stop"


def create_agent():
    """Builds and compiles the marketplace query agent workflow."""

    graph_builder = StateGraph(MarketplaceState)

    graph_builder.add_node("optimised_query", refine_marketplace_query)
    graph_builder.add_node("search_query_url", generate_search_url)
    graph_builder.add_node("crawler", product_crawler)

    graph_builder.add_edge(START, "optimised_query")
    graph_builder.add_edge("optimised_query", "search_query_url")
    graph_builder.add_edge("search_query_url", "crawler")

    graph_builder.add_conditional_edges(
        "crawler",
        crawler_decision,
        {
            "repeat": "crawler",
            "retry_query": "optimised_query",
            "stop": END,
        },
    )

    graph = graph_builder.compile()
    return graph


agent = create_agent()

display(Image(agent.get_graph().draw_mermaid_png()))

# %%

initial_state = {
    "query": "I am looking for a gaming laptop that is capable of running LLMs.",
    "ranked_queries": {},
    "web_url": "https://www.takealot.com/all?&qsearch=",
    "search_url": "",
    "results": [],
    "retries": 0,
}


async def run_marketplace_workflow():
    final_state = await agent.ainvoke(initial_state)
    print(final_state)


# asyncio.run(run_marketplace_workflow())
await run_marketplace_workflow()


# %%
# %%
