import asyncio
import json
import logging
import re
import time
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Tuple

import pandas as pd
from langchain_ollama import OllamaLLM
from pydantic.dataclasses import dataclass

from .prompts import query_improver_prompt, query_ranker_prompt, translator_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ModelID(Enum):
    """Supported AI model IDs."""

    gemma: str = "gemma3:latest"
    deepseek: str = "deepseek-r1:latest"
    mistral: str = "mistral:7b"
    llama: str = "llama3:latest"
    qwen3: str = "qwen3:latest"

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
        return query_improver_prompt | llm

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


@dataclass
class RankQuery:
    """
    Rank improved queries from multiple models using a supervisor LLM.
    """

    model_id: str = "deepseek-r1:latest"

    def __post_init__(self):
        # Initialize the supervisor LLM and chain at instance level
        self.llm_supervisor = OllamaLLM(model=self.model_id, temperature=0.1)
        self.chain = query_ranker_prompt | self.llm_supervisor

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
        self.chain = translator_prompt | self.llm

    def show_translated_query(self, query: str) -> str:
        """Translate or refine the given user query."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        response: str = self.chain.invoke({"query": query})
        return response.strip()
