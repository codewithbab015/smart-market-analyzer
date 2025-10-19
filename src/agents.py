from typing import Dict, List, TypedDict
from urllib.parse import urlencode

from langgraph.graph import END, START, StateGraph

from .agent_utils import (
    QueryBuilder,
    QueryTranslator,
    RankQuery,
    best_polished_query,
    logger,
)


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
