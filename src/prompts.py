from langchain.prompts import PromptTemplate

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
