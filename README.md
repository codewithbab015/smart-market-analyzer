# 🧠 smart-market-analyzer

An AI-driven multi-agent system that intelligently crawls, analyses, and synthesises product data from online marketplaces to provide users with reliable, context-aware purchasing recommendations.

---

## 🚀 Project Overview

**Objective:**  
To develop an AI-powered tool that assists users in identifying high-quality and relevant products by analysing product reviews, specifications, and marketplace trends. The system transforms vague user queries into actionable, data-backed insights.

---

## 🎯 Measurable Objectives

The project’s success is determined by the system’s ability to:

- **Rank Products by Review Quality and Volume:**  
  Combine average ratings and review counts to reflect both quality and user confidence.  

- **Identify Trending and Popular Products:**  
  Detect products with high engagement through review recency and frequency.  

- **Highlight High-Value Products:**  
  Recognise promotions and exceptional value through review and price analysis.  

- **Provide Reliable Recommendations:**  
  Present the top 3 curated products with clear reasoning to enhance user trust and decision-making speed.  

---

## ⚙️ System Architecture

The system is composed of **four specialised AI agents** operating in a sequential workflow:

1. **Query Clarifier** – Interprets and normalises user queries into structured intents.  
2. **Taxonomy Mapper** – Converts structured intents into crawl instructions using marketplace taxonomies.  
3. **Crawl Orchestrator** – Executes crawling tasks via `crawl4ai` and retrieves raw product data.  
4. **Results Refiner** – Validates, ranks, and presents the top products with justifications.

**Data Flow:**  

---

## 🧩 Agent Specifications

| Agent | Role | Input | Output | Key Functionality |
|-------|------|--------|---------|------------------|
| **1. Query Clarifier & Intent Analyst** | Interpreter | Raw User Query | Structured Intent (JSON) | Corrects typos, extracts entities, resolves ambiguity, and identifies non-negotiable vs inferred filters. |
| **2. Category & Taxonomy Mapper** | Translator | Structured Intent (JSON) | Crawl Instructions (JSON) | Maps user intent to marketplace taxonomies and generates crawler-ready instructions. |
| **3. CrawlFor.AI Orchestrator** | Executor | Crawl Instructions (JSON) | Raw Product Data | Interfaces with `crawl4ai` to retrieve product listings, reviews, and pricing. |
| **4. Results Refiner & Presenter** | Curator | Raw Product Data | Final Recommendations | Validates, ranks, and presents top products with justification for each recommendation. |

---

## 🔄 End-to-End Workflow Example

1. **User Input:**  
   “Good gaming laptop for VR.”  

2. **Clarification:**  
   Agent 1 → `{"category": "laptop", "use_case": ["gaming", "VR"], "key_attributes": ["dedicated_gpu", "high_ram"]}`  

3. **Crawl Planning:**  
   Agent 2 maps to category → “Electronics → Computers → Gaming Laptops”.  

4. **Data Retrieval:**  
   Agent 3 crawls the marketplace and collects relevant product data.  

5. **Result Refinement:**  
   Agent 4 filters, ranks, and outputs the top 3 recommendations with reasoning.

---

## 📦 Key Deliverables

- Fully integrated **multi-agent pipeline** from query to recommendation.  
- Structured **JSON communication schema** between agents.  
- Custom **ranking and presentation algorithm** for product scoring.  
- Clear documentation for deployment and interaction.  

---

## 🧭 Summary

The **Market Analysis Agent** unifies intelligent crawling, natural language understanding, and analytical ranking into a seamless AI pipeline — designed to simplify and improve product discovery for end users.

---

**Author:** [Nhlanhla Baloyi]  
**Repository:** [`smart-market-analyzer`](https://github.com/yourusername/market-scout-ai)

