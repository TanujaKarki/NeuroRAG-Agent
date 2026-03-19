import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from pinecone import Pinecone
from dotenv import load_dotenv

from .embeddings import get_openai_embeddings_batch

load_dotenv()

# --- Clients ---
LLM_MODEL = "gpt-4o"
llm = ChatOpenAI(model=LLM_MODEL, api_key=os.getenv("OPENAI_API_KEY"))
serper_search = GoogleSerperAPIWrapper(api_key=os.getenv("SERPER_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# --- Tools ---
async def pinecone_search(query: str, index) -> List[Dict[str, Any]]:
    print(f"[RAG] pinecone search → {query}")

    q_emb = (await get_openai_embeddings_batch([query]))[0]

    res = index.query(
        vector=q_emb,
        top_k=5,
        include_metadata=True
    )

    matches = res.get("matches", []) or []
    return [m.get("metadata", {}) for m in matches]


def web_search(query: str) -> str:
    print(f"[RAG] web search → {query}")
    try:
        r = serper_search.results(query)
        organic = r.get("organic", [])
        context_snippets = []

        for o in organic[:3]:
            title = o.get("title", "Untitled")
            snippet = o.get("snippet", "")
            link = o.get("link", "")
            date = o.get("date", "")

            #Markdown-friendly formatting for citations
            formatted_snippet = (
                f"**{title}** ({date})\n"
                f"{snippet}\n"
                f"[Source]({link})"
            )

            context_snippets.append(formatted_snippet)

        return "\n\n---\n\n".join(context_snippets), link

    except Exception as e:
        print("serper error:", e)
        return "search error"


# --- Router ---
async def router(query: str) -> str:
    prompt = f"""
You are a smart query router.

Decide which tool to use to best answer the query:

- Use "pinecone_search" if the user refers to the uploaded document, research paper, or anything that sounds like a report, model, results, accuracy, analysis, figure, dataset, etc.
- Use "web_search" if the question is general knowledge, news, or external info not likely inside the uploaded document.

Examples:
Q: What accuracy did the paper achieve? → pinecone_search
Q: Who created GPT-4? → web_search
Q: What is AIRA's model about? → pinecone_search
Q: Latest news about OpenAI? → web_search

Now classify:
{query}
"""
    r = await llm.ainvoke([{"role": "user", "content": prompt}])
    return r.content.strip().lower()


# --- Answer Synthesizer ---
async def answer_synthesizer(query: str, context: str) -> str:
    prompt = f"""
You are a research assistant. Use only the context below to answer as accurately as possible.

If the information seems relevant but incomplete, make a **concise best-effort summary** instead of refusing.

Context:
{context}

Question: {query}

Answer directly and include page or source hints if available.
"""
    r = await llm.ainvoke([{"role": "user", "content": prompt}])
    return r.content.strip()



# --- Main RAG Agent ---
async def run_rag_agent(query: str, pinecone_index) -> Dict[str, Any]:
    tool = await router(query)
    citations = []

    if "pinecone" in tool:
        ctxs = await pinecone_search(query, pinecone_index)

        # Combine text chunks + track citations
        context = ""
        seen_pages = set()

        for c in ctxs:
            page_num = c.get("page_number")
            page_img = c.get("page_img_path")
            text = c.get("text", "")
            cap = c.get("captions", [])

            context += f"\n\n[Page {page_num}] {text}"
            if cap:
                context += f"\nImage captions: {'; '.join(cap)}"

            # Save citation
            if page_num and page_num not in seen_pages:
                citations.append({
                    "page": page_num,
                    "image": page_img
                })
                seen_pages.add(page_num)

    elif "web" in tool:
        context, link = web_search(query)
        citations.append({
        "Source": link
        })

    else:
        context = "router_error"

    answer = await answer_synthesizer(query, context)

    return {
        "answer": answer,
        "citations": citations
    }
