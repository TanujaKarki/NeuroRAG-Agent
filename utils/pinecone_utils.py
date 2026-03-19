from typing import List, Dict, Any

# Use a relative import to get our new async embedding function
from .embeddings import get_openai_embeddings_batch


async def upsert_documents_to_pinecone(
    pc_async_client,
    pinecone_index_name: str,
    chunks: List[Dict[str, Any]],
    captions: List[Dict[str, Any]],
    file_name: str = None,
):
    """
    Embed text chunks + captions and upsert to Pinecone with detailed metadata.
    Metadata includes:
        - type: text or image_caption
        - source: file name
        - page_number
        - page_img_path
        - caption text (if any)
    """
    index = pc_async_client.Index(pinecone_index_name)

    vectors_to_upsert = []

    # Embed text chunks
    if chunks:
        chunk_texts = [d["text"] for d in chunks]
        print(f"[INFO] Creating embeddings for {len(chunk_texts)} text chunks...")
        embeddings = await get_openai_embeddings_batch(chunk_texts)
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            meta = {
                "type": "text",
                "source": meta.get("source", file_name),
                "page_number": meta.get("page_number"),
                "text": chunk["text"],
                "page_img_path": meta.get("page_img_path")
            }
            vectors_to_upsert.append({
                "id": f"{file_name}_text_{i}",
                "values": embeddings[i],
                "metadata": meta
            })

    # Embed image captions
    if captions:
        caption_texts = [c["caption"] for c in captions]
        print(f"[INFO] Creating embeddings for {len(caption_texts)} image captions...")
        caption_embeddings = await get_openai_embeddings_batch(caption_texts)
        for i, c in enumerate(captions):
            meta = {
                "type": "image_caption",
                "source": file_name,
                "page_number": c.get("page"),
                "caption": c["caption"],
                "page_img_path": c.get("page_img_path")
            }
            vectors_to_upsert.append({
                "id": f"{file_name}_caption_{i}",
                "values": caption_embeddings[i],
                "metadata": meta
            })

    # Upsert everything
    if vectors_to_upsert:
        print(f"[INFO] Upserting {len(vectors_to_upsert)} vectors with metadata into Pinecone index: {pinecone_index_name}")
        index.upsert(vectors=vectors_to_upsert)
        print("[INFO] Upsert complete.")
    else:
        print("[WARN] No data to upsert.")