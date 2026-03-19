from langchain_text_splitters import RecursiveCharacterTextSplitter

def semantic_chunk_text(text, chunk_size=512, chunk_overlap=50):
    # Can replace with a more advanced semantic splitter—in LangChain this is a good default
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks
