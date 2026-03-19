import asyncio, os, time, threading
from collections import defaultdict
import httpx
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


from utils.pdf_extractor import extract_text_and_images
from utils.image_captioning import caption_images_via_gemini
from utils.semantic_chunker import semantic_chunk_text
from utils.pinecone_utils import upsert_documents_to_pinecone
from utils.agent_rag import run_rag_agent

from pinecone.grpc import PineconeGRPC as PineconeAsync
from pinecone import Pinecone

# Deepgram SDK v5.x
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV1SocketClientResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

# Pinecone
pc_async = PineconeAsync(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-rag")
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index_name = "agentic-rag"  # Make sure this is your index name

    if pinecone_index_name not in pc.list_indexes().names():
        print(f"!!! ERROR: Index '{pinecone_index_name}' does not exist. Please upload a document first.")
    pinecone_index = pc.Index(pinecone_index_name)
    print("Successfully connected to Pinecone index.")
except Exception as e:
    print(f"!!! ERROR: Could not connect to Pinecone: {e}")

# Deepgram
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("Set DEEPGRAM_API_KEY in environment")

# Create regular and async clients (async used for listen v2 connect)
deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8002",
        "http://127.0.0.1:8002",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:63342",  # JetBrains / VSCode live preview
        "http://127.0.0.1:5500",   # VSCode Live Server
        "*",  # (optional — use only for testing; don't keep in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.png")
async def favicon():
    return FileResponse("favicon.png")

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF, extract text/images, caption images, chunk and upsert to Pinecone"""
    file_path = f"static/{file.filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    full_text, embedded_images, page_texts = await extract_text_and_images(file_path)
    print(f"[INFO] Extracted {len(page_texts)} pages and {len(embedded_images)} embedded images.")

    captions = []
    if embedded_images:
        captions = await caption_images_via_gemini(embedded_images, os.getenv("GEMINI_API_KEY"))
        print(f"[INFO] Generated {len(captions)} image captions.")

    # Build caption map (allow multiple per page)
    caption_map = defaultdict(list)
    for c in captions:
        caption_map[c["page_number"]].append(c["caption"])

    embeddings_to_upsert = []
    total_chunks = 0
    for page in page_texts:
        page_chunks = semantic_chunk_text(page["text"])
        total_chunks += len(page_chunks)
        page_num = page["page_number"]
        page_captions = caption_map.get(page_num, [])
        page_img_path = page.get("page_img_path")

        for chunk in page_chunks:
            embeddings_to_upsert.append({
                "text": chunk,
                "metadata": {
                    "page_number": page_num,
                    "caption": page_captions,
                    "source": file.filename,
                    "page_img_path": page_img_path
                }
            })

    captions_for_upsert = [
        {"page": c["page_number"], "caption": c["caption"], "page_img_path": c["page_img_path"]}
        for c in captions
    ]

    await upsert_documents_to_pinecone(
        pc_async_client=pc_async,
        pinecone_index_name=PINECONE_INDEX_NAME,
        chunks=embeddings_to_upsert,
        captions=captions_for_upsert,
        file_name=file.filename,
    )

    return {
        "filename": file.filename,
        "num_pages": len(page_texts),
        "num_images": len(embedded_images),
        "num_chunks": total_chunks,
        "image_captions": captions,
        "status": "success"
    }

@app.post("/ask")
async def ask_a_question(
    query: str = Body(..., embed=True, description="User question text")
):
    """
    Accepts a text query and returns the RAG answer from Pinecone + LLM.
    """
    try:
        print(f"[INFO] Received query: {query}")
        result = await run_rag_agent(query, pinecone_index)
        return {"RAG Answer": result}

    except Exception as e:
        print(f"[ERROR] RAG processing failed: {e}")
        return {"error": str(e)}

@app.post("/ask_from_audio_file")
async def transcribe_audio_and_answer(file: UploadFile = File(...)):
    """
    Accepts an audio file and returns the transcript from Deepgram's
    pre-recorded transcription API.
    """
    print("\n[INFO] Received audio file for pre-recorded transcription.")
    try:
        audio_buffer = await file.read()

        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": file.content_type
        }
        async with httpx.AsyncClient() as client:
            # Send the audio_buffer as the request data
            response = await client.post(url, content=audio_buffer, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Get the JSON response from Deepgram
            result = response.json()

            # Extract the transcript
            transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            result = await run_rag_agent(transcript, pinecone_index)
            return {"RAG Answer": result, "Transcript": transcript}


    except httpx.HTTPStatusError as e:
        print(f"[ERROR] Deepgram API error: {e.response.text}")
        return {"error": f"Deepgram API error: {e.response.text}"}

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return {"error": str(e)}

@app.websocket("/audio")
async def audio_stream(websocket: WebSocket):
    """
    Phase 1: Raw audio WebSocket endpoint.
    Receives binary audio chunks from the frontend,
    prints their size, and writes them to a .wav file for validation.
    """
    await websocket.accept()
    print("\n[INFO] Client connected to /audio WebSocket")

    # Prepare file to write incoming audio
    os.makedirs("temp_audio", exist_ok=True)
    output_path = f"temp_audio/audio_{int(time.time())}.raw"

    try:
        with open(output_path, "wb") as f:
            while True:
                try:
                    data = await websocket.receive_bytes()
                    print(f"[INFO] Received {len(data)} bytes of audio.")
                    f.write(data)
                except WebSocketDisconnect:
                    print("[INFO] Client disconnected.")
                    break
                except Exception as e:
                    print(f"[ERROR] WebSocket error: {e}")
                    break

        print(f"[INFO] Audio stream saved to: {output_path}")

    finally:
        # only close if connection is still active
        try:
            if websocket.application_state.value == 2:  # CONNECTED
                await websocket.close()
        except RuntimeError:
            pass
        print("[INFO] WebSocket connection cleanup done.")


@app.websocket("/listen")
async def listen_websocket(websocket: WebSocket):
    await websocket.accept()
    print("\n[INFO] Client connected to /listen WebSocket")

    deepgram: DeepgramClient = DeepgramClient()
    loop = asyncio.get_running_loop()

    # open Deepgram websocket
    with deepgram.listen.v1.connect(model="nova-3") as connection:

        # handle Deepgram messages
        # def on_message(message: ListenV1SocketClientResponse):
        #     if hasattr(message, "channel") and hasattr(message.channel, "alternatives"):
        #         transcript = message.channel.alternatives[0].transcript.strip()
        #         if transcript:
        #             print(f"[Deepgram] {transcript}")
        #
        #             # send transcript back to the browser safely
        #             asyncio.run_coroutine_threadsafe(
        #                 websocket.send_text(transcript),
        #                 loop
        #             )

        def on_message(message: ListenV1SocketClientResponse):
            try:
                if hasattr(message, "channel") and hasattr(message.channel, "alternatives"):
                    alt = message.channel.alternatives[0]
                    transcript = alt.transcript.strip()
                    if not transcript:
                        return
                    print(f"[Deepgram] {transcript}")

                    # Send transcript to frontend immediately
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({"type": "transcript", "data": transcript}),
                        loop
                    )

                    # Check if it's a final transcript
                    is_final = getattr(message, "is_final", False)

                    if is_final:
                        print(f"[Deepgram Final]: {transcript}")

                        # Run RAG agent on the final transcript
                        async def rag_task():
                            try:
                                rag_result = await run_rag_agent(transcript, pinecone_index)
                                await websocket.send_json({
                                    "type": "rag_answer",
                                    "data": rag_result
                                })
                                print(f"[RAG]: {rag_result}")
                            except Exception as e:
                                print(f"[ERROR] RAG agent error: {e}")
                                await websocket.send_json({
                                    "type": "rag_error",
                                    "data": str(e)
                                })

                        # Schedule the async RAG task in the FastAPI loop
                        asyncio.run_coroutine_threadsafe(rag_task(), loop)

            except Exception as e:
                print(f"[ERROR] Deepgram message handling: {e}")


        connection.on(EventType.OPEN, lambda _: print("[Deepgram] Connection opened"))
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.CLOSE, lambda _: print("[Deepgram] Connection closed"))
        connection.on(EventType.ERROR, lambda e: print(f"[Deepgram] Error: {e}"))

        lock_exit = threading.Lock()
        exit_flag = False

        # thread that listens for Deepgram messages
        def listening_thread():
            try:
                connection.start_listening()
            except Exception as e:
                print(f"[Deepgram] Listener thread error: {e}")

        thread_dg = threading.Thread(target=listening_thread, daemon=True)
        thread_dg.start()

        print("[INFO] Ready to receive audio from client.")

        try:
            while True:
                data = await websocket.receive_bytes()
                # forward audio bytes from browser to Deepgram
                lock_exit.acquire()
                if exit_flag:
                    lock_exit.release()
                    break
                lock_exit.release()
                connection.send_media(data)
        except WebSocketDisconnect:
            print("[INFO] Frontend disconnected.")
        except Exception as e:
            print(f"[ERROR] {e}")

        # clean up
        lock_exit.acquire()
        exit_flag = True
        lock_exit.release()

        try:
            # check before closing
            if connection and not getattr(connection, "_closed", True):
                connection.finish()
        except Exception as e:
            print(f"[WARN] Deepgram connection already closed or cleanup error: {e}")

        try:
            await websocket.close()
        except Exception:
            pass

        print("[INFO] Connection cleanup done.")


