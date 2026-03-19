import base64
import io
from PIL import Image
import aiohttp
import asyncio
from typing import List

async def caption_images_via_gemini(image_info_list: List[dict], api_key: str) -> List[dict]:
    """
    Asynchronously captions images using Gemini API.
    Args:
        image_info_list: [{'image': bytes, 'page': int}]
        api_key: Google API key
    Returns:
        [{'page': int, 'caption': str}]
    """
    captions = []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for img_obj in image_info_list:
            img_bytes = img_obj["image"]
            page = img_obj["page_number"]
            page_img_path = img_obj.get("page_img_path")

            try:
                image = Image.open(io.BytesIO(img_bytes))
                mime_type = f"image/{image.format.lower()}"
                b64_img = base64.b64encode(img_bytes).decode("utf-8")

                payload = {
                    "contents": [{
                        "parts": [
                            {"text": "Describe this image briefly for document retrieval context."},
                            {"inline_data": {"mimeType": mime_type, "data": b64_img}}
                        ]
                    }]
                }

                # Each caption request as coroutine
                tasks.append(fetch_caption(session, url, headers, payload, page, page_img_path))
            except Exception as e:
                captions.append({"page_number": page, "caption": f"Error preparing image: {e},", "page_img_path": page_img_path})

        # Run all caption requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        captions.extend(results)

    return captions


async def fetch_caption(session, url, headers, payload, page, page_img_path):
    """Helper coroutine to fetch one caption"""
    try:
        async with session.post(url, headers=headers, json=payload, timeout=60) as response:
            if response.status != 200:
                txt = await response.text()
                return {"page": page, "caption": f"HTTP {response.status}: {txt}", "page_img_path": page_img_path}

            res = await response.json()
            caption = res["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"page_number": page, "caption": caption, "page_img_path": page_img_path}
    except Exception as e:
        return {"page_number": page, "caption": f"Error: {e}", "page_img_path": page_img_path}


# if __name__ == "__main__":
#     import os, asyncio, base64
#
#     async def test():
#         api_key = os.getenv("GOOGLE_API_KEY", "")
#         if not api_key:
#             print("Missing GOOGLE_API_KEY.")
#             return
#
#         # dummy 1x1 PNG
#         dummy_png_bytes = base64.b64decode(
#             "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
#         )
#         image_info_list = [{"image": dummy_png_bytes, "page_number": 1, "source" : "test", "page_img_path": "/test/1.png"}]
#         captions = await caption_images_via_gemini(image_info_list, api_key)
#         print(captions[0].keys())
#
#     asyncio.run(test())
