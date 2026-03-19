import fitz
import os

async def extract_text_and_images(pdf_path: str):
    """
    Extracts text and embedded images from each page of a PDF.
    Also saves each page as a .png image for citation visualization.

    Returns:
        full_text (str): combined text of all pages
        image_bytes_list (List[bytes]): all embedded images in the PDF
        page_texts (List[dict]): [{'page_number': int, 'text': str, 'page_image_path': str}]
    """
    doc = fitz.open(pdf_path)
    pdf_name = pdf_path[pdf_path.rfind("\\")+1:]
    all_text = ""
    image_bytes_list = []
    page_texts = []

    os.makedirs("static/pdf_pages", exist_ok=True)
    file_stem = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        all_text += text

        # Save the page snapshot for citation display
        page_img_path = f"static/pdf_pages/{file_stem}_page_{page_num}.png"
        try:
            pix = page.get_pixmap(dpi=150)
            pix.save(page_img_path)
        except Exception as e:
            print(f"[WARN] Could not render page {page_num} to image: {e}")
            page_img_path = None

        # Extract embedded images on this page
        for img in page.get_images(full=True):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes_list.append({
                    "page_number": page_num,
                    "image": base_image["image"],
                    "source": pdf_name,
                    "page_img_path": page_img_path
                })
            except Exception as e:
                print(f"[WARN] Error extracting image from page {page_num}: {e}")

        page_texts.append({
            "page_number": page_num,
            "text": text,
            "source": pdf_name,
            "page_img_path": page_img_path
        })

    return all_text, image_bytes_list, page_texts



import asyncio
if __name__ == "__main__":
    async def test():
        full_text, image_bytes, page_texts = await extract_text_and_images(
            r"C:\Users\digvi\Desktop\projects\Multi_Model_Agentic_RAG\static\AiRA_Research_Paper.pdf"
        )
        print(f" Total text length: {len(full_text)}")
        print(f" Extracted {len(image_bytes)} embedded images")
        print(f" Found {len(page_texts)} pages")

        for p in page_texts[:2]:  # preview first 2 pages
            print(f"\nPage {p['page_number']} -> Image: {p['source']}")
            print(p['text'])
            print(p.keys())
        print(image_bytes[0].keys())

    asyncio.run(test())
