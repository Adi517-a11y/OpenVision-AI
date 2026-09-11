import io
import json
import os
import requests
from PIL import Image

API_URL = "https://commons.wikimedia.org/w/api.php"

IMAGE_DIR = "dataset/images"
METADATA_FILE = "dataset/metadata.jsonl"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs("dataset", exist_ok=True)

SEARCH_TERMS = [
    "cat",
    "dog",
    "car",
    "house",
    "tree",
    "mountain",
    "ocean",
    "forest",
    "city",
    "person"
]

HEADERS = {
    "User-Agent": "OpenVision-AI/0.1 (dataset research project)"
}


def search_commons(search_term, limit=10):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": 512
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    return response.json().get(
        "query",
        {}
    ).get(
        "pages",
        {}
    )


def clean_text(value):
    if not value:
        return ""

    return (
        value
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


image_number = 0

with open(METADATA_FILE, "a", encoding="utf-8") as metadata:

    for search_term in SEARCH_TERMS:

        print(f"\nSearching Wikimedia Commons for: {search_term}")

        pages = search_commons(
            search_term,
            limit=10
        )

        for page in pages.values():

            image_info = page.get(
                "imageinfo",
                [{}]
            )[0]

            url = image_info.get("thumburl") or image_info.get("url")

            mime = image_info.get(
                "mime",
                ""
            )

            if not url:
                continue

            if not mime.startswith("image/"):
                continue

            try:

                image_response = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=30
                )

                image_response.raise_for_status()

                image = Image.open(
                    io.BytesIO(
                        image_response.content
                    )
                ).convert("RGB")

                filename = f"image_{image_number:06d}.jpg"

                filepath = os.path.join(
                    IMAGE_DIR,
                    filename
                )

                image.save(
                    filepath,
                    "JPEG",
                    quality=90
                )

                extmetadata = image_info.get(
                    "extmetadata",
                    {}
                )

                description = clean_text(
                    extmetadata
                    .get("ImageDescription", {})
                    .get("value", "")
                )

                title = clean_text(
                    page.get("title", "")
                )

                caption = (
                    description
                    if description
                    else f"{search_term}: {title}"
                )

                record = {
                    "image": f"images/{filename}",
                    "caption": caption,
                    "source": "Wikimedia Commons",
                    "source_url": image_info.get("url", ""),
                    "search_term": search_term
                }

                metadata.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                metadata.flush()

                print(
                    f"Downloaded {filename} "
                    f"-> {caption[:80]}"
                )

                image_number += 1

            except Exception as error:

                print(
                    f"Skipped {page.get('title', 'image')}: "
                    f"{error}"
                )


print("\n================================")
print("OpenVision dataset collection complete")
print(f"Images downloaded: {image_number}")
print(f"Metadata: {METADATA_FILE}")
print("================================")
