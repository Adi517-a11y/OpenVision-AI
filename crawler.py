import io
import json
import os
import requests
import boto3

from PIL import Image


# ============================================================
# OpenVision AI — Dataset Crawler + Cloud Storage
# ============================================================

API_URL = "https://commons.wikimedia.org/w/api.php"

IMAGE_DIR = "dataset/images"
METADATA_FILE = "dataset/metadata.jsonl"

# Cloudflare R2 / S3-compatible storage
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.environ.get("R2_BUCKET")


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
    "User-Agent": "OpenVision-AI/0.1 dataset research project"
}


# ============================================================
# Storage
# ============================================================

def create_storage_client():

    if not all([
        R2_ENDPOINT,
        R2_ACCESS_KEY_ID,
        R2_SECRET_ACCESS_KEY,
        R2_BUCKET
    ]):

        print(
            "Cloud storage variables are not configured."
        )

        return None

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY
    )


storage = create_storage_client()


def upload_file(local_path, remote_path):

    if storage is None:
        return False

    try:

        storage.upload_file(
            local_path,
            R2_BUCKET,
            remote_path
        )

        print(
            f"Uploaded: {remote_path}"
        )

        return True

    except Exception as error:

        print(
            f"Upload failed: {error}"
        )

        return False


# ============================================================
# Wikimedia search
# ============================================================

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

    data = response.json()

    return (
        data
        .get("query", {})
        .get("pages", {})
    )


# ============================================================
# Text cleanup
# ============================================================

def clean_text(value):

    if not value:
        return ""

    return (
        value
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


# ============================================================
# Dataset creation
# ============================================================

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)

os.makedirs(
    "dataset",
    exist_ok=True
)


image_number = 0


with open(
    METADATA_FILE,
    "a",
    encoding="utf-8"
) as metadata:

    for search_term in SEARCH_TERMS:

        print()
        print(
            f"Searching Commons: {search_term}"
        )

        pages = search_commons(
            search_term,
            limit=10
        )

        for page in pages.values():

            image_info = page.get(
                "imageinfo",
                [{}]
            )[0]

            url = (
                image_info.get("thumburl")
                or image_info.get("url")
            )

            mime = image_info.get(
                "mime",
                ""
            )

            if not url:
                continue

            if not mime.startswith("image/"):
                continue

            try:

                # ----------------------------------------
                # Download image
                # ----------------------------------------

                response = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=30
                )

                response.raise_for_status()

                image = Image.open(
                    io.BytesIO(
                        response.content
                    )
                ).convert("RGB")

                filename = (
                    f"image_{image_number:06d}.jpg"
                )

                local_path = os.path.join(
                    IMAGE_DIR,
                    filename
                )

                image.save(
                    local_path,
                    "JPEG",
                    quality=90
                )

                # ----------------------------------------
                # Create caption
                # ----------------------------------------

                extmetadata = image_info.get(
                    "extmetadata",
                    {}
                )

                description = clean_text(
                    extmetadata
                    .get(
                        "ImageDescription",
                        {}
                    )
                    .get(
                        "value",
                        ""
                    )
                )

                title = clean_text(
                    page.get(
                        "title",
                        ""
                    )
                )

                caption = (
                    description
                    if description
                    else f"{search_term}: {title}"
                )

                # ----------------------------------------
                # Dataset metadata
                # ----------------------------------------

                remote_image = (
                    f"images/{filename}"
                )

                record = {
                    "image": remote_image,
                    "caption": caption,
                    "source": "Wikimedia Commons",
                    "source_url": image_info.get(
                        "url",
                        ""
                    ),
                    "search_term": search_term
                }

                metadata.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                metadata.flush()

                # ----------------------------------------
                # Upload image
                # ----------------------------------------

                upload_file(
                    local_path,
                    remote_image
                )

                print(
                    f"Processed: {filename}"
                )

                image_number += 1

            except Exception as error:

                print(
                    f"Skipped image: {error}"
                )


# ============================================================
# Upload metadata
# ============================================================

upload_file(
    METADATA_FILE,
    "metadata.jsonl"
)


print()
print("==========================================")
print("OpenVision dataset collection complete")
print("==========================================")
print(
    f"Images processed: {image_number}"
)
print(
    f"Local metadata: {METADATA_FILE}"
)
print("Cloud upload complete.")
