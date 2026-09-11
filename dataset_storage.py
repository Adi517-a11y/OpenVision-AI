import os
import boto3


R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.environ.get("R2_BUCKET")


def get_storage():

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY
    )


def download_dataset(
    metadata_key="metadata.jsonl",
    local_metadata="dataset/metadata.jsonl"
):

    os.makedirs(
        "dataset/images",
        exist_ok=True
    )

    storage = get_storage()

    print(
        "Downloading OpenVision metadata..."
    )

    storage.download_file(
        R2_BUCKET,
        metadata_key,
        local_metadata
    )

    print(
        "Metadata downloaded."
    )

    with open(
        local_metadata,
        "r",
        encoding="utf-8"
    ) as file:

        records = [
            line.strip()
            for line in file
            if line.strip()
        ]

    print(
        f"Found {len(records)} dataset records."
    )

    for record in records:

        import json

        item = json.loads(record)

        remote_image = item["image"]

        local_image = os.path.join(
            "dataset",
            remote_image
        )

        os.makedirs(
            os.path.dirname(local_image),
            exist_ok=True
        )

        if os.path.exists(local_image):
            continue

        print(
            f"Downloading {remote_image}"
        )

        storage.download_file(
            R2_BUCKET,
            remote_image,
            local_image
        )

    print(
        "Dataset download complete."
    )


if __name__ == "__main__":

    download_dataset()
