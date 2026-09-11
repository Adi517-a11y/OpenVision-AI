import requests

API_URL = "https://commons.wikimedia.org/w/api.php"

params = {
    "action": "query",
    "format": "json",
    "generator": "search",
    "gsrsearch": "cat",
    "gsrnamespace": 6,
    "gsrlimit": 5,
    "prop": "imageinfo",
    "iiprop": "url"
}

response = requests.get(
    API_URL,
    params=params,
    timeout=30
)

response.raise_for_status()

data = response.json()

pages = data.get("query", {}).get("pages", {})

for page in pages.values():
    image_info = page.get("imageinfo", [{}])[0]
    url = image_info.get("url")

    if url:
        print(url)
