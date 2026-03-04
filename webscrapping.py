import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time


news_sites = [
    "https://kathmandupost.com",
    "https://thehimalayantimes.com",
    "https://english.onlinekhabar.com"
]

keywords = ["Election", "Vote", "Ballot", "Parliament"]

headers = {
    "User-Agent": "Mozilla/5.0"
}


def scrape_article(url):
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        res.encoding = res.apparent_encoding

        soup = BeautifulSoup(res.text, "html.parser")

        paragraphs = soup.find_all("p")
        content = "\n".join(
            [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40]
        )

        return content

    except:
        return None


def scrape_site(url):
    articles_found = []
    seen_links = set()

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            title = link.get_text(strip=True)
            href = link["href"]

            if len(title) < 20:
                continue

            if any(k.lower() in title.lower() for k in keywords):

                full_url = urljoin(url, href)

                if full_url not in seen_links:
                    seen_links.add(full_url)

                    print(f"Fetching full article: {title}")

                    content = scrape_article(full_url)

                    if content:
                        articles_found.append({
                            "title": title,
                            "url": full_url,
                            "content": content
                        })

                    time.sleep(1)

        return articles_found

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []


print("🗳️ Collecting full election news articles...\n")

all_articles = []

for site in news_sites:
    print(f"Checking {site}...\n")
    results = scrape_site(site)
    all_articles.extend(results)


# Save to JSONL (for your embedding pipeline)
with open("articles.jsonl", "w", encoding="utf-8") as f:
    for article in all_articles:
        f.write(json.dumps(article, ensure_ascii=False) + "\n")

print("\n✅ Done! Articles saved to articles.jsonl")