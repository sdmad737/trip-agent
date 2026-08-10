import re
import requests
import streamlit as st
import time
from html import unescape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"

WIKI_HEADERS = {
    "User-Agent": "trip-planner-project/1.0 (Shahin M.)"
}

@st.cache_data(ttl=3600)
def fetch_wikivoyage_article(destination):

    params = {
        "action": "parse",
        "page": destination,
        "prop": "text",
        "format": "json",
        "redirects": True
    }

    last_error = None

    for attempt in range(3):

        try:

            response = requests.get(
                WIKIVOYAGE_API,
                params=params,
                headers=WIKI_HEADERS,
                timeout=20
            )

            if response.status_code == 429:

                time.sleep(
                    2 ** attempt
                )

                continue

            response.raise_for_status()

            try:
                data = response.json()

            except requests.exceptions.JSONDecodeError:

                raise ValueError(
                    "Wikivoyage returned invalid JSON."
                )

            if not isinstance(data, dict):
                return None

            if "error" in data:
                return None

            parse_data = data.get("parse")

            if not isinstance(parse_data, dict):
                return None

            html = (
                parse_data
                .get("text", {})
                .get("*")
            )

            if not html:
                return None

            return {
                "title": parse_data.get(
                    "title",
                    destination
                ),
                "html": html
            }

        except requests.RequestException as error:

            last_error = error

            if attempt < 2:
                time.sleep(
                    2 ** attempt
                )

    return None

def clean_html(html):
    html = re.sub(
        r"<(script|style).*?>.*?</\1>",
        " ",
        html,
        flags = re.DOTALL | re.IGNORECASE
    )

    html = re.sub(r"</(p|div|li|h1|h2|h3|h4)>","\n",html,flags=re.IGNORECASE)

    text = re.sub(r"<[^>]+>"," ",html)

    text = unescape(text)

    text = re.sub(r"[ \t]+", " ",text)
    text = re.sub(r"\n\s*\n+","\n\n",text)

    return text.strip()


def chunk_text(text,target_size=900):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        if len(current_chunk) + len(paragraph) + 1 <= target_size:
            if current_chunk:
                current_chunk += " "

            current_chunk += paragraph

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())


            if len(paragraph) > target_size:

                sentences = re.split(r"(?<=[.!?])\s+",paragraph)

                sentence_chunk = ""

                for sentence in sentences:
                    if (len(sentence_chunk) + len(sentence) + 1 <= target_size):
                        if sentence_chunk:
                            sentence_chunk += " "

                        sentence_chunk += sentence

                    else:
                        if sentence_chunk:
                            chunks.append(sentence_chunk.strip())
                        sentence_chunk = sentence
                current_chunk = sentence_chunk
            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

@st.cache_resource
def build_tfidf_index(chunks):
    vectorizer = TfidfVectorizer(sto_words="english",max_features=10000)

    document_vectors = vectorizer.fit_transform(chunks)

    return vectorizer, document_vectors


def retrieve_chunks(query, chunks, vectorizer, document_vectors, source, top_k=5):
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, document_vectors)[0]

    ranked_indices = similarities.argsort()[::1][:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            "chunk_id": int(index),
            "source": source,
            "text": chunks[index],
            "score": float(similarities[index])
        })

    return results


@st.cache_data(ttl=3500)
def prepare_wikivoyage(destination):
    article = fetch_wikivoyage_article(destination)

    if article is None:
        return None

    text = clean_html(article["html"])
    chunks = chunk_text(text)

    return {
        "title": article["title"],
        "chunks": chunks,
        "source": (
            f"https://en.wikivoyage.org/wiki/"
            f"{article['title'].replace(' ','_')}"
        )
    }

def search_wikivoyage(destination,query,top_k=5):
    document = prepare_wikivoyage(destination)
    if document is None:
        return []

    chunks = document['chunks']
    if not chunks:
        return []

    vectorizer,document_vectors = build_tfidf_index(tuple(chunks))

    return retrieve_chunks(query=query, chunks=chunks,vectorizer=vectorizer, document_vectors=document_vectors, source=document['source'],top_k=top_k)

def retrieve_guides(destination, query, top_k=5):
    return search_wikivoyage(
        destination=destination,
        query=query,
        top_k=top_k
    )