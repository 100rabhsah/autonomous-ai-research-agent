import os
import requests
from bs4 import BeautifulSoup
import chromadb
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB for memory
chroma_client = chromadb.PersistentClient(path="./memory_db")
collection = chroma_client.get_or_create_collection(name="research_data")

# Initialize OpenAI GPT
llm = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=OPENAI_API_KEY)

# Function to fetch and parse web content
def fetch_web_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    return None

# Function to process and summarize content
def summarize_content(content):
    prompt = f"Summarize this research content in 3 bullet points:\n{content[:3000]}"  # Limit to 3000 chars for prompt
    summary = llm([HumanMessage(content=prompt)]).content
    return summary

# Function to autonomously search, extract, and store knowledge
def research_topic(query):
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num=3"
    search_results = fetch_web_content(search_url)

    if not search_results:
        return "Failed to retrieve search results."

    # Extract the first 3 links (simplified scraping example)
    soup = BeautifulSoup(search_results, "html.parser")
    links = [a["href"] for a in soup.find_all("a", href=True) if "url?q=" in a["href"]][:3]

    summaries = []
    for link in links:
        content = fetch_web_content(link)
        if content:
            summary = summarize_content(content)
            summaries.append({"url": link, "summary": summary})
            collection.add(documents=[summary], ids=[link])  # Store in memory

    return summaries

# Example usage
if __name__ == "__main__":
    topic = "latest research on AI ethics"
    results = research_topic(topic)
    for res in results:
        print(f"\n🔗 {res['url']}\n📌 Summary: {res['summary']}")
