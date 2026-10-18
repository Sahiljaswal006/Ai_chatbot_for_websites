import os
import re
import time

from dotenv import load_dotenv
from groq import Groq

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from rag import retrieve_context

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

URLS = [
    "https://techagastyaan.agastyaan.com/"
]

LLM_TXT_PATH = "llm.txt"


def read_llm_txt(path=LLM_TXT_PATH):

    if not os.path.exists(path):
        return ""

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def scrape_website(urls=URLS):

    chrome_options = Options()

    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(
            ChromeDriverManager().install()
        ),
        options=chrome_options
    )

    all_text = []

    try:

        for url in urls:

            driver.get(url)

            time.sleep(3)

            soup = BeautifulSoup(
                driver.page_source,
                "html.parser"
            )

            for tag in soup(
                ["script", "style", "noscript", "svg"]
            ):
                tag.decompose()

            text = soup.get_text(separator="\n")

            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]+", " ", text)

            all_text.append(text)

    finally:
        driver.quit()

    return "\n\n".join(all_text)


def ask(question, chat_history):

    client = Groq(api_key=API_KEY)

    context = retrieve_context(question)

    system_prompt = f"""
You are TechAgastyaan's AI assistant.

Use ONLY the provided context.

If the answer is not found in the context, reply:

"I don't have that information. Please contact us directly."

Context:
{context}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.extend(chat_history)

    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )

    return response.choices[0].message.content