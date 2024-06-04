from flask import Flask, request, render_template
from dotenv import load_dotenv
import openai
import requests
import os
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser

# .env fájl beolvasása
load_dotenv()

# OpenAI API kulcs
openai.api_key = os.getenv('OPENAI_API_KEY')

# Bing Search API kulcs és endpoint
bing_api_key = os.getenv('BING_API_KEY')
bing_endpoint = 'https://api.bing.microsoft.com/v7.0/search'

app = Flask(__name__)

# Emlékezzen az utolsó 10 kérdésre és válaszra
message_history = []

def format_for_search(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Format the user's question for an effective web search to find up-to-date and specific information."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=50,
        n=1,
        stop=None,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response['choices'][0]['message']['content'].strip()

def search_with_bing(query):
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {"q": query, "sortBy": "date"}  # Frissebb találatok előnyben részesítése
    response = requests.get(bing_endpoint, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def fetch_webpage_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except requests.RequestException as e:
        return f"Error occurred while fetching the webpage content: {str(e)}"

def extract_top_search_results(results):
    top_results = []
    if results.get("webPages", {}).get("value"):
        for result in results["webPages"]["value"]:
            url = result.get("url")
            content = fetch_webpage_content(url)
            top_results.append((result.get("dateLastCrawled"), content))  # Tároljuk a találatok időpontját is
    # Rendezzük a találatokat a legfrissebbek alapján
    top_results.sort(key=lambda x: parser.parse(x[0]), reverse=True)
    return [content for _, content in top_results]

def combine_with_gpt_knowledge(snippets, original_question):
    prompt = f"Here are some search results snippets:\n\n"
    for snippet in snippets:
        prompt += f"- {snippet}\n"
    prompt += f"\nBased on these snippets and your own knowledge, provide a brief and precise answer to the question, including specific values where applicable. If exact values are not available, provide an approximate value: {original_question}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300,
        n=1,
        stop=None,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response['choices'][0]['message']['content'].strip()

def chat_with_gpt(prompt, messages):
    messages.append({"role": "user", "content": prompt})
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            n=1,
            stop=None,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        assistant_message = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        assistant_message = f"Error occurred while connecting to OpenAI: {str(e)}"
    messages.append({"role": "assistant", "content": assistant_message})
    return assistant_message

@app.route("/", methods=["GET", "POST"])
def index():
    global message_history
    response = []
    if request.method == "POST":
        user_input = request.form["question"]
        
        # Tájékoztatás minden kérdés előtt
        inform_prompt = (
            "You are connected to the internet. "
            "Your knowledge base is up to date until 2023. "
            "For accurate and up-to-date information, you can use the internet."
        )
        
        message_history.append({"role": "system", "content": inform_prompt})
        
        # Eldönti, hogy szükséges-e az internetes keresés a GPT modell segítségével
        decision_prompt = (
            "You are a helpful assistant. Determine if the user's question requires up-to-date information "
            "that might change frequently, like current prices, weather, or time. If yes, respond with 'yes'. "
            "Otherwise, respond with 'no'."
        )
        internet_needed = chat_with_gpt(decision_prompt + "\n\n" + user_input, message_history)
        
        if 'yes' in internet_needed.lower():
            formatted_query = format_for_search(user_input)
            search_results = search_with_bing(formatted_query)
            snippets = extract_top_search_results(search_results)
            answer = combine_with_gpt_knowledge(snippets, user_input)
        else:
            answer = chat_with_gpt(user_input, message_history)
        
        response.append({"question": user_input, "answer": answer})
        message_history.append({"role": "user", "content": user_input})
        message_history.append({"role": "assistant", "content": answer})
        
        # Csak az utolsó 20 üzenetet tartsa meg
        if len(message_history) > 20:
            message_history = message_history[-20:]
        
    return render_template("index.html", response=response)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
