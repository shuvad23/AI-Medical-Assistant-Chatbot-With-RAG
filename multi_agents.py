from langchain_core.tools import tool
import requests
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper

load_dotenv()

@tool()
def find_rxcui(drug_name: str) -> str:
    """
    Fetch RxCUI for a given drug using RxNorm API.
    """
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        rxcui = data.get('idGroup', {}).get('rxnormId', [])
        return rxcui[0] if rxcui else "Not Found"
    return "Error fetching drug information"


@tool()
def get_covid_stats(country:str ="Bangladesh") -> dict:
    """
    Fetches COVID-19 stats for a given country.
    """
    url = f"https://disease.sh/v3/covid-19/countries/{country}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            "Country": data["country"],
            "Cases": data["cases"],
            "Recovered": data["recovered"],
            "Deaths": data["deaths"]
        }
    else:
        return {"error": "Failed to fetch data"}


@tool()
def get_drug_info_openfda(drug_name:str="aspirin") -> str:
    """
    Returns drug indications from OpenFDA.
    """
    url = f"https://api.fda.gov/drug/label.json?search=generic_name:{drug_name}&limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['results'][0]['indications_and_usage'][0]
    else:
        return "Drug info not found."


@tool()
def get_myhealthfinder_content(topic:str="diabetes") -> str:
    """
    Fetches health topics from MyHealthfinder API.
    Returns joined string of topic titles.
    """
    url = f"https://health.gov/myhealthfinder/api/v3/topicsearch.json?keyword={topic}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        topics = data['Result']['Resources']['Resource']
        # Join list to a string for LangChain tool compatibility
        return "\n".join([t['Title'] for t in topics])
    else:
        return "No health info found."


@tool()
def get_wikipedia_summary(topic:str="diabetes") -> str:
    """
    Fetches a summary of the topic from Wikipedia.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['extract']
    else:
        return "No summary found."


def get_google_search_results(query: str) -> str:
    """
    Perform a Google search and return the top result.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': os.getenv('GOOGLE_API_KEY'),
        'cx': os.getenv('GOOGLE_CX'),
        'q': query
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'items' in data and len(data['items']) > 0:
            return data['items'][0]['link']
        else:
            return "No results found."
    else:
        return f"Error: {response.status_code}"

@tool
def google_search(query: str) -> str:
    """Returns the top Google search result for a given query."""
    return get_google_search_results(query)


arxiv = ArxivAPIWrapper()
@tool
def search_arxiv(query: str) -> str:
    """Search Arxiv for scientific papers on a topic."""
    return arxiv.run(query)

tavily_base = TavilySearchResults(k=3)

@tool
def search_tavily(query: str) -> str:
    """Search the web using Tavily for a given query."""
    return tavily_base.run(query)