import os
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, FirecrawlScrapeWebsiteTool, BraveSearchTool
from news_flow.tools.tavily import TavilySearchTool, TavilyScrapeTool
# Load environment variables from .env file first
load_dotenv()

# Initialize tools once here
serper_search = SerperDevTool()
brave_search = BraveSearchTool()
firecrawl = FirecrawlScrapeWebsiteTool(
    api_key=os.getenv("FIRECRAWL_API_KEY")
)
tavily_search = TavilySearchTool(topic="general", include_images=True, include_raw_content=True)
tavily_scrape = TavilyScrapeTool(include_images=True)

# You can also potentially initialize your custom tools here if needed elsewhere
# from .openai_websearch import OpenAIWebSearchTool
# from .tavily import TavilyTool
# openai_web_search = OpenAIWebSearchTool()
# tavily_tool = TavilyTool()

# Explicitly define what gets imported with 'from . import *' if desired,
# or rely on direct imports like 'from news_flow.tools import serper_search'
__all__ = ["serper_search", "brave_search", "firecrawl", "tavily_search", "tavily_scrape"]
