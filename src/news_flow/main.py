# Standard Library Imports
import json
import logging
from typing import Dict, Any, List, Optional

# Third-Party Imports (CrewAI, Pydantic, etc.)
from pydantic import BaseModel
from crewai.flow import Flow, listen, start, persist, router, or_

# Application-Specific Imports (News Flow Modules)
from news_flow.crews import *
from news_flow.types import (
    NewsList, NewsResearchPlan, SupportingEvidence, CounterArgumentSources, CritiqueList
)

# Local Module Imports (Helper Functions)
from news_flow.utils import (
    calculate_tokens_usage,
    save_flow_step_output,
    consolidate_news_json,
    cleanup_consolidated_json,
)

# --- Apply Patches ---
# Patch SQLite persistence for robust JSON handling
from news_flow.crewai_extensions import SQLiteFlowPersistenceJSON
# Patch LiteLLM completion to handle empty responses
# import news_flow.crewai_extensions # IMPORTANT: Importing this executes the patch
# if not news_flow.crewai_extensions.is_litellm_patched():
    # Use logging for warnings
#     logging.warning("\nWARNING: LiteLLM empty response patch FAILED to apply! Fallbacks for empty responses might not work.\n")
# # Patch JSON parsing (Optional - uncomment if needed)
# import crewai.utilities.converter
# from news_flow.crewai_extensions import patched_handle_partial_json
# logging.info("Applying monkey patch to crewai.utilities.converter.handle_partial_json...") # Changed print to logging.info
# crewai.utilities.converter.handle_partial_json = patched_handle_partial_json
# logging.info("Patch applied.") # Changed print to logging.info
# --- End Patches ---

class NewsState(BaseModel):
    id: str = ''
    start_from_method: Optional[str] = None # if you want to start from a specific method
    num_starting_pool_news: int = 2
    num_max_news: int = 1
    topic: str = '' 
    urls: List[str] = []
    perspective: str = 'Optimistic, positive, happy'
    tone: str = ''
    news_list: Optional[NewsList] = None
    plan: List[NewsResearchPlan] = []
    critiques: List[CritiqueList] = []
    news_evidence: List[SupportingEvidence] = []
    counter_arguments: List[CounterArgumentSources] = []
    articles: List[str] = []
    flow_tokens: Dict[str, Any] = {}
    current_date: str = ''
    language: str = 'en' # TODO not being used yet
    current_step: str = ''  # New field to track the current step

@persist(persistence=SQLiteFlowPersistenceJSON(db_path='flow_states.db'))
class NewsFlow(Flow[NewsState]):

    @start()
    def initialize(self):
        # TODO beautify printing, remove content, summarize Lists (just give counts)
        logging.info("Initializing flow with these parameters: %s", self.state)
        # TODO add more initialization logic here: load via config how much paralellism to use, log verbosity, crewai verbosity
        if self.state.current_step == '':
            self.state.current_step = "initialize"
    
    @router(initialize)
    def load_state(self):
        logging.info("Trying to load state from database...")
        if self.state.current_step != 'initialize':
            logging.info("Resuming from latest step: %s", self.state.current_step)
            return self.state.current_step  # Resume from the latest checkpoint. TODO: actually need to map from method names to event names
        elif self.state.topic:
            logging.info("Topic provided, starting with discover step.")
            return 'discover'
        elif self.state.urls:
            logging.info("URLs provided, starting with scrape step.")
            return 'scrape'
        else:
            logging.error("No valid starting point found. Provide either a topic, URLs, or resume from a previous checkpoint.")
            raise ValueError("Unable to determine the starting point for the News Flow. Please provide a topic or URLs, or ensure a valid checkpoint exists.")

    @listen('discover')
    def discover_news(self):
        logging.info("Discovering news")
        result = (
            DiscoverCrew().crew()
            .kickoff(inputs={"topic": self.state.topic, 
                             "num_starting_pool_news": self.state.num_starting_pool_news, 
                             "num_max_news": self.state.num_max_news,
                             "current_date": self.state.current_date,
                             "perspective": self.state.perspective,})
        )
        
        logging.info("Saving state variables for discover_news")
        self.state.news_list = result.pydantic
        self.state.flow_tokens['discover_news'] = {"prompt_tokens": result.token_usage.prompt_tokens, 
                                                   "completion_tokens": result.token_usage.completion_tokens}
        save_flow_step_output(result.pydantic, 'news_list.json')
        self.state.current_step = "discover_news"

    @listen('scrape')
    def scrape_news(self):
        logging.info("Scraping news")
        result = (
            ScrapeCrew().crew()
            .kickoff(inputs={"urls": self.state.urls[0]}) # for now only one url
        )

        logging.info("Saving state variables for scrape_news")
        self.state.news_list = result.pydantic
        self.state.flow_tokens['scrape_news'] = {"prompt_tokens": result.token_usage.prompt_tokens, 
                                                   "completion_tokens": result.token_usage.completion_tokens}
        save_flow_step_output(result.pydantic, 'news_list.json')
        self.state.current_step = "scrape_news"


    @listen(or_(scrape_news, discover_news))
    def critique_news(self):
        logging.info("Starting critique...")
        for news in self.state.news_list.news_list:
            logging.info("--> Kicking off critique crews for article: %s", news.news_title)
            critique_dicts = [
                {
                    "news_title": news.news_title,
                    "summary": news.summary,
                    "source": news.source_url,
                    "perspective": self.state.perspective,
                    "article_content": news.content
                }
            ]
            results = (
                CritiqueCrew().crew()
                .kickoff_for_each(inputs=critique_dicts)
            )  

            logging.info("Saving state variables for critique_news")
            for result in results:
                self.state.critiques.append(result.pydantic)
                save_flow_step_output(result.pydantic, filename=f'critique.json', subfolder=result.pydantic.news_title) 
            self.state.current_step = "critique_news"


    @listen(critique_news)
    def plan_research(self):

        # preparing the news list for the next crew
        news_list_dicts = [
            {
                "news_title": news.news_title,
                "summary": news.summary,
                "source": news.source_url,
                "topic": self.state.topic,
                "perspective": self.state.perspective,
                "article_content": news.content
            }
            for news in self.state.news_list.news_list
        ]
        logging.info("Planning research for %d news items", len(news_list_dicts))

        results = (
            PlanningCrew().crew()
            .kickoff_for_each(inputs=news_list_dicts)
        )

        logging.info("Saving state variables for plan_research")
        prompt_tokens = 0
        completion_tokens = 0
        for plan in results:
            self.state.plan.append(plan.pydantic)
            save_flow_step_output(plan.pydantic, filename=f'research_plan.json', subfolder=plan.pydantic.news_title) 
            prompt_tokens += plan.token_usage.prompt_tokens
            completion_tokens += plan.token_usage.completion_tokens
        self.state.flow_tokens['plan_research'] = {"prompt_tokens": prompt_tokens, 
                                                   "completion_tokens": completion_tokens}
        self.state.current_step = "plan_research"

    @listen(plan_research)
    def research_news(self):
        logging.info("Starting research...")
        for plan in self.state.plan: # iterate over each plan (one for each news item)
            logging.info("--> Kicking off research crews for article: %s", plan.news_title)
            
            key_idea_dicts = [
                {
                    "news_title": plan.news_title,
                    "source": plan.source_url,
                    "perspective": self.state.topic,
                    "key_idea": idea.key_idea,
                    "rationale": idea.rationale
                    #"topic": self.state.topic,
                }
                for idea in plan.key_ideas.ideas
            ]  
            
            results = (
                ResearchCrew().crew()
                .kickoff_for_each(inputs=key_idea_dicts) # kick off crews for each key idea and save results
            )

        logging.info("----Finished researching news----")
        logging.info("Saving state variables for research_news")
        prompt_tokens = 0
        completion_tokens = 0
        i = 0
        for evidence in results:
            self.state.news_evidence.append(evidence.pydantic)
            save_flow_step_output(evidence.pydantic, filename=f'evidence_{i}.json', subfolder=evidence.pydantic.news_title) 
            prompt_tokens += evidence.token_usage.prompt_tokens
            completion_tokens += evidence.token_usage.completion_tokens
            i += 1
        self.state.flow_tokens['research_news'] = {"prompt_tokens": prompt_tokens, 
                                                   "completion_tokens": completion_tokens}
        self.state.current_step = "research_news"

    @listen(research_news)
    def counter_args(self):
        logging.info("Starting counter args...")
        for plan in self.state.plan: # iterate over each plan (one for each news item)
            logging.info("--> Kicking off counterargs crews for article: %s", plan.news_title)
            
            counterargs_dicts = [
                {
                    "news_title": plan.news_title,
                    "source": plan.source_url,
                    "perspective": self.state.perspective,
                    "counterargument": counter_arg.counter_argument,
                    "counter_rationale": counter_arg.rationale
                }
                for counter_arg in plan.counter_arguments.counter_arguments
            ]  
            
            results = (
                CounterArgumentsCrew().crew()
                .kickoff_for_each(inputs=counterargs_dicts) # kick off crews for each key idea and save results
            )
        
        logging.info("----Finished finding counterargs support----")
        logging.info("Saving state variables for counter_args")
        prompt_tokens = 0
        completion_tokens = 0
        i = 0
        for counterargs in results:
            self.state.counter_arguments.append(counterargs.pydantic)
            save_flow_step_output(counterargs.pydantic, filename=f'counterargs_{i}.json', subfolder=counterargs.pydantic.news_title) 
            prompt_tokens += counterargs.token_usage.prompt_tokens
            completion_tokens += counterargs.token_usage.completion_tokens
            i += 1
        self.state.flow_tokens['counter_args'] = {"prompt_tokens": prompt_tokens, 
                                                  "completion_tokens": completion_tokens}
        self.state.current_step = "counter_args"
        
    @listen(counter_args)
    def write_articles(self):
        logging.info("Starting writing articles")
        # Use the new function which returns a plain dict.
        news_json = consolidate_news_json(
            self.state.news_evidence,
            self.state.plan,
            self.state.counter_arguments,
            self.state.news_list
        )
        save_flow_step_output(cleanup_consolidated_json(news_json), filename='final_research_output.json')

        # Iterate over the consolidated news items in the JSON structure.
        for news_item in news_json["news_list"]:
            # Create JSON string representations of the lists from the dict.
            evidence_str = json.dumps(news_item["supporting_evidence"])
            datapoints_str = json.dumps(news_item["datapoints"])
            counterargs_str = json.dumps(news_item["counter_argument_sources"])
            
            writer_dics = [
                {
                    "title": news_item["news_title"],
                    "url": news_item["source_url"],
                    "original_content": news_item["content"],
                    "evidence": evidence_str,
                    "datapoints": datapoints_str,
                    "counterarguments": counterargs_str,
                    "perspective": self.state.perspective,
                    "tone": self.state.tone,
                }
            ]
            
            results = (
                WriterCrew().crew()
                .kickoff_for_each(inputs=writer_dics)
            )
            
            prompt_tokens = 0
            completion_tokens = 0
            i = 0
            for article in results:
                self.state.articles.append(article.raw)
                save_flow_step_output(article.raw, filename=f'article_{i}.md')
                prompt_tokens += article.token_usage.prompt_tokens
                completion_tokens += article.token_usage.completion_tokens
                i += 1
            self.state.flow_tokens['write_articles'] = {
                "prompt_tokens": prompt_tokens, 
                "completion_tokens": completion_tokens
            }
        self.state.current_step = "write_articles"

    def get_state(self) -> NewsState:
        """
        Returns the complete state of the workflow
        """
        return self.state

def kickoff():
    # Basic logging configuration
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    news_flow = NewsFlow()
    news_flow.kickoff(inputs={
        'id': 'mac_yaml_test', # use an id if you want to start from the latest checkpoint
        'num_starting_pool_news': 2,
        'num_max_news': 1,
        'perspective': 'Positive, optimistic',
        'tone': 'Scientific, informative',
        'current_date': '2025-04-05',
        # 'topic': 'Depression in straight men between 30-50 years old',
        # 'topic': 'Articificial Intelligence business case ROI in Banks',
        # 'topic': 'Climate Change in Colombia',
        'topic': 'Economic outlook of Peru',
        # 'urls': ['https://www.foxbusiness.com/media/gold-soars-dollar-sinks-forbes-warns-us-headed-towards-1970s-style-inflation-nightmare']
        #'urls': ['https://edition.cnn.com/2025/04/05/business/trump-reciprocal-tariffs-real-numbers/index.html']
        #'start_from_method': 'counter_args', # use this parameter to start from a specific method (starts after this one)
    })

    # Use logging for completion messages
    logging.info("------ Flow completed ------")
    total_cost = calculate_tokens_usage(news_flow.state.flow_tokens)
    logging.info(f"-----> Total cost of the flow: ~${total_cost['total_costs']:.4f}")
    logging.info(f"-----> Total tokens used: {total_cost['total_tokens']}")

def plot():
    news_flow = NewsFlow()
    news_flow.plot()

if __name__ == "__main__":
    kickoff()