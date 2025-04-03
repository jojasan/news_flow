import json
import os
import re
from pydantic import BaseModel
from crewai.flow import Flow, listen, start, persist, router
from news_flow.crews.a_discover_crew.discover_crew import DiscoverCrew
from news_flow.crews.b_planning_crew.planning_crew import PlanningCrew
from news_flow.crews.c_research_crew.research_crew import ResearchCrew
from news_flow.crews.d_counterargs_crew.counterargs_crew import CounterArgumentsCrew
from news_flow.crews.e_writer_crew.writer_crew import WriterCrew
from news_flow.types import NewsList, NewsResearchPlan, SupportingEvidence, CounterArgumentSources, ConsolidatedNews, ConsolidatedNewsItem, CounterArguments
from typing import Dict, Any, List, Optional
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
import difflib
import string
from typing import List, Optional

class NewsState(BaseModel):
    id: str = ''
    start_from_method: Optional[str] = None # if you want to start from a specific method
    num_starting_pool_news: int = 2
    num_max_news: int = 1
    topic: str = '' 
    perspective: str = ''
    tone: str = ''
    news_list: Optional[NewsList] = None
    plan: List[NewsResearchPlan] = []
    news_evidence: List[SupportingEvidence] = []
    counter_arguments: List[CounterArgumentSources] = []
    articles: List[str] = []
    flow_tokens: Dict[str, Any] = {}
    current_date: str = ''
    language: str = 'en' # TODO not being used yet

@persist(persistence=SQLiteFlowPersistence(db_path='flow_states.db'))
class NewsFlow(Flow[NewsState]):

    @start()
    def initialize(self):
        print(f"Initializing flow with these parameters: {self.state}")
        # TODO add more initialization logic here
    
    @router(initialize)
    def load_state(self):
        print("Trying to load state from database...")
        if self.state.start_from_method:
            print(f"Starting from method: {self.state.start_from_method}")
            return self.state.start_from_method # generates the event of FINISHING this method (so the next method is the one that actually starts!)
        print("No start_from_method provided")
        return 'start_from_beginning' # if no start_from_method is provided, start from the beginning

    @listen('start_from_beginning')
    def discover_news(self):
        print("Discovering news")
        result = (
            DiscoverCrew().crew()
            .kickoff(inputs={"topic": self.state.topic, 
                             "num_starting_news": self.state.num_starting_pool_news, 
                             "num_max_news": self.state.num_max_news,
                             "current_date": self.state.current_date,
                             "perspective": self.state.perspective,})
        )
        
        print("Saving state variables")
        self.state.news_list = result.pydantic
        self.state.flow_tokens['discover_news'] = {"prompt_tokens": result.token_usage.prompt_tokens, 
                                                   "completion_tokens": result.token_usage.completion_tokens}
        save_flow_step_output(result.pydantic, 'news_list.json')

    @listen(discover_news)
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
        print(f"Planning research for {len(news_list_dicts)} news")

        results = (
            PlanningCrew().crew()
            .kickoff_for_each(inputs=news_list_dicts)
        )

        print("Saving state variables")
        prompt_tokens = 0
        completion_tokens = 0
        for plan in results:
            self.state.plan.append(plan.pydantic)
            save_flow_step_output(plan.pydantic, filename=f'research_plan.json', subfolder=plan.pydantic.news_title) 
            prompt_tokens += plan.token_usage.prompt_tokens
            completion_tokens += plan.token_usage.completion_tokens
        self.state.flow_tokens['plan_research'] = {"prompt_tokens": prompt_tokens, 
                                                   "completion_tokens": completion_tokens}

    @listen(plan_research)
    def research_news(self):
        print("Starting research...")
        for plan in self.state.plan: # iterate over each plan (one for each news item)
            print(f"--> Kicking off research crews for article: {plan.news_title}")
            
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

        print("----Finished researching news----")
        print("Saving state variables")
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

    @listen(research_news)
    def counter_args(self):
        print("Starting counter args...")
        for plan in self.state.plan: # iterate over each plan (one for each news item)
            print(f"--> Kicking off counterargs crews for article: {plan.news_title}")
            
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
        
        print("----Finished finding counterargs support----")
        print("Saving state variables")
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
        
    @listen(counter_args)
    def write_articles(self):
        print("Starting writing articles")
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


def kickoff():
    news_flow = NewsFlow()
    news_flow.kickoff(inputs={
        'id': 'test_jsonconsolidation', # use an id if you want to start from the latest checkpoint
        'start_from_method': 'counter_args', # use this parameter to start from a specific method (starts after this one)
        'num_starting_pool_news': 2,
        'num_max_news': 1,
        'topic': 'Depression in straight men between 30-50 years old',
        # 'topic': 'Articificial Intelligence business case ROI in Banks',
        # 'topic': 'Climate Change in Colombia',
        # 'topic': 'Economic outlook of Peru',
        'perspective': 'Positive, optimistic',
        'tone': 'Scientific, informative',
        'current_date': '2025-04-03'
    })

    print("------ Flow completed ------")
    total_cost = calculate_tokens_usage(news_flow.state.flow_tokens)
    print(f"-----> Total cost of the flow: ~${total_cost['total_costs']:.4f}")
    print(f"-----> Total tokens used: {total_cost['total_tokens']}")

def plot():
    news_flow = NewsFlow()
    news_flow.plot()
    
def calculate_tokens_usage(flow_tokens_usage: Dict[str, Any]) -> float:
    total_costs = 0
    total_tokens = 0
    percentage_gpt_4o = 0.2 # adjust this value to reflect the actual distribution
    cost_gpt4o_input = 2.5
    cost_gpt4o_output = 1.25
    cost_gpt40_mini_input = 0.150
    cost_gpt40_mini_output = 0.075
    average_input_cost = cost_gpt4o_input * percentage_gpt_4o + cost_gpt40_mini_input * (1 - percentage_gpt_4o)
    average_output_cost = cost_gpt4o_output * percentage_gpt_4o + cost_gpt40_mini_output * (1 - percentage_gpt_4o)
    for key, value in flow_tokens_usage.items():
        print(f"------ Calculating Flow '{key}' costs ------")
        print(f"Total '{key}' crew prompt tokens: {value['prompt_tokens']}")
        print(f"Total '{key}' crew completion tokens: {value['completion_tokens']}")
        costs = (average_input_cost * value['prompt_tokens'] / 1_000_000) + (average_output_cost * value['completion_tokens'] / 1_000_000)
        print(f"Total '{key}'  crew costs: ~${costs:.4f}")
        total_costs += costs
        total_tokens += value['prompt_tokens'] + value['completion_tokens']
    return {"total_costs": total_costs, "total_tokens": total_tokens}

def save_flow_step_output(flow_step_output: Any, filename: str, subfolder: str = None):
    """Serialize and save an object to a JSON or Markdown file in the 'outputs' directory or a specified subfolder."""
    
    # Check the file extension first.
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ('.json', '.md'):
        print(f"Unsupported file extension: {ext}")
        return

    base_output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    
    # Sanitize subfolder name if provided.
    if subfolder:
        subfolder = re.sub(r'[<>:"/\\|?*]', '_', subfolder)
        output_dir = os.path.join(base_output_dir, subfolder)
    else:
        output_dir = base_output_dir

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    if ext == '.json':
        # For JSON files, process the data accordingly.
        # Convert Pydantic models to a dictionary.
        if isinstance(flow_step_output, BaseModel):
            flow_step_output = flow_step_output.model_dump()

        # If the data is a string, try parsing it into JSON.
        if isinstance(flow_step_output, str):
            try:
                flow_step_output = json.loads(flow_step_output)
            except json.JSONDecodeError:
                pass  # Leave it as a string if parsing fails.

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(flow_step_output, f, indent=4, ensure_ascii=False)
        except TypeError as e:
            print(f"Error saving JSON file {filename}: {e}")

    elif ext == '.md':
        # For Markdown files, assume the input is already a markdown string.
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(flow_step_output)
        except Exception as e:
            print(f"Error saving Markdown file {filename}: {e}")

def normalize_title(title: str) -> str:
    """Normalize the title by lowercasing and removing punctuation and extra whitespace."""
    translator = str.maketrans('', '', string.punctuation)
    return title.lower().translate(translator).strip()

def find_close_key(norm_title: str, keys: List[str], threshold: float = 0.95) -> Optional[str]:
    """
    Look for an existing key in `keys` that is very similar to `norm_title`.
    Returns the matching key if found, otherwise None.
    """
    for key in keys:
        if difflib.SequenceMatcher(None, norm_title, key).ratio() > threshold:
            return key
    return None

def consolidate_news_json(
    evidence_list: List[SupportingEvidence],
    news_research_plans: List[NewsResearchPlan],
    counter_arguments_list: List[CounterArgumentSources],
    news_list: NewsList
) -> dict:
    # Use a dictionary keyed by normalized title.
    news_dict = {}
    
    # Pre-populate using the news_list from NewsList
    for news in news_list.news_list:
        norm_title = normalize_title(news.news_title)
        existing_key = find_close_key(norm_title, list(news_dict.keys()))
        key = existing_key if existing_key is not None else norm_title
        
        if key not in news_dict:
            news_dict[key] = {
                "news_title": news.news_title,
                "summary": news.summary,
                "source_url": news.source_url,
                "content": news.content,
                "supporting_evidence": [],
                "datapoints": [],
                "counter_argument_sources": []
            }
    
    def ensure_entry(original_title: str) -> str:
        """Ensure that an entry exists for the given title, returning the normalized key."""
        norm_title = normalize_title(original_title)
        existing_key = find_close_key(norm_title, list(news_dict.keys()))
        key = existing_key if existing_key is not None else norm_title
        if key not in news_dict:
            news_dict[key] = {
                "news_title": original_title,
                "summary": None,
                "source_url": None,
                "content": None,
                "supporting_evidence": [],
                "datapoints": [],
                "counter_argument_sources": []
            }
        return key

    # Process SupportingEvidence objects
    for evidence in evidence_list:
        key = ensure_entry(evidence.news_title)
        news_dict[key]["supporting_evidence"].append(evidence.dict())
    
    # Process NewsResearchPlan objects to extract datapoints
    for plan in news_research_plans:
        key = ensure_entry(plan.news_title)
        news_dict[key]["datapoints"].append(plan.key_datapoints.dict())
    
    # Process CounterArgumentSources objects
    for cas in counter_arguments_list:
        key = ensure_entry(cas.news_title)
        news_dict[key]["counter_argument_sources"].append(cas.dict())
    
    consolidated_list = list(news_dict.values())
    return {"news_list": consolidated_list}

def cleanup_consolidated_json(data: dict) -> dict:
    """
    Cleans up a consolidated JSON structure by removing duplicate fields from nested records.
    For each news item in data["news_list"], if nested objects have a field (like "news_title"
    or "source_url") that is identical to the top-level value, that field is removed.
    
    Args:
        data: The consolidated JSON dictionary.
    
    Returns:
        The cleaned-up JSON dictionary.
    """
    for news_item in data.get("news_list", []):
        top_title = news_item.get("news_title")
        top_source = news_item.get("source_url")
        
        # Clean up duplicates in the supporting_evidence list.
        for evidence in news_item.get("supporting_evidence", []):
            # Remove "news_title" if it matches the top-level news_title.
            if evidence.get("news_title") == top_title:
                evidence.pop("news_title", None)
            # Remove "source_url" if it matches the top-level source_url.
            if evidence.get("source_url") == top_source:
                evidence.pop("source_url", None)
        
        # Clean up duplicates in the counter_argument_sources list.
        for cas in news_item.get("counter_argument_sources", []):
            if cas.get("news_title") == top_title:
                cas.pop("news_title", None)
    
    return data

if __name__ == "__main__":
    kickoff()