import json
import os
import re
import difflib
import string
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

def calculate_tokens_usage(flow_tokens_usage: Dict[str, Any]) -> dict:
    total_costs = 0
    total_tokens = 0
    percentage_gpt_4o = 0.2  # adjust this value to reflect the actual distribution
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
        print(f"Total '{key}' crew costs: ~${costs:.4f}")
        total_costs += costs
        total_tokens += value['prompt_tokens'] + value['completion_tokens']
    return {"total_costs": total_costs, "total_tokens": total_tokens}

def save_flow_step_output(flow_step_output: Any, filename: str, subfolder: str = None):
    """Serialize and save an object to a JSON or Markdown file in the 'outputs' directory or a specified subfolder."""
    
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ('.json', '.md'):
        print(f"Unsupported file extension: {ext}")
        return

    base_output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    if subfolder:
        subfolder = re.sub(r'[<>:"/\\|?*]', '_', subfolder)
        output_dir = os.path.join(base_output_dir, subfolder)
    else:
        output_dir = base_output_dir

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    if ext == '.json':
        if isinstance(flow_step_output, BaseModel):
            flow_step_output = flow_step_output.model_dump()
        if isinstance(flow_step_output, str):
            try:
                flow_step_output = json.loads(flow_step_output)
            except json.JSONDecodeError:
                pass
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(flow_step_output, f, indent=4, ensure_ascii=False)
        except TypeError as e:
            print(f"Error saving JSON file {filename}: {e}")
    elif ext == '.md':
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
    for key in keys:
        if difflib.SequenceMatcher(None, norm_title, key).ratio() > threshold:
            return key
    return None

def consolidate_news_json(
    evidence_list: List[Any],
    news_research_plans: List[Any],
    counter_arguments_list: List[Any],
    news_list: Any
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

    for evidence in evidence_list:
        key = ensure_entry(evidence.news_title)
        news_dict[key]["supporting_evidence"].append(evidence.dict())
    
    for plan in news_research_plans:
        key = ensure_entry(plan.news_title)
        news_dict[key]["datapoints"].append(plan.key_datapoints.dict())
    
    for cas in counter_arguments_list:
        key = ensure_entry(cas.news_title)
        news_dict[key]["counter_argument_sources"].append(cas.dict())
    
    consolidated_list = list(news_dict.values())
    return {"news_list": consolidated_list}

def cleanup_consolidated_json(data: dict) -> dict:
    """
    Cleans up a consolidated JSON structure by removing duplicate fields from nested records.
    """
    for news_item in data.get("news_list", []):
        top_title = news_item.get("news_title")
        top_source = news_item.get("source_url")
        
        for evidence in news_item.get("supporting_evidence", []):
            if evidence.get("news_title") == top_title:
                evidence.pop("news_title", None)
            if evidence.get("source_url") == top_source:
                evidence.pop("source_url", None)
        
        for cas in news_item.get("counter_argument_sources", []):
            if cas.get("news_title") == top_title:
                cas.pop("news_title", None)
    
    return data
