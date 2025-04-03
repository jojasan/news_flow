from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from news_flow.main import NewsFlow

# In-memory store for tasks; note that this will be lost if the server restarts.
task_store = {}

class Task(BaseModel):
    id: Optional[str] = None
    news_urls: Optional[List[str]] = []
    num_starting_pool_news: Optional[int] = 2
    num_max_news: Optional[int] = 1
    topic: Optional[str] = ''
    perspective: Optional[str] = ''
    tone: Optional[str] = ''

app = FastAPI()

@app.post("/happifynews")
async def happify_news(task: Task, background_tasks: BackgroundTasks):
    # Validate that at least one of news_urls or topic is provided.
    if not task.news_urls and not task.topic:
        raise HTTPException(status_code=400, detail="Either news_urls or topic must be provided.")
    
    # Use the provided task.id if available; otherwise generate one.
    task_id = task.id if task.id is not None else str(len(task_store) + 1)
    task.id = task_id  # ensure the task carries this id
    
    # Initialize the task status as "starting".
    task_store[task_id] = {"status": "initializing", "result": None}
    
    # Schedule the background workflow.
    background_tasks.add_task(kickoff_workflow, task, task_id)
    
    # Return the task_id along with all the input values provided.
    return {
        "task_id": task_id,
        "status": "starting",
        "news_urls": task.news_urls,
        "num_starting_pool_news": task.num_starting_pool_news,
        "num_max_news": task.num_max_news,
        "topic": task.topic,
        "perspective": task.perspective,
        "tone": task.tone,
    }

def kickoff_workflow(task: Task, task_id: str):
    news_flow = NewsFlow()
    task_store[task_id] = {"status": "starting_workflow", "result": news_flow}
    # Build the inputs dictionary using the task_id as the id.
    inputs = {
        'id': task_id,  
        'num_starting_pool_news': task.num_starting_pool_news,
        'num_max_news': task.num_max_news,
        'current_date': '2025-03-07',
        'perspective': task.perspective,
        'tone': task.tone
    }
    # If news_urls are provided, include them; otherwise, use topic.
    if task.news_urls and len(task.news_urls) > 0:
        inputs['news_urls'] = task.news_urls
    elif task.topic:
        inputs['topic'] = task.topic

    # Execute the workflow and update the in-memory store.
    # (In a real scenario, this may take 30-40 minutes to complete.)
    result = news_flow.kickoff(inputs=inputs)
    task_store[task_id] = {"status": "completed", "result": news_flow}

@app.get("/happifynews/{task_id}")
def get_task_status(task_id: str):
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_store[task_id]
    result = task_info["result"].get_state().model_dump() # this is a pydantic object, so we need to serialize this
    
    # Handle case where result is still None (task hasn't completed or crashed early)
    if result is None:
        return {
            "task_id": task_id,
            "status": "in progress",
            "current_step": None,
            "news_list": None,
            "plan": None,
            "news_evidence": None,
            "counter_arguments": None,
            "articles": None,
            "flow_tokens": None,
        }

    current_step = result.get('current_step', '')
    is_complete = current_step == 'write_articles'

    return {
        "task_id": task_id,
        "status": "completed" if is_complete else "in progress",
        "current_step": current_step,
        "news_list": result.get('news_list'),
        "plan": result.get('plan'),
        "news_evidence": result.get('news_evidence'),
        "counter_arguments": result.get('counter_arguments'),
        "articles": result.get('articles'),
        "flow_tokens": result.get('flow_tokens'),
    }
