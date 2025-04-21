from crewai.llm import LLM

def o4_mini_with_gemini_flash_fallback():
    """
    LLM Configuration:
    - Model: openai/o4-mini
    - Retries: 3
    - Fallbacks:
        1. openrouter/google/gemini-2.5-flash-preview (via OpenRouter)
        2. openai/gpt-4o
    """
    return LLM(
        model="openai/o4-mini", 
        num_retries=3,
        fallbacks=[
            {
                "model": "openrouter/google/gemini-2.5-flash-preview",
                "base_url": "https://openrouter.ai/api/v1",
            },
            {
                "model": "openai/gpt-4.1"
            },
        ],
    )

def gemini_flash_with_gpt4_1_mini_fallback():
    """
    LLM Configuration:
    - Model: openrouter/google/gemini-2.5-flash-preview (via OpenRouter)
    - Retries: 3
    - Fallbacks:
        1. openai/gpt-4.1-mini
        2. groq/llama-3.3-70b-versatile
    """
    return LLM(
        model="openrouter/google/gemini-2.5-flash-preview", 
        base_url="https://openrouter.ai/api/v1",
        num_retries=3,
        fallbacks=[
            {
                "model": "openai/gpt-4.1-mini",
            },
            {
                "model": "groq/llama-3.3-70b-versatile",
            }
        ],
    )

def o4_mini_with_gpt4_1_fallback():
    """
    LLM Configuration:
    - Model: openai/o4-mini
    - Retries: 3
    - Fallbacks:
        1. openai/gpt-4.1
    """
    return LLM(
        model="openai/o4-mini",
        num_retries=3,
        fallbacks=[
            {
                "model": "openai/gpt-4.1"
            },
        ],
    )

def gpt4_1_mini_with_gemini_flash_fallback():
    """
    LLM Configuration:
    - Model: openai/gpt-4.1-mini
    - Retries: 3
    - Fallbacks:
        1. openrouter/google/gemini-2.5-flash-preview (via OpenRouter)
        2. groq/llama-3.3-70b-versatile
    """
    return LLM(
        model="openai/gpt-4.1-mini", 
        num_retries=3,
        fallbacks=[
            {
                "model": "openrouter/google/gemini-2.5-flash-preview",
                "base_url": "https://openrouter.ai/api/v1",
            },
            {
                "model": "groq/llama-3.3-70b-versatile"
            },
        ],
    )

def o4_mini_high_reasoning():
    """
    LLM Configuration:
    - Model: openai/o4-mini
    - Reasoning Effort: high
    """
    return LLM(
        model='openai/o4-mini',
        reasoning_effort="high",
    ) 