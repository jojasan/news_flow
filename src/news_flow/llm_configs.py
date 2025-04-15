from crewai.llm import LLM

def o3_mini_with_gemini_flash_fallback():
    """
    LLM Configuration:
    - Model: openai/o3-mini
    - Retries: 3
    - Fallbacks:
        1. openrouter/google/gemini-2.0-flash-001 (via OpenRouter)
        2. openai/gpt-4o
    """
    return LLM(
        model="openai/o3-mini", 
        num_retries=3,
        fallbacks=[
            {
                "model": "openrouter/google/gemini-2.0-flash-001",
                "base_url": "https://openrouter.ai/api/v1",
            },
            {
                "model": "openai/gpt-4o"
            },
        ],
    )

def gemini_flash_with_gpt4o_mini_fallback():
    """
    LLM Configuration:
    - Model: openrouter/google/gemini-2.0-flash-001 (via OpenRouter)
    - Retries: 3
    - Fallbacks:
        1. openai/gpt-4o-mini
        2. groq/llama-3.3-70b-versatile
    """
    return LLM(
        model="openrouter/google/gemini-2.0-flash-001", 
        base_url="https://openrouter.ai/api/v1",
        num_retries=3,
        fallbacks=[
            {
                "model": "openai/gpt-4o-mini",
            },
            {
                "model": "groq/llama-3.3-70b-versatile",
            }
        ],
    )

def o3_mini_with_gpt4o_fallback():
    """
    LLM Configuration:
    - Model: openai/o3-mini
    - Retries: 3
    - Fallbacks:
        1. openai/gpt-4o
    """
    return LLM(
        model="openai/o3-mini",
        num_retries=3,
        fallbacks=[
            {
                "model": "openai/gpt-4o"
            },
        ],
    )

def gpt4o_mini_with_gemini_flash_fallback():
    """
    LLM Configuration:
    - Model: openai/gpt-4o-mini
    - Retries: 3
    - Fallbacks:
        1. openrouter/google/gemini-2.0-flash-001 (via OpenRouter)
        2. groq/llama-3.3-70b-versatile
    """
    return LLM(
        model="openai/gpt-4o-mini", 
        num_retries=3,
        fallbacks=[
            {
                "model": "openrouter/google/gemini-2.0-flash-001",
                "base_url": "https://openrouter.ai/api/v1",
            },
            {
                "model": "groq/llama-3.3-70b-versatile"
            },
        ],
    )

def gpt4o_mini_with_gemini_flash_fallback():
    """
    LLM Configuration:
    - Model: openai/gpt-4o-mini
    - Retries: 3
    - Fallbacks:
        1. openrouter/google/gemini-2.0-flash-001 (via OpenRouter)
        2. openrouter/meta-llama/llama-3.3-70b-instruct (via OpenRouter)
    """
    return LLM(
        model="openai/gpt-4o-mini", 
        num_retries=3,
        fallbacks=[
            {
                "model": "openrouter/google/gemini-2.0-flash-001",
                "base_url": "https://openrouter.ai/api/v1",
            },
            {
                "model": "openrouter/meta-llama/llama-3.3-70b-instruct",
                "base_url": "https://openrouter.ai/api/v1",
            },
        ],
    )

def o3_mini_high_reasoning():
    """
    LLM Configuration:
    - Model: openai/o3-mini
    - Reasoning Effort: high
    """
    return LLM(
        model='openai/o3-mini',
        reasoning_effort="high",
    ) 