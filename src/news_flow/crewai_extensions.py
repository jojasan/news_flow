import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type, Union

from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from crewai.utilities.serialization import to_serializable
from pydantic import BaseModel, ValidationError

# Necessary imports from crewai.utilities.converter
from crewai.utilities.converter import (
    Converter,
    ConverterError,
    convert_with_instructions,
    validate_model,
)
from crewai.utilities.printer import Printer


class SQLiteFlowPersistenceJSON(SQLiteFlowPersistence):
    """SQLite persistence with robust JSON serialization for state data."""

    def save_state(
        self,
        flow_uuid: str,
        method_name: str,
        state_data: Union[Dict[str, Any], BaseModel],
    ) -> None:
        """Save the current flow state to SQLite using robust serialization."""
        # Use to_serializable for robust conversion to JSON-compatible types
        serializable_state = to_serializable(state_data)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
            INSERT INTO flow_states (
                flow_uuid,
                method_name,
                timestamp,
                state_json
            ) VALUES (?, ?, ?, ?)
            """,
                (
                    flow_uuid,
                    method_name,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(serializable_state),  # Dump the serializable dict
                ),
            )

# --- New Monkey Patch Logic ---

def clean_llm_json_output(text: str) -> str:
    """
    Cleans potential markdown fences (e.g., ```json ... ```)
    and extracts the core JSON string.
    """
    # Regex to capture JSON content within optional markdown fences
    # Handles optional leading/trailing whitespace and optional 'json' language tag
    match = re.search(
        r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", text, re.DOTALL | re.IGNORECASE
    )
    if match:
        # Return the captured group, stripped of leading/trailing whitespace
        return match.group(1).strip()

    # If no markdown fence found, return the original text stripped
    return text.strip()


def patched_handle_partial_json(
    result: str,
    model: Type[BaseModel],
    is_json_output: bool,
    agent: Any,
    converter_cls: Optional[Type[Converter]] = None,
) -> Union[dict, BaseModel, str]:
    """
    Patched version of handle_partial_json.
    Cleans markdown fences before attempting validation or regex extraction.
    """
    cleaned_result = clean_llm_json_output(result)

    # Attempt 1: Try direct validation on the cleaned result
    try:
        # Use json.loads with strict=False for flexibility
        json_data = json.loads(cleaned_result, strict=False)
        # Re-dump to ensure consistent string format for Pydantic validation
        escaped_json_string = json.dumps(json_data)
        return validate_model(escaped_json_string, model, is_json_output)
    except json.JSONDecodeError as e:
        Printer().print(
            content=f"Direct JSON decoding failed on cleaned result: {e}. Falling back to regex.",
            color="yellow",
        )
        # Proceed to regex extraction
        pass
    except ValidationError as e:
        Printer().print(
            content=f"Direct Pydantic validation failed on cleaned result: {e}. Falling back to regex.",
            color="yellow",
        )
        # Proceed to regex extraction
        pass
    except Exception as e:
         Printer().print(
            content=f"Unexpected error during direct validation of cleaned JSON: {type(e).__name__}: {e}. Falling back to regex.",
            color="red",
        )

    # Attempt 2: Regex extraction on the cleaned result (like original fallback)
    # Looks for the outermost curly braces or square brackets
    match = re.search(r"({.*})|(\[.*\])", cleaned_result, re.DOTALL)
    if match:
        extracted_json = match.group(0) # The entire matched JSON string
        try:
            # Validate the JSON extracted via regex
            return validate_model(extracted_json, model, is_json_output)
        except json.JSONDecodeError as e:
             Printer().print(
                content=f"JSON decoding failed even after regex extraction: {e}. Falling back to instructional conversion.",
                color="yellow",
            )
        except ValidationError as e:
             Printer().print(
                content=f"Pydantic validation failed even after regex extraction: {e}. Falling back to instructional conversion.",
                color="yellow",
            )
        except Exception as e:
            Printer().print(
                content=f"Unexpected error during regex-extracted JSON validation: {type(e).__name__}: {e}. Falling back to instructional conversion.",
                color="red",
            )
    else:
         Printer().print(
            content=f"Could not find JSON object/array via regex in cleaned output. Falling back to instructional conversion.",
            color="yellow",
        )

    # Fallback: Use the instructional conversion (passing the ORIGINAL result)
    # This allows the LLM to retry fixing the format if cleaning/regex failed
    return convert_with_instructions(
        result, model, is_json_output, agent, converter_cls
    )

# --- LiteLLM Monkey Patch for Empty Response Fallback (Targeting Fallback Logic) ---

import litellm
import litellm.litellm_core_utils.fallback_utils # Ensure the module is loaded
import functools
import uuid
from copy import deepcopy
from litellm.types.utils import ModelResponse
from litellm.utils import verbose_logger
# Import exceptions LiteLLM uses for retry/fallback logic
from litellm.exceptions import APIError, Timeout, RateLimitError, ServiceUnavailableError, APIConnectionError

print("Attempting to monkey-patch litellm's fallback logic...")

_patch_successful = False
original_async_completion_with_fallbacks = None

# --- Safety Check: Ensure the target function exists ---
try:
    original_async_completion_with_fallbacks = litellm.litellm_core_utils.fallback_utils.async_completion_with_fallbacks
except AttributeError:
    print("ERROR: litellm.litellm_core_utils.fallback_utils.async_completion_with_fallbacks not found. Patch cannot be applied.")

if original_async_completion_with_fallbacks:

    # Define exceptions that LiteLLM's fallback logic might catch and proceed on
    # (Consult LiteLLM retry logic for the exact set if needed)
    FALLBACK_TRIGGER_EXCEPTIONS = (
        APIError,
        Timeout,
        RateLimitError,
        ServiceUnavailableError,
        APIConnectionError,
        # Add others if necessary
    )

    # Define the new wrapper function
    @functools.wraps(original_async_completion_with_fallbacks)
    async def patched_async_completion_with_fallbacks(**kwargs):
        nested_kwargs = kwargs.pop("kwargs", {})
        original_model = kwargs["model"]
        fallbacks = [original_model] + nested_kwargs.pop("fallbacks", [])
        kwargs.pop("acompletion", None)  # Remove to prevent keyword conflicts
        litellm_call_id = str(uuid.uuid4())
        base_kwargs = {**kwargs, **nested_kwargs, "litellm_call_id": litellm_call_id}
        base_kwargs.pop("model", None)  # Remove model as it will be set per fallback
        logger_fn = base_kwargs.get("logger_fn", litellm.print_verbose)

        # Try each fallback model
        for fallback in fallbacks:
            current_model = "unknown"
            try:
                completion_kwargs = deepcopy(base_kwargs)

                # Handle dictionary fallback configurations
                if isinstance(fallback, dict):
                    current_model = fallback.pop("model", original_model)
                    completion_kwargs.update(fallback)
                else:
                    current_model = fallback

                is_streaming = completion_kwargs.get('stream', False)
                logger_fn(f"Patched Fallback Logic: Attempting model '{current_model}' (Streaming: {is_streaming})")

                # *** CORE LOGIC: Call the actual completion function ***
                response = await litellm.acompletion(**completion_kwargs, model=current_model)
                # *******************************************************

                if response is not None:
                    # --- PATCH LOGIC: Check for empty successful non-streaming response ---
                    is_successful_empty_non_streaming = False
                    if not is_streaming and isinstance(response, ModelResponse) and response.choices and len(response.choices) > 0:
                        first_choice = response.choices[0]
                        message_content = getattr(getattr(first_choice, 'message', None), 'content', None)
                        finish_reason = getattr(first_choice, 'finish_reason', None)

                        if (message_content is None or message_content == "") and finish_reason == "stop":
                            is_successful_empty_non_streaming = True
                    # --- END PATCH LOGIC ---

                    if is_successful_empty_non_streaming:
                        # Instead of returning, log and continue to the next fallback
                        logger_fn(f"Patched Fallback Logic: Model '{current_model}' returned a successful but empty non-streaming response. Proceeding to next fallback.")
                        continue # Try the next model in the fallbacks list
                    else:
                        # Return the valid (non-empty or streaming) response
                        logger_fn(f"Patched Fallback Logic: Model '{current_model}' returned a valid response. Returning.")
                        return response
                else:
                    # Original litellm.acompletion returned None, maybe log this?
                    logger_fn(f"Patched Fallback Logic: litellm.acompletion for model '{current_model}' returned None. Proceeding to next fallback.")
                    continue # Should ideally not happen unless acompletion itself fails weirdly

            except FALLBACK_TRIGGER_EXCEPTIONS as e:
                verbose_logger.warning(
                    f"Patched Fallback Logic: Fallback attempt failed for model {current_model} with retryable error: {type(e).__name__}. Proceeding."
                )
                # Allow LiteLLM's logic to proceed to the next fallback on these errors
                continue
            except Exception as e:
                 verbose_logger.exception(
                    f"Patched Fallback Logic: Fallback attempt failed for model {current_model} with non-retryable error: {type(e).__name__}: {str(e)}. Proceeding."
                )
                 # For non-specified exceptions, we still continue to allow other fallbacks
                 # This mirrors the original logic which catches generic Exception
                 continue

        # If loop finishes without returning, raise the final error
        raise Exception(
            "Patched Fallback Logic: All fallback attempts failed. Enable verbose logging with `litellm.set_verbose=True` for details."
        )

    # Apply the patch
    litellm.litellm_core_utils.fallback_utils.async_completion_with_fallbacks = patched_async_completion_with_fallbacks
    # Since the sync version calls the async one, patching the async one should cover both
    print("SUCCESS: litellm's async_completion_with_fallbacks has been monkey-patched.")
    print("-> NOTE: Patch targets empty non-streaming responses within the fallback loop.")
    _patch_successful = True

def is_litellm_patched() -> bool:
    """Checks if the LiteLLM fallback patch was successfully applied."""
    return _patch_successful

# --- End of LiteLLM Patch ---
