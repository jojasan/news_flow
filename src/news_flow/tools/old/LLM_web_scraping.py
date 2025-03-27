from typing import Any
from crewai_tools import ScrapeWebsiteTool
from crewai import LLM
import tiktoken
import time
import math

class LLMScrapeWebsiteTool(ScrapeWebsiteTool):

    def count_tokens(self, text, encoding_name='cl100k_base'):
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)

    def split_into_batches(self, text):
        token_count = self.count_tokens(text)
        num_batches = math.ceil(token_count / 100_000)
        avg_chars_per_batch = math.ceil(len(text) / num_batches)

        batches = []
        for i in range(0, len(text), avg_chars_per_batch):
            batches.append(text[i:i + avg_chars_per_batch])

        return batches

    def call_llm_with_retry(self, llm, prompt):
        try:
            return llm.call(prompt)
        except Exception as e:
            print(f"Initial call failed with error: {e}. Retrying in 60 seconds...")
            time.sleep(60)
            try:
                return llm.call(prompt)
            except Exception as e:
                print(f"Retry failed again with error: {e}.")
                return "Error: The website couldn't be scrapped."

    def _run(self, **kwargs: Any) -> Any:
        # Call the original _run to get the text
        text = super()._run(**kwargs)
        MAX_TOKENS_LIMIT = 150_000
        
        llm = LLM(model="groq/llama-3.3-70b-versatile")

        token_count = self.count_tokens(text)
        if token_count > 400_000:
            return "Error: The website content is too long to be processed. Please try a different resource"

        if token_count <= MAX_TOKENS_LIMIT:
            prompt = f"""The following is the scrapped content of a web page. 
                        Remove all navigation elements, ads, and non-relevant content. Just keep the title and the 
                        content of the article itself, and do not change anything from that original content. Content: {text}"""
            response = self.call_llm_with_retry(llm, prompt)
        else:
            # Split text into batches
            print(f"------Text is too long ({token_count} tokens). Splitting into batches...------")
            batches = self.split_into_batches(text)

            responses = []
            for batch_text in batches:
                prompt = f"""The following is the scrapped content of a web page. 
                        Remove all navigation elements, ads, and non-relevant content. Just keep the title and the 
                        content of the article itself, and do not change anything from that original content. Content: {batch_text}"""
                response = self.call_llm_with_retry(llm, prompt)
                responses.append(response)
                time.sleep(30) # Sleep for 30 seconds between calls

            # Combine responses sequentially
            response = '\n'.join(responses)

        return response