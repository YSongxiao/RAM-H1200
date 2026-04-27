import os
import json
from .openai_client import build_content, call_openai_messages
from utils import concise_system, concise_user, SHORT_REASON_RULE

class GPT_Decider:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        """
        Initialize the LLM_Decider object with the OpenAI API Key.

        Args:
            api_key (str): OpenAI 的 API Key
        """
        self.api_key = api_key
        self.model = model

    def decide(self, output_file, prompt, image_paths=None, field=None):
        """
        Decide the output of the LLM model based on the prompt.

        Args:
            output_file (str): output file path
            prompt (str): prompt for the LLM model

        Returns:
            dict: result of the LLM model
        """
        system_prompt = concise_system(
            "You are a helpful assistant. Please help me make a decision based on the following information."
        )
        user_prompt = concise_user(prompt, SHORT_REASON_RULE)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_content(user_prompt, image_paths)}
        ]

        result, elapsed_ms = call_openai_messages(
            messages=messages,
            model=self.model,
            api_key=self.api_key,
        )

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = {}
        existing_data[field] = result
        existing_data[f"{field}_time_ms"] = elapsed_ms
        existing_data[f"{field}_trace"] = {
            "model": self.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_paths": image_paths or [],
            "raw_response": result,
            "time_ms": elapsed_ms,
        }
        
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4, ensure_ascii=False)

        return existing_data
