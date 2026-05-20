import requests
import time
from typing import List, Dict, Any, Optional, Union
import json


class ModelClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url

    def chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class DeepSeekClient(ModelClient):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        timeout: int = 120
    ):
        super().__init__(api_key, base_url)
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        retry_count: int = 0
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']

            elif response.status_code == 429:
                if retry_count < self.max_retries:
                    wait_time = 2 ** retry_count
                    time.sleep(wait_time)
                    return self.chat(messages, temperature, max_tokens, retry_count + 1)
                else:
                    raise Exception(f"Rate limit exceeded after {self.max_retries} retries")

            elif response.status_code == 500:
                if retry_count < self.max_retries:
                    wait_time = 2 ** retry_count
                    time.sleep(wait_time)
                    return self.chat(messages, temperature, max_tokens, retry_count + 1)
                else:
                    raise Exception(f"Server error after {self.max_retries} retries")

            else:
                error_detail = response.json() if response.content else {}
                raise Exception(
                    f"API request failed with status {response.status_code}: "
                    f"{error_detail.get('error', response.text)}"
                )

        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                return self.chat(messages, temperature, max_tokens, retry_count + 1)
            raise Exception("Request timeout after maximum retries")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")

    def chat_with_functions(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "functions": functions,
            "temperature": 0.7
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']
            return {
                "content": message.get('content'),
                "function_call": message.get('function_call')
            }

        else:
            raise Exception(f"API request failed: {response.text}")

    @staticmethod
    def format_tool_response(tool_name: str, result: Any) -> str:
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)
        return str(result)

    @staticmethod
    def create_function_call_response(
        function_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "name": function_name,
            "arguments": arguments
        }
