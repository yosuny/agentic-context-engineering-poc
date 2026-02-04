import requests
import json
import time
from typing import List, Dict, Any, Optional

class NCloudResponse:
    """Mock OpenAI response structure for NCloud"""
    class Choice:
        class Message:
            def __init__(self, content):
                self.content = content
        def __init__(self, content):
            self.message = self.Message(content)
    
    class Usage:
        def __init__(self, prompt_tokens=0, completion_tokens=0):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = prompt_tokens + completion_tokens

    def __init__(self, content, prompt_tokens=0, completion_tokens=0):
        self.choices = [self.Choice(content)]
        self.usage = self.Usage(prompt_tokens, completion_tokens)

class NCloudChat:
    """Mock OpenAI chat completions structure"""
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url
        self.completions = self

    def create(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.5, 
               max_tokens: int = 4096, response_format: Dict = None, **kwargs) -> NCloudResponse:
        
        # Use the official v3 stream endpoint
        # Example: https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007
        base_url = self.api_url
        if "/v3/chat-completions/" not in base_url and not base_url.endswith("/v1/openai"):
            # Try to construct correct v3 URL if it looks like a base URL or older version
            if "apigw.ntruss.com" in base_url:
                # Gateway URL often has different routing, but we prioritize the stream.ntruss.com from docs
                pass
            elif not base_url.endswith("/v3/chat-completions/" + model):
                base_url = f"https://clovastudio.stream.ntruss.com/v3/chat-completions/{model}"

        auth_header = self.api_key if self.api_key.startswith("Bearer ") else f"Bearer {self.api_key}"
        
        headers = {
            "Authorization": auth_header,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": "ace-framework-call",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }

        # [V3 SPEC] topP default 0.8, topK default 0, repetitionPenalty range (0.0 < x <= 2.0, default 1.1)
        max_tok = kwargs.get("max_completion_tokens", max_tokens)
        top_p = kwargs.get("topP", 0.8)
        top_k = kwargs.get("topK", 0)
        rep_penalty = kwargs.get("repetitionPenalty", 1.1) 
        
        data = {
            "messages": messages,
            "topP": top_p,
            "topK": top_k,
            "temperature": temperature,
            "repetitionPenalty": rep_penalty,
            "stopBefore": kwargs.get("stopBefore", []),
            "includeAiFilters": kwargs.get("includeAiFilters", False),
            "seed": kwargs.get("seed", 0)
        }
        
        # Handle Structured Outputs (JSON)
        # v3 SO requires thinking.effort: "none" AND a schema definition
        is_json_mode = response_format and response_format.get("type") in ["json_object", "json"]
        effort = kwargs.get("thinking_effort", "none")
        
        if is_json_mode:
            # [V3 SPEC] responseFormat for SO must be {"type": "json", "schema": {...}}
            # If schema is not provided by ACE core, we provide a default one for Curator operations
            schema = response_format.get("schema")
            if not schema:
                schema = {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "operations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["ADD", "UPDATE", "MERGE", "DELETE", "CREATE_META"]},
                                    "section": {"type": "string"},
                                    "content": {"type": "string"}
                                },
                                "required": ["type", "section", "content"]
                            }
                        }
                    },
                    "required": ["reasoning", "operations"]
                }
            
            data["responseFormat"] = {
                "type": "json",
                "schema": schema
            }
            # [V3 SPEC] Explicitly set thinking effort to none for SO
            effort = "none"
        
        if effort != "none":
            data["thinking"] = {"effort": effort}
            data["maxCompletionTokens"] = max_tok
        else:
            # For non-thinking or SO, explicitly send none and ensure budget
            data["thinking"] = {"effort": "none"}
            data["maxCompletionTokens"] = max_tok if max_tok > 512 else 4096

        # [V3 SPEC] Structured Output often works better in non-streaming mode
        use_stream = not is_json_mode
        if not use_stream:
            headers["Accept"] = "application/json"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(base_url, headers=headers, json=data, stream=use_stream)
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                
                if not response.ok:
                    print(f"[NCLOUD ERROR] {response.status_code} for {base_url}")
                    print(f"Payload: {json.dumps(data)}")
                    print(f"Response: {response.text}")
                
                response.raise_for_status()
                
                full_content = ""
                usage = {"prompt_tokens": 0, "completion_tokens": 0}

                if use_stream:
                    current_event = ""
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith('event:'):
                                current_event = decoded_line.split(':', 1)[1].strip()
                            elif decoded_line.startswith('data:'):
                                data_str = decoded_line.split(':', 1)[1]
                                try:
                                    data_json = json.loads(data_str)
                                    if current_event == "result":
                                        if "message" in data_json:
                                            full_content = data_json["message"].get("content", "")
                                        if "usage" in data_json:
                                            usage["prompt_tokens"] = data_json["usage"].get("inputTokens", 0)
                                            usage["completion_tokens"] = data_json["usage"].get("outputTokens", 0)
                                except json.JSONDecodeError:
                                    pass
                else:
                    # Non-streaming response parsing - more robust
                    try:
                        resp_json = response.json()
                        # V3 Non-stream structure: {"status": ..., "result": {"message": ..., "usage": ...}}
                        target = resp_json.get("result", resp_json)
                        
                        if "message" in target:
                            full_content = target["message"].get("content", "")
                        
                        if "usage" in target:
                            u = target["usage"]
                            usage["prompt_tokens"] = u.get("promptTokens", u.get("inputTokens", 0))
                            usage["completion_tokens"] = u.get("completionTokens", u.get("outputTokens", 0))
                    except Exception as e:
                        print(f"[NCLOUD PARSE ERROR] Failed to parse JSON: {e}")
                        print(f"Raw Response: {response.text[:1000]}")
                        raise e

                return NCloudResponse(
                    full_content.strip(), 
                    prompt_tokens=usage["prompt_tokens"], 
                    completion_tokens=usage["completion_tokens"]
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise e
        
        return NCloudResponse("", 0, 0)

class NCloudClient:
    """Wrapper that mimics OpenAI client structure"""
    def __init__(self, api_key: str, api_url: str):
        self.chat = NCloudChat(api_key, api_url)
