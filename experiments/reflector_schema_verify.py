
import os
import sys
import json
from unittest.mock import MagicMock

# Add ace directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ace')))

from ace.core.reflector import Reflector
from ncloud_llm import NCloudClient

def test_reflector_schema_enforcement():
    print("--- Testing Reflector Schema Enforcement ---")
    
    # 1. Setup proper NCloud Client
    api_key = os.getenv('NCLOUD_API_KEY')
    api_url = os.getenv('NCLOUD_API_URL')
    
    if not api_key or not api_url:
        print("Skipping test: NCLOUD_API_KEY or NCLOUD_API_URL not found")
        return

    client = NCloudClient(api_key=api_key, api_url=api_url)
    
    # 2. Initialize Reflector
    reflector = Reflector(client, "ncloud", "HCX-007")
    
    # 3. Prepare dummy data
    question = "지게차 운전 중 사고가 났습니다. 원인은 무엇인가요?"
    reasoning_trace = "{\"reasoning\": \"운전 미숙입니다.\"}"
    predicted_answer = "운전 미숙"
    ground_truth = "지게차 유도자 미배치 및 시야 미확보"
    bullets_used = "[safe-001] 지게차 작업 시 유도자를 배치해야 한다."
    
    print(f"Sending reflection request with explicit schema enforcement...")
    
    # 4. Call reflect with use_json_mode=True
    response, bullet_tags, call_info = reflector.reflect(
        question=question,
        reasoning_trace=reasoning_trace,
        predicted_answer=predicted_answer,
        ground_truth=ground_truth,
        environment_feedback="Predicted answer does not match ground truth",
        bullets_used=bullets_used,
        use_ground_truth=True,
        use_json_mode=True,
        call_id="test_reflector_schema_sample"
    )
    
    print(f"\nResponse received:\n{response}")
    
    # 5. Verify Structure
    try:
        data = json.loads(response)
        has_tags_list = "bullet_tags" in data
        has_reasoning = "reasoning" in data
        has_correction = "correct_approach" in data
        
        # Check if we mistakenly got 'operations' (Curator schema)
        has_curator_artifact = "operations" in data
        
        if has_curator_artifact:
             print("\n❌ FAILURE: Reflector returned 'operations' field (Curator Schema leaked!).")
             return
             
        if has_tags_list:
            print("\n✅ SUCCESS: 'bullet_tags' field found in response!")
            print(f"Extracted Tags: {bullet_tags}")
        else:
            print("\n❌ FAILURE: 'bullet_tags' field MISSING in response.")
            print(f"Keys found: {list(data.keys())}")
            
    except json.JSONDecodeError:
        print("\n❌ FAILURE: Response is not valid JSON.")

if __name__ == "__main__":
    test_reflector_schema_enforcement()
