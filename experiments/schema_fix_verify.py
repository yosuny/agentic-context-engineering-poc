
import os
import sys
import json
from unittest.mock import MagicMock

# Add ace directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ace')))

from ace.core.generator import Generator
from ncloud_llm import NCloudClient

def test_generator_schema_enforcement():
    print("--- Testing Generator Schema Enforcement ---")
    
    # 1. Setup proper NCloud Client (assuming keys are in env)
    api_key = os.getenv('NCLOUD_API_KEY')
    api_url = os.getenv('NCLOUD_API_URL')
    
    if not api_key or not api_url:
        print("Skipping test: NCLOUD_API_KEY or NCLOUD_API_URL not found")
        return

    client = NCloudClient(api_key=api_key, api_url=api_url)
    
    # 2. Initialize Generator
    gen = Generator(client, "ncloud", "HCX-007")
    
    # 3. Call generate with use_json_mode=True
    # The fix we implemented ensures GENERATOR_SCHEMA is passed when use_json_mode=True
    question = "1+1은 무엇인가요?"
    playbook = "## STRATEGIES & INSIGHTS\n[calc-00001] helpful=0 harmful=0 :: 간단한 산술 연산은 정확히 계산하십시오."
    
    print(f"Sending request with explicit schema enforcement...")
    response, bullet_ids, call_info = gen.generate(
        question=question,
        playbook=playbook,
        use_json_mode=True,
        call_id="test_schema_fix_sample"
    )
    
    print(f"\nResponse received:\n{response}")
    
    # 4. Verify Structure
    try:
        data = json.loads(response)
        has_final_answer = "final_answer" in data
        has_reasoning = "reasoning" in data
        
        if has_final_answer:
            print("\n✅ SUCCESS: 'final_answer' field found in response!")
            print(f"Final Answer: {data['final_answer']}")
        else:
            print("\n❌ FAILURE: 'final_answer' field MISSING in response.")
            print(f"Keys found: {list(data.keys())}")
            
    except json.JSONDecodeError:
        print("\n❌ FAILURE: Response is not valid JSON.")

if __name__ == "__main__":
    test_generator_schema_enforcement()
