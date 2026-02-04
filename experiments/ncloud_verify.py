import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "ace"))

from utils import initialize_clients, load_dotenv
from ace.core import Generator, Reflector, Curator

def test_ncloud_connection():
    load_dotenv()
    
    provider = os.getenv("API_PROVIDER", "ncloud")
    model = os.getenv("MODEL_NAME", "HCX-007")
    
    print(f"--- Starting NCloud Connection Test ({provider}/{model}) ---")
    
    try:
        # 1. Initialize Clients
        g_client, r_client, c_client = initialize_clients(provider)
        
        # 2. Setup Agents
        generator = Generator(g_client, provider, model)
        reflector = Reflector(r_client, provider, model)
        curator = Curator(c_client, provider, model)
        
        # 3. Test Generator (Low Thinking)
        print("\n[Testing Generator - Expected Low Thinking]")
        gen_resp, bullet_ids, call_info = generator.generate(
            question="What is the capital of France?",
            playbook="## STRATEGIES\n[gen-00001] helpful=0 harmful=0 :: Use clear language."
        )
        print(f"Generator Response: {gen_resp[:100]}...")
        print(f"Thinking Effort used: {call_info.get('thinking_effort', 'N/A')}")
        
        # 4. Test Reflector (Medium Thinking)
        print("\n[Testing Reflector - Expected Medium Thinking]")
        ref_text, tags, ref_info = reflector.reflect(
            question="What is the capital of France?",
            reasoning_trace=gen_resp,
            predicted_answer="Paris",
            ground_truth="Paris",
            environment_feedback="Correct answer.",
            bullets_used="[gen-00001] helpful=0 harmful=0 :: Use clear language.",
            use_ground_truth=True
        )
        print(f"Reflector Feedback: {ref_text[:100]}...")
        print(f"Thinking Effort used: {ref_info.get('thinking_effort', 'N/A')}")
        
        # 5. Test Curator (High Thinking / JSON Mode)
        # Note: If JSON mode is True, thinking_effort should be 'none' internally for V3 compatibility
        print("\n[Testing Curator - Expected JSON Mode / Thinking None]")
        # Ensure log_dir exists for curator
        log_dir = root_dir / "logs" / "test"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        playbook, next_id, ops, cur_info = curator.curate(
            current_playbook="## STRATEGIES\n[gen-00001] helpful=0 harmful=0 :: Use clear language.",
            recent_reflection=ref_text,
            question_context="General Knowledge",
            current_step=1,
            total_samples=1,
            token_budget=1000,
            playbook_stats={"total_bullets": 1},
            use_json_mode=True,
            log_dir=str(log_dir)
        )
        print(f"Curator Operations Found: {len(ops)}")
        print(f"Thinking Effort used (should be none for JSON): {cur_info.get('thinking_effort', 'N/A')}")
        
        print("\n✅ NCloud Connection Test Passed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Connection Test Failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ncloud_connection()
