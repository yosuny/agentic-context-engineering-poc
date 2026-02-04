
import os
import sys
import json
from typing import List, Dict, Any

# Add ace directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ace')))

from ace import ACE

class KoreanDataProcessor:
    """Simple data processor for Korean real estate tax scenario."""
    def __init__(self, task_name: str = "korean_tax"):
        self.task_name = task_name
        
    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        processed_data = []
        for item in raw_data:
            processed_data.append({
                "question": item["question"],
                "target": item["target"],
                "context": "", # No extra context for this simple test
                "others": {"task": self.task_name}
            })
        return processed_data
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        # Simple heuristic: if key terms from ground truth are in predicted
        # In a real scenario, this would be more complex.
        # For this test, we care more about the training flow.
        ground_truth_keywords = ["비과세", "과세", "80%", "장기보유특별공제", "특례"]
        for kw in ground_truth_keywords:
            if kw in ground_truth and kw in predicted:
                return True
        return False
    
    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        if not out:
            return 0.0
        correct = 0
        for p, t in zip(out, target):
            if self.answer_is_correct(p, t):
                correct += 1
        return correct / len(out)

def run_korean_test():
    # 1. Load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'korean_samples.jsonl'))
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # 2. Initialize ACE
    # Using NCloud HCX-007
    ace_system = ACE(
        api_provider="ncloud",
        generator_model="HCX-007",
        reflector_model="HCX-007",
        curator_model="HCX-007",
        max_tokens=2048
    )
    
    # 3. Configure run
    config = {
        'num_epochs': 1,
        'max_num_rounds': 2,
        'curator_frequency': 1,
        'online_eval_frequency': 1, # Update every sample for faster feedback
        'task_name': 'korean_real_estate_tax',
        'json_mode': True, # Use SO for Curator
        'no_ground_truth': False,
        'save_dir': './results',
        'test_workers': 1
    }
    
    processor = KoreanDataProcessor()
    processed_data = processor.process_task_data(data)
    
    # 4. Run Online Adaptation
    print("\n🚀 Starting ACE Korean Scenario Test...")
    results = ace_system.run(
        mode='online',
        test_samples=processed_data,
        data_processor=processor,
        config=config
    )
    
    print("\n✅ ACE Korean Test Completed!")
    print(f"Final Playbook:\n{ace_system.playbook}")

if __name__ == "__main__":
    run_korean_test()
