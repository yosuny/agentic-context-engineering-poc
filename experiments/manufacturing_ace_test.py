
import os
import sys
import json
from typing import List, Dict, Any

# Add ace directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ace')))

from ace import ACE

class ManufacturingDataProcessor:
    """Data processor for Manufacturing Safety scenario."""
    def __init__(self, task_name: str = "manufacturing_safety"):
        self.task_name = task_name
        
    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Convert JSONL items to ACE task format."""
        processed_data = []
        for item in raw_data:
            processed_data.append({
                "question": item["question"],
                "target": item["target"],
                "context": item.get("context", ""),
                "others": {
                    "task": self.task_name,
                    "scenario_id": item.get("scenario_id"),
                    "difficulty": item.get("difficulty")
                }
            })
        return processed_data
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer covers key safety concepts in ground truth.
        """
        # 1. Extract Keywords from Ground Truth
        # We assume ground truth is "근본 원인: A, B\n대책: C, D"
        # We'll split by likely separators and filters
        gt_keywords = []
        
        # Simple extraction: split by space, comma, newline, remove common words
        ignore_words = ["근본", "원인", "대책", "및", "함", "것", "등", "수", "있음", "해야", "위해", ":", "-"]
        
        raw_tokens = ground_truth.replace(",", " ").replace("\n", " ").split()
        for token in raw_tokens:
            token = token.strip()
            if len(token) > 1 and token not in ignore_words:
                gt_keywords.append(token)
                
        # 2. Check coverage
        if not gt_keywords:
            return False
            
        hit_count = 0
        predicted_normalized = predicted.replace(" ", "")
        
        for kw in gt_keywords:
            # Check if keyword (or simple subword) exists in prediction
            # We remove spaces in prediction to handle "안전 수칙" vs "안전수칙"
            if kw in predicted or kw in predicted_normalized:
                hit_count += 1
                
        # heuristic: if > 50% of keywords are present (handling partial matches)
        # For strict safety evaluation, we might want higher recall, 
        # but for this pilot, 40-50% keyword overlap suggests the model got the core concept.
        overlap_ratio = hit_count / len(gt_keywords)
        return overlap_ratio >= 0.4
    
    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        if not out:
            return 0.0
        correct = 0
        for p, t in zip(out, target):
            if self.answer_is_correct(p, t):
                correct += 1
        return correct / len(out)

def run_manufacturing_test():
    # 1. Load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'manufacturing_eval_v1.jsonl'))
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    all_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(json.loads(line))
            
    # Split train/val by 'phase' field
    train_data = [d for d in all_data if d.get("phase") == "train"]
    val_data = [d for d in all_data if d.get("phase") == "validation"]
    
    print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples.")
    
    # 2. Initialize ACE
    # Using 'ncloud' provider with 'HCX-007' as per user context
    ace_system = ACE(
        api_provider="ncloud",
        generator_model="HCX-007",
        reflector_model="HCX-007",
        curator_model="HCX-007",
        max_tokens=2048,
        initial_playbook=None # Cold Start
    )
    
    # 3. Configure run
    config = {
        'num_epochs': 1,
        'max_num_rounds': 2, # Reflection rounds
        'curator_frequency': 5, # Curate every 5 steps
        'eval_steps': 10, # Evaluate every 10 steps
        'task_name': 'manufacturing_safety',
        'json_mode': True,
        'no_ground_truth': False,
        'save_dir': './results',
        'test_workers': 1
    }
    
    processor = ManufacturingDataProcessor()
    processed_train = processor.process_task_data(train_data)
    processed_val = processor.process_task_data(val_data)
    
    # 4. Run Offline Training Mode
    # We use offline mode to clearly see the epoch-based evolution
    print("\n🚀 Starting ACE Manufacturing Safety Test (Offline Mode)...")
    results = ace_system.run(
        mode='offline',
        train_samples=processed_train,
        val_samples=processed_val,
        data_processor=processor,
        config=config
    )
    
    print("\n✅ ACE Manufacturing Test Completed!")
    print(f"Final Playbook:\n{ace_system.playbook}")

if __name__ == "__main__":
    run_manufacturing_test()
