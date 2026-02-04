
import os
import sys
import json
import random
from datetime import datetime
from typing import List, Dict

# Add ace directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ace')))

from ace.ace import ACE

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
                "target": item["answer"],  # V2 uses 'answer', V1 used 'target'
                "context": "", # Context is embedded in question for V2
                "others": {
                    "task": self.task_name,
                    "category": item.get("category"),
                    "metadata": item.get("metadata")
                }
            })
        return processed_data
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer covers key safety concepts in ground truth."""
        # Simple containment check for sample test
        # Normalize: remove spaces
        p_norm = predicted.replace(" ", "")
        g_norm = ground_truth.replace(" ", "")
        
        # If ground truth keywords are in prediction
        # Extract simple keywords (space separated)
        keywords = ground_truth.split()
        hit = 0
        for k in keywords:
            if len(k) < 2: continue
            if k in predicted:
                hit += 1
        
        # Pass if > 30% keywords match (lenient for sample test)
        if len(keywords) == 0: return False
        return (hit / len(keywords)) > 0.3

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        # Simple exact match or keyword match for sample test
        # We can implement a smarter one or rely on ACE's internal logs
        # giving a dummy implementation as offline mode handles eval internally via Reflector
        return 0.0 

def run_sample_test_v2():
    print(f"--- Starting ACE Manufacturing Safety SAMPLE Test V2 ---")
    
    # 1. Initialize ACE
    ace_system = ACE(
        api_provider="ncloud",
        generator_model="HCX-007",
        reflector_model="HCX-007",
        curator_model="HCX-007",
        max_tokens=2048,
        initial_playbook=None
    )

    # 2. Load V2 Dataset & Select Samples
    dataset_path = "tests/manufacturing_eval_v2.jsonl"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    all_data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(json.loads(line))
    
    # Select 1 sample from each category
    categories = ["machinery", "chemical", "forklift"]
    sample_data = []
    
    print("\n[Selected Samples for Testing]")
    for cat in categories:
        for item in all_data:
            if item['category'] == cat:
                sample_data.append(item)
                print(f"- Category: {cat.upper()}")
                print(f"  Question: {item['question'][:50]}...")
                break
    
    # 3. Process Data
    processor = ManufacturingDataProcessor()
    processed_samples = processor.process_task_data(sample_data)
    
    # 4. Configure & Run
    run_name = f"sample_v2_{datetime.now().strftime('%H%M%S')}"
    config = {
        'num_epochs': 1,
        'max_num_rounds': 1,      # 1 Reflection round for speed
        'curator_frequency': 1,   # Curate every step for this small sample
        'eval_steps': 1,
        'task_name': 'manufacturing_sample',
        'json_mode': True,
        'save_dir': f'./results/{run_name}',
        'test_workers': 1
    }
    
    print("\nStarting sample training loop (1 Epoch)...")
    ace_system.run(
        mode='offline',
        train_samples=processed_samples,
        val_samples=processed_samples, # Use same for validation in sample test
        data_processor=processor,
        config=config
    )

    print(f"\nSample test completed. Results saved to ./results/{run_name}")

if __name__ == "__main__":
    run_sample_test_v2()
