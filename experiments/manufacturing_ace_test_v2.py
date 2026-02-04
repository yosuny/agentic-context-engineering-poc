
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
                "target": item["answer"],  # V2 uses 'answer'
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
        
        # Pass if > 30% keywords match (lenient)
        if len(keywords) == 0: return False
        return (hit / len(keywords)) > 0.3

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        return 0.0 

def run_manufacturing_test_v2():
    # 1. Configuration
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ace_run_{current_time}_manufacturing_v2_offline"
    
    # 5 Epochs for deep learning of case differentiation
    config = {
        'num_epochs': 5,
        'max_num_rounds': 2,      # 2 Reflection rounds
        'curator_frequency': 5,   # Curate every 5 steps
        'eval_steps': 10,
        'task_name': 'manufacturing_safety_v2',
        'json_mode': True,
        'save_dir': f'./results/{run_name}',
        'test_workers': 1
    }

    print(f"--- Starting ACE Manufacturing Safety Test V2 ---")
    print(f"Run Name: {run_name}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # 2. Initialize ACE
    ace = ACE(
        api_provider="ncloud",
        generator_model="HCX-007",
        reflector_model="HCX-007",
        curator_model="HCX-007",
        max_tokens=2048
    )

    # 3. Load V2 Dataset
    dataset_path = "tests/manufacturing_eval_v2.jsonl"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    training_data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            training_data.append(json.loads(line))
            
    # Rapid Verification: Select 5 per category
    samples_per_category = 5
    grouped_data = {"machinery": [], "chemical": [], "forklift": []}
    
    for item in training_data:
        cat = item['category']
        if len(grouped_data[cat]) < samples_per_category:
            grouped_data[cat].append(item)
            
    final_training_data = []
    for cat in grouped_data:
        final_training_data.extend(grouped_data[cat])
        
    print(f"Selected {len(final_training_data)} samples for Rapid Verification (5 per category)")
            
    # Shuffle data
    random.seed(42)
    random.shuffle(final_training_data)
    
    print(f"Loaded {len(final_training_data)} samples from {dataset_path}")
    
    # 4. Run Training
    processor = ManufacturingDataProcessor()
    processed_samples = processor.process_task_data(final_training_data)

    print("Starting training loop...")
    ace.run(
        mode='offline',
        train_samples=processed_samples,
        val_samples=processed_samples, # Use same for validation
        data_processor=processor,
        config=config
    )

    # 5. Categorical Analysis
    print("\n--- Performing Categorical Analysis ---")
    result_path = os.path.join("results", run_name, "val_results.json")
    
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            val_results = json.load(f)
            
        # Get final epoch results
        final_epoch = val_results[-1]
        errors = final_epoch.get("error_log", {}).get("errors", [])
        
        category_stats = {
            "machinery": {"total": 0, "correct": 0},
            "chemical": {"total": 0, "correct": 0},
            "forklift": {"total": 0, "correct": 0}
        }
        
        error_indices = {e['index'] for e in errors}
        
        for i, sample in enumerate(final_training_data):
            cat = sample['category']
            category_stats[cat]['total'] += 1
            if i not in error_indices:
                category_stats[cat]['correct'] += 1
                
        print("\n[Final Categorical Accuracy]")
        for cat, stats in category_stats.items():
            acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"  - {cat.upper()}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
            
        # Check for LOTO Overfitting (High Machinery score, Low Chemical/Forklift score)
        machinery_acc = (category_stats['machinery']['correct'] / category_stats['machinery']['total']) if category_stats['machinery']['total'] > 0 else 0
        others_acc = (category_stats['chemical']['correct'] + category_stats['forklift']['correct']) / (category_stats['chemical']['total'] + category_stats['forklift']['total']) if (category_stats['chemical']['total'] + category_stats['forklift']['total']) > 0 else 0
        
        if machinery_acc > 0.5 and others_acc < 0.2:
            print("\n⚠️  WARNING: LOTO Overfitting detected! (Machinery high, Others low)")
        elif others_acc > 0.5:
             print("\n✅ SUCCESS: Case Differentiation Logic is working!")
             
    else:
        print("Warning: val_results.json not found, skipping analysis.")

if __name__ == "__main__":
    run_manufacturing_test_v2()
