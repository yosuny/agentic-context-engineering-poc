
import os
import sys
import json
from typing import List, Dict, Any

# Add ace directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ace')))

from ace import ACE
from utils import initialize_clients

def run_performance_pilot():
    print("\n🚀 Starting ACE Performance Pilot Test (Baseline vs ACE)...")
    
    # 1. Initialize ACE System
    api_provider = "ncloud"
    client, _, _ = initialize_clients(api_provider)
    
    ace_system = ACE(
        api_provider=api_provider,
        generator_model="HCX-007",
        reflector_model="HCX-007",
        curator_model="HCX-007",
        max_tokens=2048
    )

    # 2. Define the Test Question
    question = "15억 아파트를 5년 보유 및 거주한 1세대 1주택자의 양도소득세 계산 과정과 예상 세액을 알려주세요. (취득가액은 10억으로 가정)"

    # 3. Get Baseline Response (Empty Playbook)
    print("\n[A] Generating Baseline Response (No Playbook)...")
    baseline_response, _, _ = ace_system.generator.generate(
        question=question,
        playbook="",  # Empty playbook for baseline
        context="",
        reflection="(none)",
        call_id="pilot_baseline"
    )

    # 4. Get ACE Augmented Response (Using Evolved Playbook from Phase 3)
    # We'll use the playbook evolved in Phase 3
    evolved_playbook = """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS
[calc-00001] 양도소득세 비과세 요건을 충족하는지 확인하는 공식 및 계산 단계. 1세대 1주택자의 경우 12억 원까지는 비과세이며, 이를 초과하는 부분에 대해서만 과세합니다.
[calc-00004] 고가주택의 양도차익 계산 시, (양도가액 - 12억) / 양도가액 비율을 전체 양도차익에 곱하여 과세대상 양도차익을 산출합니다.
[calc-00006] 장기보유특별공제는 1세대 1주택자의 경우 보유 기간별 연 4%, 거주 기간별 연 4%를 합산하여 최대 80%까지 공제 가능합니다. (10년 이상 보유/거주 시)

## COMMON MISTAKES TO AVOID
[err-00002] 12억 원 이하 주택이라고 해서 무조건 비과세가 아니며, 2년 이상 보유(조정지역은 거주 포함) 요건을 확인해야 합니다.

## OTHERS
[misc-00007] 상속주택 등 일시적 2주택 특례가 적용되는지 반드시 확인해야 합니다.
"""
    
    print("\n[B] Generating ACE Augmented Response (With Evolved Playbook)...")
    ace_response, _, _ = ace_system.generator.generate(
        question=question,
        playbook=evolved_playbook,
        context="",
        reflection="(none)",
        call_id="pilot_ace"
    )

    # 5. Display Comparison
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)
    
    print("\n[BASELINE RESPONSE]")
    print("-" * 20)
    print(baseline_response)
    
    print("\n" + "="*40)
    
    print("\n[ACE AUGMENTED RESPONSE]")
    print("-" * 20)
    print(ace_response)
    print("="*80)

    # 6. Save Comparison Report
    report_path = "tests/pilot_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# ACE Performance Pilot Comparison Report\n\n")
        f.write("## 1. Test Question\n")
        f.write(f"> {question}\n\n")
        f.write("## 2. Comparison Results\n\n")
        f.write("### [A] Baseline (No Playbook)\n")
        f.write("```text\n" + baseline_response + "\n```\n\n")
        f.write("### [B] ACE Augmented (Phase 3 Playbook)\n")
        f.write("```text\n" + ace_response + "\n```\n\n")
        f.write("## 3. Analysis\n")
        f.write("- **Correctness**: Did the model use the 12억 threshold correctly?\n")
        f.write("- **Precision**: Did it calculate the tax ratio for high-value property correctly?\n")
        f.write("- **Specifics**: Was the Long-term Special Deduction (장특공제) accuracy improved?\n")

    print(f"\n✅ Pilot Comparison Report saved to: {report_path}")

if __name__ == "__main__":
    run_performance_pilot()
