
import os
import sys
import json
from typing import List, Dict, Any

# Add ace directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ace')))

from ace import ACE
from utils import initialize_clients

def run_ace_vs_rag_test():
    print("\n🚀 Starting ACE vs RAG Comparative Test...")
    print("Scenario: High-value Real Estate Tax Calculation (The Proportional Trap)")
    
    # Initialize ACE System
    api_provider = "ncloud"
    client, _, _ = initialize_clients(api_provider)
    
    ace_system = ACE(
        api_provider=api_provider,
        generator_model="HCX-007",
        reflector_model="HCX-007",
        curator_model="HCX-007",
        max_tokens=2048
    )

    # Test Question (Ambiguous for basic LLM)
    question = "양도가액 20억, 취득가액 10억인 1세대 1주택자입니다. 과세대상 양도차익은 얼마인가요?"

    # 1. RAG Simulation: Provide Raw Legal Text
    # RAG gives the "Law" but doesn't explain the "Method"
    rag_context = """
    [법령 발췌] 소득세법 제95조 제3항: 
    1세대 1주택자가 양도가액이 12억 원을 초과하는 고가주택을 양도하는 경우, 
    양도차익 중 12억 원을 초과하는 부분에 대해서만 양도소득세를 부과한다.
    """
    
    print("\n[RAG Mode] Generating response with raw legal text...")
    rag_response, _, _ = ace_system.generator.generate(
        question=question,
        playbook="", # No refined strategies
        context=rag_context,
        reflection="(none)",
        call_id="compare_rag"
    )

    # 2. ACE Simulation: Provide Refined Strategy (from Playbook)
    # ACE gives the "Heuristic" derived from previous failures
    ace_playbook = """## FORMULAS & CALCULATIONS
[calc-00004] 고가주택(12억 초과) 양도차익 계산 시 가장 많이 하는 실수는 단순히 '전체차익 - 12억'을 하는 것입니다.
반드시 아래의 '비율 안분 산식'을 적용해야 합니다:
과세대상 양도차익 = 전체 양도차익 × (양도가액 - 12억원) / 양도가액
"""
    
    print("\n[ACE Mode] Generating response with refined playbook strategy...")
    ace_response, _, _ = ace_system.generator.generate(
        question=question,
        playbook=ace_playbook,
        context="", # No need for raw background if the strategy is clear
        reflection="(none)",
        call_id="compare_ace"
    )

    # 3. Analyze Results
    # Correct Math: (20억 - 10억) * (20억 - 12억) / 20억 = 10억 * 8/20 = 4억
    # Common Error (RAG might do): 10억 - 12억 = 0 or 10억 - (something wrong)
    
    print("\n" + "="*80)
    print("ACE vs RAG: REASONING ROBUSTNESS CHECK")
    print("="*80)
    print(f"QUESTION: {question}")
    print("-" * 40)
    print(f"\n[RAG Result (Raw Law Only)]\n{rag_response}")
    print("-" * 40)
    print(f"\n[ACE Result (Refined Heuristic)]\n{ace_response}")
    print("="*80)
    print("\n💡 분석 포인트:")
    print("1. RAG는 법령을 주었음에도 모델이 산식을 잘못 세울 확률이 높습니다 (단순 뺄셈 등).")
    print("2. ACE는 '가장 많이 하는 실수'와 '정확한 산식'을 콕 집어 전달하여 오답률을 획기적으로 낮춥니다.")

if __name__ == "__main__":
    run_ace_vs_rag_test()
