# ACE Performance Pilot Comparison Report

## 1. Test Question
> 15억 아파트를 5년 보유 및 거주한 1세대 1주택자의 양도소득세 계산 과정과 예상 세액을 알려주세요. (취득가액은 10억으로 가정)

## 2. Comparison Results

### [A] Baseline (No Playbook)
```text
```json
{
  "reasoning": "The individual qualifies as a 1세대 1주택자 with 5-year ownership and residency. Capital gains tax exemption applies to the first 12억 of gains. Total gain = 15억 - 10억 = 5억, which is below the 12억 exemption. Taxable gain = 0. No additional deductions mentioned, assuming standard rates.",
  "bullet_ids": ["tax-001", "exemptions-002"],
  "final_answer": "예상 세액: 0원"
}
```
```

### [B] ACE Augmented (Phase 3 Playbook)
```text
```json
{
  "reasoning": "매각가격 15억원 중 12억원 초과분 3억원에 대해 과세됨. 양도차익 5억원((15억-10억)의 (3억/15억) 비율 적용 → 1억원. 장기보유공제 20%(5년×4%) 적용으로 2천만원 공제 후 과세표준 8천만원",
  "bullet_ids": ["calc-00001", "calc-00004", "calc-00006"],
  "final_answer": "예상 과세표준 8,000만원"
}
```
```

## 3. Analysis
- **Correctness**: Did the model use the 12억 threshold correctly?
- **Precision**: Did it calculate the tax ratio for high-value property correctly?
- **Specifics**: Was the Long-term Special Deduction (장특공제) accuracy improved?
