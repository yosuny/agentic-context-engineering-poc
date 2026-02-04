# ACE 시스템 성능 평가 방법론 및 로직

본 문서는 ACE(Agent-Curator-Environment) 시스템에서 LLM의 기본 답변과 플레이북 기반 답변 간의 품질 차이를 측정하고 검증하는 방법론을 정의합니다.

## 1. 평가 개요: Delta-Performance Analysis
ACE의 핵심 가치는 **"경험(학습)을 통한 지식 누적"**에 있습니다. 이를 위해 학습 전(Baseline)과 학습 후(Augmented)의 성능 차이(Delta)를 분석합니다.

---

## 2. 주요 평가 지표 (KPIs)

### 📊 정량적 지표 (Quantitative)
1. **정확도 향상폭 (Accuracy Gain)**:
   - `Initial Test Accuracy` vs `Final Test Accuracy`
   - 동일한 테스트 세트에 대해 플레이북 주입 전후의 정답률 차이를 측정합니다.
2. **지식 추출 효율 (Extraction Rate)**:
   - `총 학습 샘플 수 / 플레이북에 추가된 Bullet 수`
   - 새로운 정보를 얼마나 정교하게 전략화(Strategizing)하는지 측정합니다.
3. **Fail-to-Pass 전환율**:
   - Baseline에서는 오답이었으나, 플레이북 적용 후 정답으로 전환된 샘플의 비율입니다.

### 📝 정성적 지표 (Qualitative)
1. **추론 과정의 정교함 (Reasoning Depth)**:
   - NCloud HCX-007의 `Thinking` 블록 내에서 플레이북의 전략(예: 고가주택 계산 공식)을 명시적으로 참조하는지 분석합니다.
2. **할루시네이션 억제**:
   - `common_mistakes_to_avoid` 섹션의 가이드를 통해 과거의 계산 실수를 반복하지 않는지 검증합니다.
3. **전문성 및 구체성**:
   - 답변에 법적 근거, 정확한 수치, 도메인 특화 용어가 포함되는 빈도를 측정합니다.

---

## 3. 세부 평가 로직 (Evaluation Logic)

### ① 결정론적 채점 (Deterministic Scoring)
- `DataProcessor.answer_is_correct()` 로직을 통해 정답 유무를 판별합니다.
- **수치형 데이터**: 허용 오차 범위 내 일치 여부 확인.
- **텍스트형 데이터**: 핵심 키워드(Keyword) 및 법률 용어 포함 여부 매칭.

### ② 모델 기반 평가 (LLM-as-a-Judge) - *선택 사항*
- GPT-4o 또는 별도의 HCX-007 인스턴스를 'Judge'로 활용합니다.
- **평가 항목**: 가독성, 논리성, 세무지식 정확성 (1~5점 척도).

---

## 4. 검증 파이프라인 (Workflow)

1. **Zero-shot Baseline**: 빈 플레이북으로 테스트 실행 및 결과 저장.
2. **ACE Training**: 학습 데이터를 통한 플레이북 진화.
3. **Playbook-Augmented Eval**: 진화된 플레이북을 프롬프트에 주입하여 재테스트.
4. **Final Comparison**: 두 결과 리포트 간의 지표 비교 및 개선 리포트 생성.

---

## 5. 결론
본 방법론을 통해 ACE 시스템이 단순히 '답변을 잘하는 모델'을 넘어, **"실수를 통해 배우고 지식을 고도화하는 에이전트 연합체"**임을 증명할 수 있습니다.
