# 프로젝트 컨텍스트 (Project Context)

## 1. 프로젝트 개요 (Project Overview)
이 프로젝트는 **ACE (Agentic Context Engineering)** 프레임워크의 논문과 코드베이스를 분석하고, 로컬 환경에서 실행 가능한 테스트를 구현하여 동작 원리를 파악하는 것을 목표로 합니다.
*   **논문**: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/html/2510.04618v2)
*   **Repo**: [ace-agent/ace](https://github.com/ace-agent/ace)

## 2. 주요 목표 (Main Goals)
1.  **논문 및 코드 분석**: ACE의 핵심 개념(Generator, Reflector, Curator)과 코드 구조를 파악합니다. (완료)
2.  **검증 환경 구축**: 외부 API 키 없이 로직을 검증할 수 있는 **Mock Test** 환경을 구축합니다. (진행 중)
3.  **동작 검증**: Mock 테스트를 통해 플레이북 생성, 업데이트 과정이 코드상에서 정상 작동하는지 확인합니다.

## 3. 상세 진행 이력 (Detailed Progress History)

### Phase 1: 로컬 환경 구축 및 Mock UI 검증
- **주요 내용**: ACE 프레임워크 구조 분석 및 Streamlit 기반의 시각화 도구(MockApp) 구축.
- **성과**: 에이전트 간의 상호작용 및 플레이북 업데이트 로직을 시각적으로 확인.

### Phase 2: NCloud HCX-007 공식 연동
- **주요 내용**: NCloud v3 API(Streaming, Structured Output) 통합 및 OpenAI 의존성 완전 분리.
- **성과**: 
  - 에이전트별 추론 깊이(`Thinking Effort`) 차등 제어 구현.
  - JSON 모드(Structured Output)와 Thinking 모드의 상충 문제 해결.

### Phase 3: 한국어 실전 테스트 (양도소득세 시나리오)
- **주요 내용**: 대한민국 양도소득세 비과세 및 고가주택 계산 로직 학습 테스트.
- **성과**: 
  - HCX-007 모델이 한국의 복잡한 세법 조항을 스스로 분석하여 플레이북 전문 지식으로 추출함.

### Phase 4: 성능 검증 파일럿 (Baseline vs ACE)
- **주요 내용**: 플레이북 유무에 따른 고난도 세무 질문 답변 품질 사이드-바이-사이드 비교.
- **성과**: 
  - **할루시네이션 완벽 보정**: 기본 모델이 세액을 0원으로 잘못 산출한 고가주택 시나리오에서, ACE는 12억 초과분 산식을 적용하여 정확한 과세표준 도출 성공.

### Phase 5: ACE vs RAG 비교 분석 (Knowledge vs Strategy)
- **주요 내용**: 단답형 정보 검색(RAG)과 전략적 추론(ACE)의 답변 품질 비교.
- **성과**: 
  - **함정 회피**: 법령 원문을 주고도 계산 실수를 하는 RAG와 달리, ACE는 '함정 패턴'을 전략화하여 정확한 계산 수행.
  - **전략의 가치 입증**: 정보(Information) 제공을 넘어 방법(Method)을 제시하는 ACE의 우월성 확인.

### Phase 6: 제조 안전 도메인 평가 (Manufacturing Safety)
- **주요 내용**: KOSHA 재해사례 기반의 산업 안전 수칙(LOTO, 비상정지 등) 학습 및 전략화.
- **성과**: 
  - **Cold Start 학습**: 초기 지식 없이 20건의 사고 사례만으로 8개의 핵심 안전 수칙을 스스로 정립.
  - **정성적 추론 성공**: 작업자의 변명(급박함)을 거부하고 정확한 안전 절차(퍼지 미이행)를 지적하는 추론 능력 확인.

### Phase 7: 제조 안전 심화 평가 (Case Differentiation)
- **주요 내용**: 3가지 이질적 카테고리(기계, 화학, 지게차)가 혼합된 상황에서 상황별 특화 규칙 생성 여부 검증.
- **성과**:
  - **규칙 분화 성공**: "지게차 사고 시 유도자 배치" 등 도메인 특화 규칙이 LOTO 규칙과 별도로 생성됨을 확인.
  - **성장통 관측**: 초기 LOTO 만능주의 규칙이 다른 카테고리와 충돌하며 조정되는 'Organic Learning' 과정 실증.

## 4. 기술적 세부 사항 (Technical Details)
### 🚀 NCloud CLOVA Studio v3 최적화 연동
### 🚀 NCloud CLOVA Studio v3 최적화 연동
단순 API 연결을 넘어 v3 공식 사양을 바탕으로 기능을 고도화했습니다. 상세 내용은 [NCloud HCX-007 Optimization Guide](NCLOUD_OPTIMIZATION_GUIDE.md)를 참고하십시오.

*   **V3 전용 파라미터 적용**: `topP`(0.8), `topK`(0) 등 최적화 파라미터 적용.
*   **Safety Guard**: Thinking 기능과 JSON 모드 상충 방지 로직 구현.
*   **Adaptive Thinking**: 에이전트 역할별 추론 깊이(Thinking Effort) 자동 제어.

### 주요 수정 파일
*   **[MODIFY] ncloud_llm.py**: V3 사양 및 상충 방지 로직 반영.
*   **[MODIFY] llm.py**: 역할 기반 추론 깊이 자동 할당 인터페이스 확장.
