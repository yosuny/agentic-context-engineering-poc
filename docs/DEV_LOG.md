# 프로젝트 컨텍스트 (Project Context)

## 1. 프로젝트 개요 (Project Overview)
이 프로젝트는 **ACE (Agentic Context Engineering)** 프레임워크의 논문과 코드베이스를 분석하고, 로컬 환경에서 실행 가능한 테스트를 구현하여 동작 원리를 파악하는 것을 목표로 합니다.
*   **논문**: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/html/2510.04618v2)
*   **Repo**: [ace-agent/ace](https://github.com/ace-agent/ace)

## 2. 주요 목표 (Main Goals)
1.  **논문 및 코드 분석**: ACE의 핵심 개념(Generator, Reflector, Curator)과 코드 구조를 파악합니다. (완료)
2.  **검증 환경 구축**: 외부 API 키 없이 로직을 검증할 수 있는 **Mock Test** 환경을 구축합니다. (진행 중)
3.  **동작 검증**: Mock 테스트를 통해 플레이북 생성, 업데이트 과정이 코드상에서 정상 작동하는지 확인합니다.

## 3. 현재 상태 (Current Status)
*   **단계**: 실제 API 연동 및 검증 (Real API Integration & Verification)
*   **최근 활동**:
    *   Streamlit 기반 Mock App 구현 및 테스트 완료.
    *   NCloud CLOVA Studio HCX-007 연동을 위한 `ace/ncloud_llm.py` 구현.
    *   `ace/utils.py`, `ace/llm.py`에 NCloud 지원 로직 추가.
*   **다음 작업**: `.env` 설정 확인 후 실제 데이터를 이용한 NCloud 연동 테스트 실행.

## 4. 기술 스택 검토 (Tech Stack Review)
### 핵심 의존성 (Core Dependencies)
*   **언어**: Python 3.9+
*   **LLM 클라이언트**: `openai` (OpenAI, SambaNova, Together API 호환)
*   **유틸리티**:
    *   `tiktoken`: 토큰 계산 (비용 및 컨텍스트 관리)
    *   `python-dotenv`: 환경 변수 관리
*   **데이터/분석** (선택적):
    *   `sentence-transformers`, `faiss-cpu`: BulletpointAnalyzer 사용 시 필요 (벡터 유사도 기반 중복 제거). Mock Test에서는 비활성화 가능.
    *   `scikit-learn`: 평가 메트릭 계산.

### 검토 의견 (Review Code)
*   **적합성**: 최신 LLM 에이전트 개발에 표준적인 스택을 사용하고 있습니다.
*   **테스트 고려사항**:
    *   `ace/utils.py`가 `openai` 패키지를 강하게 의존(import)하고 있으므로, 테스트 환경에도 해당 패키지 설치가 필요합니다.
    *   외부 API 호출 없이 테스트하려면 `initialize_clients` 함수나 `openai.OpenAI` 클래스를 Mocking 해야 합니다.
    *   `BulletpointAnalyzer`는 무거운 의존성(`faiss`, `transformers`)을 가지므로, 초기 테스트에서는 `use_bulletpoint_analyzer=False`로 설정하여 의존성을 최소화하는 것이 권장됩니다.

## 5. 진행 이력 (Progress History)
| 날짜 | 단계 | 내용 | 비고 |
| :--- | :--- | :--- | :--- |
| 2026-02-03 | 분석 | 논문(2510.04618v2) 주요 내용 파악 및 GitHub Repo 분석 완료 | [분석 리포트](file:///Users/user/Hands-on/Agentic_Context_Engineering/ACE_ANALYSIS_REPORT.md) |
| 2026-02-03 | 구현 | Streamlit Mock App (`tests/mock_app.py`) 구현 및 시각화 테스트 완료 | |
| 2026-02-03 | 연동 | NCloud CLOVA Studio HCX-007 연동 코드 구현 (`ace/ncloud_llm.py`) | |
| 2026-02-03 | 계획 | 실제 API 연동을 위한 `.env` 템플릿 생성 및 추론 전략 수립 | |

## 6. 기술적 세부 사항 (Technical Details)
### 🚀 NCloud CLOVA Studio v3 최적화 연동
단순 API 연결을 넘어 v3 공식 사양을 바탕으로 다음 기능을 고도화했습니다.

*   **V3 전용 파라미터 적용**: `topP`(0.8), `topK`(0), `repetitionPenalty`(5.0) 등 HCX-007에 최적화된 기본값 및 파라미터 제어 로직을 `ncloud_llm.py`에 이식했습니다.
*   **에이전트별 추론 깊이(Thinking) 자동화**: `llm.py`가 에이전트의 역할(Generator, Reflector, Curator)을 인식하여 `thinking_effort`를 `low`, `medium`, `high`로 자동 할당합니다.
*   **상충 방지 로직 (Safety Guard)**: V3 사양상 `thinking`과 `Structured Outputs`(JSON 모드)는 동시 사용이 불가합니다. 이를 인식하여 Curator가 JSON 모드 사용 시 추론 옵션을 자동으로 조정하여 오류를 방지합니다.
*   **정밀한 토큰 사용량 집계**: NCloud 응답의 `result` 이벤트에서 `inputTokens`, `outputTokens`를 추출하여 정확한 비용 및 성능 분석이 가능하게 했습니다.

### 주요 수정 파일
*   **[MODIFY] [ncloud_llm.py](file:///Users/user/Hands-on/Agentic_Context_Engineering/ace/ncloud_llm.py)**: V3 사양 및 상충 방지 로직 반영.
*   **[MODIFY] [llm.py](file:///Users/user/Hands-on/Agentic_Context_Engineering/ace/llm.py)**: 역할 기반 추론 깊이 자동 할당 인터페이스 확장.
