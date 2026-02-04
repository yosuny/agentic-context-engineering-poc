# NCloud HCX-007 Optimization Guide

본 문서는 ACE 프레임워크를 **NAVER Cloud CLOVA Studio v3 API (HCX-007)** 환경에 최적화하기 위해 적용한 기술적 전략과 구현 상세를 다룹니다.

## 1. HyperCLOVA X (HCX-007) 파라미터 튜닝

HCX-007 모델의 특성에 맞춰 기본 샘플링 파라미터를 조정했습니다. (`ace/ncloud_llm.py`)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **`topP`** | `0.8` | 다양성과 정확성의 균형. 너무 낮으면 창의성이 떨어지고, 너무 높으면 불안정해짐. |
| **`topK`** | `0` | `0`으로 설정하여 비활성화. `topP` 샘플링에 전적으로 의존. |
| **`repetitionPenalty`** | `1.1` | 한국어 생성 시 발생할 수 있는 문장 반복 현상을 억제하기 위한 약한 페널티 적용. |
| **`temperature`** | `0.5` | (Default) Generator/Reflector는 상황에 따라 유동적으로 조절. |

## 2. Thinking vs Structured Output 충돌 해결 (Safety Guard)

CLOVA Studio v3 API의 제약 사항 중 하나는 **"Thinking(추론 과정 출력) 기능과 Structured Output(JSON 모드)을 동시에 사용할 수 없다"**는 점입니다. ACE는 이 문제를 해결하기 위해 에이전트 역할별로 모드를 자동 전환하는 로직을 구현했습니다.

### Logic (`ncloud_llm.py`)
```python
# Handle Structured Outputs (JSON)
is_json_mode = response_format and response_format.get("type") == "json"

if is_json_mode:
    # [Conflict Resolution] JSON 모드 사용 시 Thinking 강제 비활성화
    effort = "none" 
    # [Schema Injection] NCloud 포맷에 맞춰 schema 필드 조립
    data["responseFormat"] = {"type": "json", "schema": schema}

if effort != "none":
    data["thinking"] = {"effort": effort} # Thinking 활성화
```

### Role-Based Strategy
*   **Generator**: `Thinking: High`, `JSON: Off` (복잡한 추론이 필요하므로 자유 텍스트 생성)
*   **Curator**: `Thinking: Off`, `JSON: On` (플레이북 수정이라는 정밀한 작업이 필요하므로 포맷 준수 최우선)

## 3. 에이전트별 추론 깊이 (Thinking Effort) 자동화

에이전트의 임무 복잡도에 따라 HCX-007의 추론 리로스(Thinking Effort)를 차등 배분하여 비용 효율성을 높였습니다. (`ace/llm.py`)

| Agent | Thinking Effort | Rationale |
| :--- | :--- | :--- |
| **Generator** | **High** | 사용자 의도 파악과 창의적 해법 도출에 가장 많은 리소스 투입. |
| **Reflector** | **Medium** | 논리적 오류 검증을 위한 적절한 수준의 추론 필요. |
| **Curator** | **Low / None** | 단순 편집 및 병합 작업 위주. JSON 모드 사용 시 자동으로 `None` 처리. |

## 4. 토큰 사용량 정밀 추적 (Stream Parsing)

NCloud v3의 Streaming 응답은 OpenAI와 포맷이 상이합니다. 정확한 비용 산정을 위해 `event: result` 이벤트를 파싱하여 토큰 사용량을 집계합니다.

```python
if current_event == "result":
    if "usage" in data_json:
        # NCloud v3 specific fields
        usage["prompt_tokens"] = data_json["usage"].get("inputTokens", 0)
        usage["completion_tokens"] = data_json["usage"].get("outputTokens", 0)
```

이 로직을 통해 스트리밍 중에도 정확한 과금 정보를 로그에 남길 수 있습니다.
