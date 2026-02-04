# ACE (Agentic Context Engineering) - Manufacturing Safety Pilot

> **Project**: `agentic-context-engineering-poc`
> **Domain**: Manufacturing Safety (제조 현장 안전)

본 프로젝트는 [Agentic Context Engineering (ACE)](https://arxiv.org/html/2510.04618v2) 프레임워크를 기반으로, **"스스로 진화하는 안전 가이드(Safety Playbook)"** 시스템을 구현하고 검증한 PoC(Proof of Concept)입니다.

## 📁 Project Structure

```text
agentic-context-engineering-poc/
├── ace/                 # ACE Core Framework (Generator, Reflector, Curator)
├── data/                # Evaluation Datasets (JSONL)
├── docs/                # Documentation & Analysis Reports
│   ├── reports/         # Evaluation Reports (1st & 2nd Round)
│   ├── DEV_LOG.md       # Development Log & Technical Details
│   └── ACE_FINAL_REVIEW.md # Final Strategic Review
├── experiments/         # Experiment & Test Scripts
├── results/             # Experiment Results (Best Run Only)
└── scripts/             # Utility Scripts
```

## 🚀 Key Features
1.  **Self-Evolving Playbook**: NCloud LLM(HCX-007)을 사용하여 안전 수칙을 스스로 생성, 수정, 최적화합니다.
2.  **Case Differentiation**: 상황(지게차, 화학, 기계)에 따라 적절한 대응 수칙을 분기하여 학습하는 능력을 갖췄습니다.
3.  **Conflict Resolution**: 서로 충돌하는 안전 수칙 간의 우선순위를 시행착오를 통해 학습합니다.

## 📚 Documentation
*   **[최종 검토 의견서 (Final Review)](docs/reports/ACE_FINAL_REVIEW.md)**: 프로젝트의 성과와 비즈니스 가치, 아키텍처 다이어그램 포함.
*   **[1차 평가 결과](docs/reports/1st_evaluation.md)**: 초기 LOTO 과적합 현상 분석.
*   **[2차 평가 결과](docs/reports/2nd_evaluation.md)**: 데이터셋 확장을 통한 상황별 분기 능력 검증.
*   **[NCloud 최적화 가이드](docs/NCLOUD_OPTIMIZATION_GUIDE.md)**: HCX-007 API 연동 기술 상세 (Thinking vs JSON).
*   **[개발 로그 (Dev Log)](docs/PROJECT_CONTEXT.md)**: 프로젝트 기술 스택 및 진행 이력.

## 🛠️ Getting Started
```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Set Environment Variables
# Create .env file with NCLOUD_API_KEY, NCLOUD_API_URL

# 3. Run Pilot Test
python experiments/manufacturing_ace_test_v2.py
```

## ⚖️ Attribution & License
This project is based on the **[ACE (Agentic Context Engineering)](https://github.com/ace-agent/ace)** framework.
*   **Original Source**: [https://github.com/ace-agent/ace](https://github.com/ace-agent/ace)
*   **License**: Licensed under the Apache License 2.0. See `LICENSE` file for details.
*   **Modification**: This repository (`agentic-context-engineering-poc`) is a Proof of Concept (PoC) implementation modified to work with **NAVER Cloud HyperCLOVA X (HCX-007)** and includes domain-specific scenarios (Manufacturing Safety).
