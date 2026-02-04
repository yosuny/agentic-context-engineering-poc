
import json
import random
import os

def generate_manufacturing_safety_data():
    """Generates evaluation dataset for Manufacturing Safety domain."""
    
    # 1. KOSHA-style Accident Scenarios (Seed Data)
    # These represent the 'structural knowledge' extracted from KOSHA accident reports.
    scenarios = [
        {
            "id": "CASE_001_CONVEYOR",
            "title": "컨베이어 이물질 제거 중 협착",
            "context_template": "{worker}가 {location}에서 컨베이어 작동 중 {activity}를 하던 중, 벨트와 롤러 사이에 {body_part}가 끼임.",
            "root_causes": ["LOTO(Lock-Out, Tag-Out) 미이행", "비상정지장치 미가동", "방호덮개 개방"],
            "correct_action": "컨베이어 운전 정지 및 전원 차단 후 작업, 잠금장치 및 표지판(LOTO) 설치",
            "symptoms": ["이물질 발견", "소음 발생", "벨트 편중"],
            "difficulty": "Medium"
        },
        {
            "id": "CASE_002_FORKLIFT",
            "title": "지게차 운반 작업 중 충돌",
            "context_template": "{worker}가 {location}에서 지게차로 {load} 운반 작업을 하던 중, 시야 확보 미흡으로 {victim}와 충돌함.",
            "root_causes": ["지게차 유도자 미배치", "전방 시야 미확보", "제한속도 위반"],
            "correct_action": "작업지휘자 또는 유도자 배치, 제한속도(10km/h 이하) 준수, 경보장치 작동",
            "symptoms": ["적재물 과다", "급선회", "후진 이동"],
            "difficulty": "Easy"
        },
        {
            "id": "CASE_003_PIPE_LEAK",
            "title": "화학배관 플랜지 교체 중 유해물질 누출",
            "context_template": "{worker}가 {location}에서 낡은 배관 플랜지를 교체하기 위해 {tool}을 사용하여 볼트를 풀던 중 잔류 {substance}이 비산되어 {injury_type}을 입음.",
            "root_causes": ["잔류 가스/액체 퍼지(Purge) 미실시", "개인보호구(PPE) 미착용", "작업허가서 미발행"],
            "correct_action": "작업 전 배관 내용물 비우기 및 세정(Purge), 내화학 보호구 착용, 밀폐공간 작업 허가 획득",
            "symptoms": ["밸브 부식", "압력 게이지 미확인", "냄새 발생"],
            "difficulty": "Hard"
        }
    ]

    # 2. Variable Slots for Augmentation
    variables = {
        "worker": ["작업자 A씨", "정비팀 김 대리", "신입사원 이 군", "협력업체 박 반장"],
        "location": ["제2공장 조립라인", "물류창고 B동", "화학제품 출하장", "식품가공 포장실", "반도체 세정 공정"],
        "activity": ["끼인 포장박스 제거", "롤러 청소", "센서 위치 조정", "낙하물 수거"],
        "body_part": ["오른손", "작업복 소매", "장갑", "팔"],
        "load": ["파레트", "원료 드럼통", "완제품 박스", "폐기물 톤백"],
        "victim": ["보행 중인 동료", "현장 순찰자", "청소 작업자"],
        "tool": ["임팩트 렌치", "스패너", "그라인더"],
        "substance": ["황산", "가성소다", "유기용제", "고온의 스팀"],
        "injury_type": ["화상", "질식", "피부 발진", "호흡 곤란"]
    }

    # 3. Generate Samples
    dataset = []
    
    # Phase 1: Training (20 samples)
    # Focus on clear, textbook cases where applying the rule directly solves it.
    for i in range(20):
        scenario = random.choice(scenarios)
        context = scenario["context_template"].format(
            worker=random.choice(variables["worker"]),
            location=random.choice(variables["location"]),
            activity=random.choice(variables["activity"]) if "{activity}" in scenario["context_template"] else "",
            body_part=random.choice(variables["body_part"]) if "{body_part}" in scenario["context_template"] else "",
            load=random.choice(variables["load"]) if "{load}" in scenario["context_template"] else "",
            victim=random.choice(variables["victim"]) if "{victim}" in scenario["context_template"] else "",
            tool=random.choice(variables["tool"]) if "{tool}" in scenario["context_template"] else "",
            substance=random.choice(variables["substance"]) if "{substance}" in scenario["context_template"] else "",
            injury_type=random.choice(variables["injury_type"]) if "{injury_type}" in scenario["context_template"] else ""
        )
        
        # Simple Question Formulation
        question = f"다음 상황에서 사고의 근본 원인과 재발 방지 대책은 무엇인가?\n상황: {context}"
        
        target = f"근본 원인: {', '.join(scenario['root_causes'])}\n대책: {scenario['correct_action']}"
        
        dataset.append({
            "phase": "train",
            "id": f"train_{i:03d}",
            "scenario_id": scenario["id"],
            "question": question,
            "target": target,
            "context": context,
            "difficulty": scenario["difficulty"]
        })

    # Phase 2: Validation (10 samples)
    # Slightly ambiguous or "Trap" cases where workers often make excuses.
    for i in range(10):
        scenario = random.choice(scenarios)
        context = scenario["context_template"].format(
            worker=random.choice(variables["worker"]),
            location=random.choice(variables["location"]),
            activity=random.choice(variables["activity"]) if "{activity}" in scenario["context_template"] else "",
            body_part=random.choice(variables["body_part"]) if "{body_part}" in scenario["context_template"] else "",
            load=random.choice(variables["load"]) if "{load}" in scenario["context_template"] else "",
            victim=random.choice(variables["victim"]) if "{victim}" in scenario["context_template"] else "",
            tool=random.choice(variables["tool"]) if "{tool}" in scenario["context_template"] else "",
            substance=random.choice(variables["substance"]) if "{substance}" in scenario["context_template"] else "",
            injury_type=random.choice(variables["injury_type"]) if "{injury_type}" in scenario["context_template"] else ""
        )
        
        # Trap: Worker claims they were careful or in a hurry.
        trap_context = context + " 작업자는 '급해서 어쩔 수 없었다'고 주장하고 있음."
        question = f"다음 사고 상황을 분석하고, 작업자의 주장에도 불구하고 반드시 지켰어야 할 핵심 안전 수칙을 제시하라.\n상황: {trap_context}"
        
        target = f"작업자의 주장은 수용될 수 없음. 필수 안전 수칙: {scenario['correct_action']}"
        
        dataset.append({
            "phase": "validation",
            "id": f"val_{i:03d}",
            "scenario_id": scenario["id"],
            "question": question,
            "target": target,
            "context": trap_context,
            "difficulty": "Hard"
        })

    # 4. Save to JSONL
    output_path = os.path.join(os.path.dirname(__file__), '../tests/manufacturing_eval_v1.jsonl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"✅ Generated {len(dataset)} samples at {output_path}")
    print("Sample 0:", json.dumps(dataset[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    generate_manufacturing_safety_data()
