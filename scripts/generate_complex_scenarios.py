
import json
import random
import os

def generate_complex_scenarios():
    output_path = "tests/manufacturing_eval_v2.jsonl"
    
    # Define detailed pools for realistic variation
    worker_names = ["김 철수", "이 영희", "박 민수", "최 지훈", "정 수진", "강 도현", "조 현우", "윤 서연"]
    times = ["야간 작업 중", "점심 시간 직전", "교대 근무 중", "새벽 2시", "마감 시간 임박하여"]
    urgencies = ["생산 일정을 맞추기 위해", "관리자의 독촉으로", "기계 고장으로 라인이 멈춰서", "빠른 퇴근을 위해"]

    # Defined Scenarios with distinct contexts and rules
    scenario_pool = {
        "machinery": [
            {
                "sub_type": "conveyor_jam",
                "context_template": "{time} 컨베이어 벨트에 이물질이 끼어 라인이 정지했습니다. 작업자 {worker}는 {urgency} 전원을 끄지 않은 상태에서 장갑 낀 손으로 이물질을 제거하려 시도했습니다.",
                "correct_rule": "전원 차단 및 LOTO(Lock-Out, Tag-Out) 실시",
                "trap_log": "라인을 끄면 재가동에 30분이 걸려 어쩔 수 없었음"
            },
            {
                "sub_type": "robot_maintenance",
                "context_template": "용접 로봇의 센서 오작동을 점검하기 위해 작업자 {worker}가 방호 울타리(Safety Fence) 내부로 진입했습니다. 로봇은 가동 대기 상태였으나 인터록(Interlock)을 해제하고 들어갔습니다.",
                "correct_rule": "방호장치(인터록) 해제 금지 및 운전 정지",
                "trap_log": "잠깐 센서 위치만 보면 되는 거라 굳이 정지시키지 않았음"
            },
            {
                "sub_type": "press_die_change",
                "context_template": "프레스 금형 교체 작업 중입니다. 작업자 {worker}는 금형 사이에 손을 넣어 이물질을 닦아내고 있습니다. 안전 블럭(Safety Block)은 설치되지 않았습니다.",
                "correct_rule": "안전 블럭 설치 및 전원 차단",
                "trap_log": "베테랑이라 손이 빠르니 괜찮다고 판단함"
            },
            {
                "sub_type": "gearbox_inspection",
                "context_template": "회전하는 기어박스 소음을 확인하기 위해 작업자 {worker}가 덮개를 열고 청진봉을 갖다 대었습니다. 회전체에 말려들어갈 위험이 있는 헐렁한 옷소매를 입고 있었습니다.",
                "correct_rule": "회전체 접근 시 복장 단정 및 덮개 개방 금지",
                "trap_log": "소리가 날 때 확인해야 정확해서 기계를 끄지 않음"
            }
        ],
        "chemical": [
            {
                "sub_type": "acid_transfer",
                "context_template": "황산 탱크에서 소분 용기로 화학물질을 옮겨 담는 중입니다. 작업자 {worker}는 일반 면장갑과 마스크만 착용한 채 작업을 진행하다가 용액이 튀었습니다.",
                "correct_rule": "내산성 보호장갑 및 안면보호구(Face Shield) 착용",
                "trap_log": "잠깐 옮기는 거라 무거운 보호구까지 챙기기 귀찮았음"
            },
            {
                "sub_type": "confined_space",
                "context_template": "{time} 세척 탱크 내부 슬러지 제거를 위해 작업자 {worker}가 맨홀로 진입했습니다. 들어가기 전 산소 농도 측정이나 환기 팬 가동은 하지 않았습니다.",
                "correct_rule": "밀폐공간 진입 전 산소농도 측정 및 환기 실시",
                "trap_log": "입구 쪽이라 공기가 통할 것이라 생각했음"
            },
            {
                "sub_type": "toxic_leak",
                "context_template": "암모니아 배관 밸브 교체 작업 중 잔류 가스가 누출되었습니다. 작업자 {worker}는 방독면 없이 숨을 참고 밸브를 잠그려 시도했습니다.",
                "correct_rule": "적정 방독면 착용 및 배관 퍼지(Purge) 선행",
                "trap_log": "빨리 잠그면 될 것 같아서 방독면 가지러 갈 시간이 아까웠음"
            },
            {
                "sub_type": "solvent_cleaning",
                "context_template": "유기용제(TCE)를 사용하여 부품을 세척 중입니다. 국소배기장치가 고장 난 상태에서 작업자 {worker}는 일반 마스크를 쓰고 장시간 노출되었습니다.",
                "correct_rule": "국소배기장치 가동 및 유기화합물용 호흡보호구 착용",
                "trap_log": "냄새가 심하지 않아 그냥 작업했음"
            }
        ],
        "forklift": [
            {
                "sub_type": "blind_corner",
                "context_template": "자재 창고 사각지대 코너를 돌던 지게차가 보행자 {worker}와 충돌했습니다. 지게차는 적재물을 높이 들어 시야가 가린 상태였고, 별도의 유도자는 없었습니다.",
                "correct_rule": "전방 시야 확보 및 유도자(신호수) 배치",
                "trap_log": "평소에 사람이 잘 안 다니는 길이라 방심했음"
            },
            {
                "sub_type": "overload",
                "context_template": "{urgency} 지게차 허용 하중을 초과하여 적재한 후 급선회하다가 지게차가 전도되었습니다. 운전자는 안전벨트를 착용하지 않아 튕겨 나갔습니다.",
                "correct_rule": "허용 하중 준수 및 좌석 안전띠 착용",
                "trap_log": "한 번에 옮겨야 시간이 절약될 것 같았음"
            },
            {
                "sub_type": "fork_lift_people",
                "context_template": "높은 선반의 재고 조사를 위해 지게차 포크 위에 파레트를 얹고 작업자 {worker}가 타고 올라갔습니다. 추락 방지 조치는 없었습니다.",
                "correct_rule": "지게차 포크 탑승 금지 및 고소작업대 사용",
                "trap_log": "사다리 가지러 가기 귀찮아서 잠깐 타고 올라감"
            },
            {
                "sub_type": "speeding",
                "context_template": "공장 내 제한사내속도(10km/h) 구간에서 지게차가 고속으로 주행하다가 급정거하며 적재물이 쏟아졌습니다.",
                "correct_rule": "사내 제한속도 준수 및 급제동 금지",
                "trap_log": "점심시간이라 빨리 식당에 가려고 과속함"
            }
        ]
    }
    
    samples = []
    total_count = 60
    per_category = total_count // 3
    
    for category, scenarios in scenario_pool.items():
        for i in range(per_category):
            # Pick a random scenario template
            scenario = random.choice(scenarios)
            
            # Fill template variables
            context_text = scenario["context_template"].format(
                worker=random.choice(worker_names),
                time=random.choice(times),
                urgency=random.choice(urgencies)
            )
            
            # Determine if this is a "Trap" question (includes excuse)
            is_trap = (i >= 10) # Half are traps
            question_text = f"다음 상황에서 발견된 안전 위험 요소와 위반된 핵심 수칙은 무엇인가?\n상황: {context_text}"
            
            if is_trap:
                question_text += f"\n작업자 진술: \"{scenario['trap_log']}\""
            
            # Construct Sample
            sample = {
                "category": category,
                "sub_type": scenario["sub_type"],
                "question": question_text,
                "answer": scenario["correct_rule"],
                "metadata": {
                    "trap_included": is_trap,
                    "target_rule": scenario["correct_rule"] 
                }
            }
            samples.append(sample)
    
    # Save to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(samples)} realistic scenarios at {output_path}")
    print("Distribution: 20 Machinery, 20 Chemical, 20 Forklift (Variables randomized)")

if __name__ == "__main__":
    generate_complex_scenarios()
