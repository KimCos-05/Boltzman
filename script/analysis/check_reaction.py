import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main(run_count):
    # 1. 경로 설정 
    # 현재 실행 중인 파일(check_reaction.py)의 위치를 기준으로 상위 폴더를 탐색합니다.
    current_file_path = os.path.abspath(__file__)
    analysis_dir = os.path.dirname(current_file_path) # script/analysis
    script_dir = os.path.dirname(analysis_dir)        # script
    
    # 예상되는 results 폴더 위치 후보들
    possible_paths = [
        os.path.join(script_dir, 'results'),          # script/results (가장 유력)
        os.path.join(script_dir, '..', 'results'),    # 프로젝트 루트/results
        "results"                                     # 현재 실행 위치/results
    ]

    base_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            base_dir = path
            print(f"폴더를 찾았습니다: {base_dir}")
            break
    
    if base_dir is None:
        print("오류: 'results' 폴더를 찾을 수 없습니다.")
        print(f"탐색한 경로들: {possible_paths}")
        return

    run_dir = os.path.join(base_dir, f'run_{run_count}')
    step_dir = os.path.join(run_dir, 'step')
    config_path = os.path.join(run_dir, 'config.json')

    # 2. 설정 파일(dt) 읽기
    if not os.path.exists(config_path):
        print(f"오류: {config_path} 가 없습니다.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dt = config.get('dt', 1e-15) # dt를 못 찾으면 기본값
    
    # 3. 파일 리스트 가져오기
    file_pattern = os.path.join(step_dir, 'step_*.npz')
    files = glob.glob(file_pattern)
    
    # 파일명에서 숫자만 추출해서 정렬 (step_1, step_2, ... step_10 순서대로)
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if not files:
        print("데이터 파일이 없습니다.")
        return

    print(f"총 {len(files)}개의 스텝 파일을 분석합니다...")

    # 4. 데이터 추출
    times = []
    counts_reactant = [] # 반응물 (Type 1)
    counts_product = []  # 생성물 (Type 2 or 3)

    for filepath in files:
        try:
            # npz 파일 로드
            data = np.load(filepath)
            
            # types 배열이 있는지 확인
            if 'types' not in data:
                print(f"경고: {os.path.basename(filepath)} 파일에 'types' 데이터가 없습니다. (boltzman.py 수정 후 다시 돌리셨나요?)")
                continue
                
            types = data['types']
            
            # 스텝 번호 추출 -> 시간 계산
            step_num = int(os.path.basename(filepath).split('_')[1].split('.')[0])
            current_time = step_num * dt
            
            # 개수 세기
            n_reactant = np.sum(types == 1) # Type 1: 반응물
            n_product = np.sum(types >= 2)  # Type 2 이상: 생성물
            
            times.append(current_time)
            counts_reactant.append(n_reactant)
            counts_product.append(n_product)

        except Exception as e:
            print(f"파일 읽기 오류 ({filepath}): {e}")

    # numpy 배열로 변환
    times = np.array(times)
    counts_reactant = np.array(counts_reactant)
    
    # 데이터가 비었으면 종료
    if len(times) == 0:
        print("유효한 데이터가 없습니다.")
        return

    # 0으로 나누기 방지 (로그 계산용)
    # 반응물이 0개인 데이터는 제외하거나 아주 작은 값 더해주기
    valid_mask = counts_reactant > 0
    times_valid = times[valid_mask]
    counts_valid = counts_reactant[valid_mask]

    # 5. 그래프 그리기 (3단 콤보)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) 농도 vs 시간 ([A] vs t)
    axes[0].plot(times, counts_reactant, 'b-', label='Reactant (Type 1)')
    axes[0].plot(times, counts_product, 'r--', label='Product (Type 2+)')
    axes[0].set_title('Concentration vs Time')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Number of Particles')
    axes[0].legend()
    axes[0].grid(True)

    # (2) 1차 반응 검증 (ln[A] vs t)
    # 직선이면 1차 반응
    ln_A = np.log(counts_valid)
    axes[1].plot(times_valid, ln_A, 'g-')
    axes[1].set_title('First Order Check (ln[A] vs t)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('ln(N)')
    axes[1].grid(True)
    
    # 추세선 그리기 (상관계수 확인)
    if len(times_valid) > 1:
        slope, intercept = np.polyfit(times_valid, ln_A, 1)
        axes[1].plot(times_valid, slope*times_valid + intercept, 'k:', alpha=0.5, label=f'R-sq check')
        axes[1].legend()

    # (3) 2차 반응 검증 (1/[A] vs t)
    # 직선이면 2차 반응
    inv_A = 1.0 / counts_valid
    axes[2].plot(times_valid, inv_A, 'm-')
    axes[2].set_title('Second Order Check (1/[A] vs t)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('1 / N')
    axes[2].grid(True)

    plt.tight_layout()
    
    save_path = os.path.join(run_dir, 'distribution', 'reaction_kinetics.png')
    # 폴더가 없을 수 있으니 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    print(f"분석 완료! 그래프가 저장되었습니다: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reaction Kinetics Analyzer")
    parser.add_argument('--run', type=int, required=True, help="Run number to analyze (e.g., 1)")
    args = parser.parse_args()
    
    main(args.run)