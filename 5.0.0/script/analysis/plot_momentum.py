import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import logging

import utility.project_setup # type: ignore

BOLTZMANN_CONSTANT = 1.380649e-23 # J/K 

def getdir(config):

    base_dir = config['file_base_path']
    return base_dir

def main(step_index, filename, count, config, root_dir=None):
    """
    step_index (a): 분석할 스텝 파일의 인덱스 (-1은 마지막 파일)
    filename: 저장할 그래프 이미지 파일 이름
    count: 불러올 시뮬레이션 시행 번호 (예: 1 -> results/run_1)
    """

    if root_dir:
        base_dir = root_dir

    else:
        base_dir = config['file_base_path']

    m = config['physical_params']['m']

    # 1. count를 이용해 대상 폴더 경로 생성 (예: results/run_3)
    target_dir = os.path.join(base_dir, f'run_{count}/step')
    
    logging.info(f"[{count}번 시행] 운동량 분포 분석 시작...")
    logging.info(f"타겟 디렉토리: {target_dir}")

    # 폴더 존재 여부 확인
    if not os.path.exists(target_dir):
        logging.error(f"'{target_dir}' 폴더가 존재하지 않습니다.")
        return

    # 2. 해당 폴더 내의 모든 스텝 파일 찾기
    file_pattern = os.path.join(target_dir, 'step_*.npz')
    
    # 파일 리스트 가져오기 및 정렬 (숫자 기준)
    step_files = sorted(glob.glob(file_pattern), key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

    if not step_files:
        logging.error(f"'{target_dir}' 디렉토리에서 npz 파일을 찾을 수 없습니다.")
        return

    # 3. 분석할 스텝 파일 선택 (인덱스 에러 방지 포함)
    try:
        final_step_file = step_files[step_index]
    except IndexError:
        logging.error(f"인덱스 {step_index}에 해당하는 파일이 없습니다. (파일 개수: {len(step_files)})")
        return

    logging.info(f"분석할 파일: {final_step_file}")

    try:
        data = np.load(final_step_file)
        if 'velocities' not in data:
            logging.error("데이터에 'velocities' 키가 없습니다.")
            return
        velocities = data['velocities']
    except Exception as e:
        logging.error(f"파일 로딩 중 오류 발생: {e}")
        return

    # --- 데이터 계산 로직 (기존과 동일) ---
    speeds = np.linalg.norm(velocities, axis=1)
    momenta = speeds * m

    finite_mask = np.isfinite(momenta)
    num_infinite = len(momenta) - np.sum(finite_mask)
    if num_infinite > 0:
        logging.warning(f"{num_infinite}개의 무한대 값 감지됨.")
    
    finite_momenta = momenta[finite_mask]

    # --- 히스토그램 그리기 ---
    logging.info("히스토그램 생성 중...")
    plt.figure(figsize=(12, 7))

    if len(finite_momenta) > 0:
        plt.hist(finite_momenta, bins=50, density=True, alpha=0.7, label='Simulation Data')
    else:
        logging.error("데이터가 없어 그래프를 그릴 수 없습니다.")
        return

    # --- 이론 곡선 그리기 ---
    if len(finite_momenta) > 1:
        finite_speeds = speeds[finite_mask]
        avg_kinetic_energy = 0.5 * m * np.mean(finite_speeds**2)
        kT = (2.0/3.0) * avg_kinetic_energy
        T_kelvin = kT / BOLTZMANN_CONSTANT
        
        logging.info(f"계산된 온도: {T_kelvin:.2f} K")

        if np.isfinite(kT) and kT > 0:
            p = np.linspace(0, np.max(finite_momenta), 500)
            with np.errstate(divide='ignore', invalid='ignore'):
                log_f_p = (0.5 * np.log(2/np.pi) + 
                           2 * np.log(p) - 
                           1.5 * np.log(m * kT) - 
                           p**2 / (2 * m * kT))
                f_p = np.exp(log_f_p)
                f_p[p == 0] = 0

            plt.plot(p, f_p, 'r-', lw=2, label=f'Maxwell-Boltzmann (T≈{T_kelvin:.2f} K)')

    plt.title(f'Momentum Distribution (Run {count}, Step Index {step_index})')
    plt.xlabel('Momentum Magnitude')
    plt.ylabel('Probability Density')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    
    # 4. 그래프 저장 (해당 run 폴더 안에 저장)
    target_dir = os.path.join(base_dir, f'run_{count}/distribution')
    save_path = os.path.join(target_dir, filename)
    plt.savefig(save_path)
    plt.close() # 메모리 누수 방지를 위해 닫기
    
    logging.info(f"그래프 저장 완료: {save_path}")

if __name__ == '__main__':

    config = project_setup.getConfig()
    base_dir = getdir(config)

    # 테스트를 위한 코드
    # 단독 실행 시 로그 출력을 위한 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        import matplotlib
        
        # 예시: results/run_3 폴더의 마지막 파일(-1)을 분석하고 graph.png로 저장
        # 실제 사용할 때는 이 값을 바꾸거나, 외부에서 main()을 호출하세요.
        target_run_count = 3 
        
        # 폴더가 실제로 있는지 확인 후 실행
        if os.path.exists(os.path.join(base_dir, f'run_{target_run_count}')):
             main(step_index=-1, filename='distribution_analysis.png', count=target_run_count)
        else:
            logging.warning(f"테스트를 수행하려 했으나 'results/run_{target_run_count}' 폴더가 없습니다.")
            logging.warning("코드 하단의 target_run_count 값을 실제 존재하는 폴더 번호로 변경해주세요.")
            
    except ImportError:
        logging.error("matplotlib이 설치되어 있지 않습니다.")