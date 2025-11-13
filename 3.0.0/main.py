import boltzman
import logging
import time
import plot_momentum as plm
import os
import json

# 시뮬레이션 번호 설정 구문
def get_and_increment_count(config_file = 'simulation_config.json'):
    
    # config 파일이 없다면 생성
    if not os.path.exists(config_file):
        data = {'count': 0}

    else:
        with open(config_file, 'r') as f:
            data = json.load(f)

    current_count = data['count']

    data['count'] = current_count + 1
    with open(config_file, 'w') as f:
        json.dump(data, f)

    return current_count

if __name__ == '__main__':

    # 1. 시뮬레이션 데이터 저장할 폴더 생성
    count = get_and_increment_count() # 현재 시뮬레이션 번호 가져오기
    result_dir = f'results/run_{count}' # 폴더 주소 생성

    os.makedirs(f'{result_dir}', exist_ok= True)
    os.makedirs(f'{result_dir}/step', exist_ok= True)
    os.makedirs(f'{result_dir}/distribution', exist_ok= True)

    # 2. 로깅 설정

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level = logging.INFO, format = log_format, handlers=[logging.FileHandler(f"{result_dir}/simulation.log", encoding='utf-8'), logging.StreamHandler()])

    # 3. 시뮬레이션 실행
    logging.info(f"{count}번째 시뮬레이션 실행")
    simulation_start_time = time.time()

    # 시뮬레이션 설정, num_side = 입자개수, dt = 시간간격, total_time = 전체 시간, count = 시뮬레이션 번호, dlog = data 저장하는 step 간격
    num_side = 37
    dt = 2e-15
    total_time = 1.5e-10
    count = count
    dlog = 100

    # 저장 파일 예상 용량

    N = num_side ** 3
    bytes_per_file = N * 48
    mb_per_file = bytes_per_file / (1024 * 1024)
    num_file = (total_time / dt) / dlog
    total_mb_file = mb_per_file * num_file
    total_gb_file = total_mb_file / 1024

    logging.info(f'데이터 사전 용량 예측')
    logging.info(f'설정된 입자 수 : {N:,}개')
    logging.info(f'총 step 수 : {total_time / dt}개')
    logging.info(f'총 file 수 : {num_file}개')
    logging.info(f'파일 1개당 예상 Raw 데이터 크기 : 약{mb_per_file:.2f} MB')
    logging.info(f'예상 총 파일 크기 : {total_mb_file:.2f} MB | {total_gb_file:.2f} GB')

    # 함수 실행
    boltzman.main(num_side = num_side, dt = dt, total_time = total_time, count = count, dlog =  dlog)

    simulation_end_time = time.time()
    simulation_time = simulation_end_time - simulation_start_time
    logging.info(f"시뮬레이션 완료. 소요 시간 : {simulation_time:.2f}초")

    # 4. 시각화 (초가 결과, 최종 결과)
    plot_start_time = time.time()
    plm.main(0, f'momentum_distribution_inital.png', count)
    logging.info("초기 결과 시각화 완료")

    plm.main(-1, f'momentum_distribution_final.png', count)
    logging.info("최종 결과 시각화 완료")

    plot_end_time = time.time()
    plot_time = plot_end_time - plot_start_time
    logging.info(f"그래프 그리기 완료. 소요 시간 : {plot_time:.2f}초")

    logging.info(f"전체 프로그램 종료. 소요 시간 : {plot_end_time - simulation_start_time:.2f}초")