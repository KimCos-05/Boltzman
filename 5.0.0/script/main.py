import sys
import os
import argparse
import logging
import time

# module 경로 지정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import utility.project_setup as project_setup
import core.boltzman as boltzman
import analysis.plot_momentum as plot_momentum

def main():

    # 1. config 폴더 내 원하는 파일 지정해서 실행  
    parser = argparse.ArgumentParser(description="MD Simulation V2")
    parser.add_argument('--config', type=str, default='../config/default.json', 
                        help='실행할 설정 파일의 경로 (예: ../config/test.json)') # 기본 지정 파일 : default.json
    args = parser.parse_args() # 파일 경로 읽어옴

    # 2. 시뮬레이션 데이터 불러오기
    # 경로가 실제로 존재하는지 확인
    config_path = args.config
    if not os.path.exists(config_path):
        # 상대 경로 문제일 수 있으니 절대 경로로 변환 시도
        config_path = os.path.join(os.path.dirname(current_dir), args.config)
        if not os.path.exists(config_path):
            print(f"[Error] 설정 파일을 찾을 수 없습니다: {args.config}")
            return
        
    # config 값 읽기
    config = project_setup.getConfig(config_path) 

    # 3. 시뮬레이션 파일 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir_name = config.get('file_base_path', 'results')
    base_dir = os.path.join(project_root, base_dir_name)

    count = project_setup.get_current_count(base_dir)

    result_dir = project_setup.setup_result_directories(base_dir, count)
    project_setup.save_nowConfig(config, result_dir)

    # 4. 로깅 설정
    log_path = os.path.join(result_dir, "simulation.log")
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, 
                        format=log_format,
                        handlers=[logging.FileHandler(log_path, encoding='utf-8'),
                                  logging.StreamHandler()])
    
    logging.info(f"Now : MD Simulation V2")
    logging.info(f"사용된 Config : {args.config}")
    logging.info(f"결과 저장 위치 : {result_dir}")

    # 5. 시뮬레이션 실행
    logging.info(f"{count}번째 시뮬레이션 실행")
    simulation_start_time = time.time()

    # 5.1 예상 용량 로깅
    project_setup.estimated_storge(config)

    # 5.2 함수 실행
    boltzman.main(sim_params=config['physical_params'],
                  run_params=config,
                  result_dir = result_dir,
                  count = count)

    simulation_end_time = time.time()
    simulation_time = simulation_end_time - simulation_start_time
    logging.info(f" ")
    logging.info(f"시뮬레이션 완료. 소요 시간 : {simulation_time:.2f}초")

    # 6. 시각화 (초가 결과, 최종 결과)
    plot_start_time = time.time()

    try:
        plot_momentum.main(0, f'momentum_distribution_inital.png', count, config, root_dir=base_dir)
        logging.info("초기 결과 시각화 완료")
    except Exception as e:
        logging.error(f"초기 결과 운동량 분포 시각화 중 오류 발생 : {e}")

    try:
        plot_momentum.main(-1, f'momentum_distribution_final.png', count, config, root_dir=base_dir)
        logging.info("최종 결과 시각화 완료")
    except Exception as e:
        logging.error(f"최종 결과 운동량 분포 시각화 중 오류 발생 : {e}")

    plot_end_time = time.time()
    plot_time = plot_end_time - plot_start_time
    logging.info(f"그래프 그리기 완료. 소요 시간 : {plot_time:.2f}초")

    logging.info(f"전체 프로그램 종료. 소요 시간 : {plot_end_time - simulation_start_time:.2f}초")

if __name__ == '__main__':
    main()