import boltzman
import project_setup
import plot_momentum as plm

import logging
import time

if __name__ == '__main__':

    # 1. 시뮬레이션 데이터 불러오기
    config, count = project_setup.getConfig()

    # 2. 시뮬레이션 파일 경로 설정
    base_dir = config['file_base_path']
    result_dir = project_setup.setup_result_directories(base_dir, count)
    project_setup.save_nowConfig(config, count)

    # 3. 로깅 설정
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level = logging.INFO, format = log_format, 
                        handlers=[logging.FileHandler(f"{result_dir}/simulation.log", encoding='utf-8'), 
                                  logging.StreamHandler()])

    # 4. 시뮬레이션 실행
    logging.info(f"{count}번째 시뮬레이션 실행")
    simulation_start_time = time.time()

    # 4.1 예상 용량 로깅
    project_setup.estimated_storge(config)

    # 4.2 함수 실행
    boltzman.main(sim_params=config['physical_params'],
                  run_params=config,
                  result_dir = result_dir,
                  count = count)

    simulation_end_time = time.time()
    simulation_time = simulation_end_time - simulation_start_time
    logging.info(f" ")
    logging.info(f"시뮬레이션 완료. 소요 시간 : {simulation_time:.2f}초")

    # 5. 시각화 (초가 결과, 최종 결과)
    plot_start_time = time.time()
    plm.main(0, f'momentum_distribution_inital.png', count, config)
    logging.info("초기 결과 시각화 완료")

    plm.main(-1, f'momentum_distribution_final.png', count, config)
    logging.info("최종 결과 시각화 완료")

    plot_end_time = time.time()
    plot_time = plot_end_time - plot_start_time
    logging.info(f"그래프 그리기 완료. 소요 시간 : {plot_time:.2f}초")

    logging.info(f"전체 프로그램 종료. 소요 시간 : {plot_end_time - simulation_start_time:.2f}초")