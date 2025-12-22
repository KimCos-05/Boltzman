import os
import json
import logging
import glob

# 몇번쨰 folder까지 생성되었는지 감지하는 구문
def get_current_count(base_dir):

    if not os.path.exists(base_dir):
        return 1

    search_pattern = os.path.join(base_dir, 'run_*')
    dirs = glob.glob(search_pattern)

    if not dirs:
        return 1

    max_count = 0
    for d in dirs:
        try:
            folder_name = os.path.basename(d)
            count = int(folder_name.split('_')[1])

            if count > max_count:
                max_count = count

        except (IndexError, ValueError):
            continue

    return max_count + 1 

# 시뮬레이션 config 데이터 읽는 구문
def getConfig(config_path):

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

# 프로젝트 폴더 생성 구문
def setup_result_directories(base_dir, count):

    result_dir = os.path.join(base_dir, f'run_{count}')

    os.makedirs(result_dir, exist_ok = True)
    os.makedirs(os.path.join(result_dir, 'step'), exist_ok = True)
    os.makedirs(os.path.join(result_dir, 'distribution'), exist_ok = True)

    return result_dir

# 실험 조건 저장 구문
def save_nowConfig(config, result_dir):

    save_path = os.path.join(result_dir, 'config.json')

    with open(save_path, 'w') as f:
        json.dump(config, f, indent = 4)

# 프로젝트 용량 예상 구문
def estimated_storge(config):

    log_prefix = f'[데이터 사전 용량 예측]'
    
    num_side = config['num_side']
    total_time = config['total_time']
    dt = config['dt']
    dlog = config['dlog']

    N = num_side ** 3
    bytes_per_file = N * 48
    mb_per_file = bytes_per_file / (1024 * 1024)
    num_file = (total_time / dt) / dlog
    total_mb_file = mb_per_file * num_file
    total_gb_file = total_mb_file / 1024

    logging.info(f' ')
    logging.info(f'{log_prefix} | 설정된 입자 수 : {N:,}개')
    logging.info(f'{log_prefix} | 총 step 수 : {total_time / dt}개')
    logging.info(f'{log_prefix} | 총 file 수 : {num_file}개')
    logging.info(f'{log_prefix} | 파일 1개당 예상 Raw 데이터 크기 : 약{mb_per_file:.2f} MB')
    logging.info(f'{log_prefix} | 예상 총 파일 크기 : {total_mb_file:.2f} MB | {total_gb_file:.2f} GB')