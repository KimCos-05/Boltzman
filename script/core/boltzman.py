import time
import numpy as np
import cupy as cp
import logging
import os

from core.particle_setup import setParticle, setParticleSpeed

# CDUA 스크립트 경로 얻기
current_dir = os.path.dirname(os.path.abspath(__file__))
kernel_dir = os.path.join(current_dir, '..', 'cuda_kernel')

# CUDA 스크립트 불러오기
with open(os.path.join(kernel_dir, 'build_list.cu'), 'r', encoding='utf-8') as f:
    build_list_kernel_src = f.read()

with open(os.path.join(kernel_dir, 'collision.cu'), 'r', encoding='utf-8') as f:
    collision_kernel_src = f.read()

# C++ 커널 컴파일
# 생성된 함수는 함수의 매개변수 외에도 GPU에 작업을 할당하는데 사용되는 2개의 인자를 추가로 보내주어야 함.
# CUDA 함수 구조 : 함수명((grid dimensions), (block dimensions), (function argument))
# grid dimensions : block의 개수. 함수 내부에서는 blockIdx가 block 수의 배열[0,N]으로 인식됨
# block dimensions : block당 thread의 개수. 함수 내부에서는 threadIdx가 thread 수의 배열 [0,N]으로 인식됨
build_list_kernel = cp.RawKernel(build_list_kernel_src, 'build_linked_list')
collision_kernel = cp.RawKernel(collision_kernel_src, 'collide_particles', options=('-use_fast_math',))

# 격자 알고리즘을 위한 격자 생성 함수.
def setup_grid(sx, sy, sz, r): # 전체 공간의 크기와 격자를 생성하는 기준이 되는 반지름 r의 값 가져옴

    d = 2.0 * r 
    cell_size = d * 7 # 격자 하나에 입자가 4개정도 들어가게 하는 것이 적정

    #격자의 개수 계산 함수. max(1, ~)은 우측의 값과 1을 비교해서 더 큰 값(최소 1칸 보장 위함)을 저장하는 함수
    nx = max(1, int(sx // cell_size))
    ny = max(1, int(sy // cell_size))
    nz = max(1, int(sz // cell_size))
    logging.info(f"격자 생성 : {nx} x {ny} x {nz}")
    return cell_size, nx, ny, nz

# 남은 시간 출력 함수
def estimated_left(step, steps, log_count, dlog, last_time, avg_time_per_log):

    log_prefix = f'[시뮬레이션 진행]'

    # 걸린 시간 계산
    current_time = time.time()
    deltatime = current_time - last_time

    # 이상 감지 구문
    if log_count < 20:
        avg_time_per_log = (avg_time_per_log * log_count + deltatime) / (log_count + 1) # 누적 평균시간 계산

    else:
        avg_time_per_log = avg_time_per_log * 0.9 + deltatime * 0.1

        if deltatime > avg_time_per_log * 1.5:
            logging.warning(f'{log_prefix} | 시뮬레이션 속도 저하 감지 | 평균 : {avg_time_per_log:.2f} , 현재 : {deltatime:.2f}')

    log_count = log_count + 1

    # 남은 시간 계산
    remain_steps = steps - (step + 1)
    estimated_seconds = (deltatime / dlog) * remain_steps

    m, s = divmod(estimated_seconds, 60)
    h, m = divmod(m, 60)

    if h > 0:
        time_str = f'{int(h)}시간 {int(m)}분 {int(s)}초'
    elif m > 0:
        time_str = f'{int(m)}분 {int(s)}초'
    else:
        time_str = f'{int(s)}초'

    progress = ((step + 1) / steps) * 100

    logging.info(f"{log_prefix} | Step {step+1}/{steps} ({progress:.1f}%)완료 | 소요시간 : {deltatime:.2f}초 / 평균대비 {deltatime - avg_time_per_log:.2f}초 | 예상 남은시간 : {time_str}") # 로그 출력
    
    return current_time, log_count, avg_time_per_log


# 시뮬레이션 알고리즘
# num_size의 3승이 전체 입자의 개수, total_time은 총 시뮬레이션 시간, count는 파일 번호
def main(sim_params, run_params, result_dir, count):

    # 1. 파라미터 설정 불러오기

    # 입자의 정보
    sx, sy, sz = sim_params['sx'], sim_params['sy'], sim_params['sz'] # 시뮬레이션 공간 크기 (단위 : m)
    r = sim_params['r']                                               # 입자의 반지름 (단위 : m)
    m = sim_params['m']                                               # 입자의 질량 (단위 : kg)
    T = sim_params['T']                                               # 시뮬레이션 온도 (단위 : K)

    # 시뮬레이션 정보
    dt = run_params['dt']                                             # 시뮬레이션 1step의 시간 간격
    total_time = run_params['total_time']                             # 시뮬레이션 전체 시간
    dlog = run_params['dlog']                                         # 시뮬레이션 기록을 남길 step 간격
    num_side = run_params['num_side']                                 # 입자의 한 방향당 개수

    steps = int(total_time / dt)         # 총 계산해야하는 step 개수 계산
    collision_count = cp.zeros(1, dtype=cp.uint64) # 충돌 횟수 변수

    logging.info(f" ")
    logging.info(f"시뮬레이션 시작: N={num_side**3}, steps={steps}")
    t0 = time.time()

    # 2. 입자 생성 및 CPU 데이터를 GPU로 전송
    pos_cpu, vel_cpu, prop_cpu, type_cpu = setParticle(sx, sy, sz, num_side, r, m) # 입자의 Position 생성
    positions = cp.asarray(pos_cpu)
    velocities = cp.asarray(vel_cpu)
    properties = cp.asarray(prop_cpu)
    types = cp.asarray(type_cpu)
    N = positions.shape[0] # 생성된 입자의 전체 개수 정의
    velocities = setParticleSpeed(velocities, properties, T, m) #입자에 속도 부여

    #초깃값 저장
    logging.info(f"입자 정보 설정 완료") # 로그 출력
    cp.savez_compressed(os.path.join(result_dir, 'step','step_0.npz'), positions=cp.asnumpy(positions), velocities=cp.asnumpy(velocities), types=cp.asnumpy(types)) #물성 결과(positions, velocities) 저장


    # 3. 격자 알고리즘의 설정
    cell_size, nx, ny, nz = setup_grid(sx, sy, sz, r) # 격자 생성. cell의 size와 각 축별 격자의 수 구하기
    ncell = nx * ny * nz #격자의 개수

    #linked list 구현을 위한 head와 nxt 배열 생성.
    head = cp.full(ncell, -1, dtype=cp.int32) # 각 격자 내 입자를 linked list 형태로 저장할 때, 연결고리의 첫번째 입자의 정보.
    nxt = cp.full(N, -1, dtype=cp.int32) # 그 입자 뒤에 연결되어있는 입자의 정보.

    # 4. 병렬 연산 설정. GPU에 명령을 할당하기 위한 구문
    threads_per_block = 128 # 1개의 block에 할당되는 thread의 개수
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block # 전체 명령을 시행하기 위해 필요한 block의 개수.
    # ( N + threads_per_block - 1 ) // threads_per_block은 올림을 하기 위한 연산 방법임.
    # 블록당 10개의 스레드, 93개의 연산을 수행해야 하면 N // TPB = 9로 3개의 연산을 수행하지 못함.

    # CPU가 GPU에 연산을 할당하는 구조
    # 1) CPU가 GPU에 명령어와 데이터를 전달함
    # 2) GPU 스케쥴러가 명령을 block로 분리하여 SM에 할당함
    # 3) SM에서 1block을 32스레드씩 4개의 warp로 분리함
    # 4) SM 내의 warp 스케쥴러가 SM 내의 CUDA에게 연산 지시.
    # 
    # 사용하는 RTX 3050은 2048개의 CUDA를 가지며, 16개의 SM이 128개의 CUDA를 지휘.
    # 즉, 1block당 128개의 스레드를 할당하면, 1개의 CUDA마다 1개의 연산이 지시되므로 본 설정이 가장 효율적인 상황임. 
    # 
    # 예시 : 20 * 20 * 20개의 입자에 대한 연산을 수행. 블럭당 128스레드 할당. 총 필요한 연산 수가 8000개라고 가정.
    # 필요한 block의 개수 : ( 8000 + 128 - 1 ) // 128 = 63개. CPU가 연산지시하면 GPU 스케쥴러가 16개의 SM에 63개의 block 배열.
    # 순차적으로 배열하고, 먼저 끝나는 block이 있으면 거기에 바로 다음 block 넣음. (동적 순환 방식)

    # 5. 시뮬레이션 실행

    t0 = time.time()
    last_time = t0

    avg_time_per_log = 0.0
    log_count = 0

    # 시뮬레이션은 step 단위로 진행. 하나의 step 루프는 CPU에서 진행되므로, 이전 단계가 완료되어야 다음 단계로 진행됨.
    for step in range(steps): # step의 개수만큼 반복
        positions += velocities * dt # Particle를 움직이는 연산

        # 벽과의 충돌
        for axis, bound in enumerate([sx, sy, sz]): # 총 3번의 루프로 x축, y축, z축을 검사.
            mask = (positions[:, axis] < 0) | (positions[:, axis] > bound) # 입자의 위치가 경계 밖에 있는지를 판단
            if mask.any(): 
                velocities[mask, axis] *= -1.0 # 해당 방향의 속도를 반대로 바꾸어 튕겨내는 연산 수행
            positions[:, axis] = cp.clip(positions[:, axis], 0.0, bound) # position을 조정(경계 밖에서 안으로 데려오기)

        head.fill(-1) #head의 초기화

        # 격자의 생성 및 격자에 입자 배치
        build_list_kernel((blocks_per_grid,), (threads_per_block,),
                          (positions.ravel(), head, nxt, np.int32(N),
                           np.float64(cell_size), np.int32(nx), np.int32(ny), np.int32(nz)))
    
        # 충돌의 인식 및 물리적 연산 시행
        collision_kernel((blocks_per_grid,), (threads_per_block,),
                         (positions.ravel(), velocities.ravel(), properties.ravel(), types,
                          head, nxt, np.int32(N), np.float64(cell_size),
                          np.int32(nx), np.int32(ny), np.int32(nz), collision_count))

        # 결괏값 저장 및 로그 출력
        if (step + 1) % dlog == 0: # 로그 빈도 조절

            cp.savez_compressed(os.path.join(result_dir, 'step',f'step_{step+1}.npz'), positions=cp.asnumpy(positions), velocities=cp.asnumpy(velocities), types=cp.asnumpy(types)) #물성 결과(positions, velocities) 저장

            last_time, log_count, avg_time_per_log = estimated_left(step, steps, log_count, dlog, last_time, avg_time_per_log) # 남은 시간 출력

    # 5. 시뮬레이션의 종료
    t1 = time.time()
    logging.info(f"총 실행시간: {t1 - t0:.2f}초")
    cp.savez_compressed(os.path.join(result_dir, 'step',f'step_{step+1}.npz'), positions=cp.asnumpy(positions), velocities=cp.asnumpy(velocities), types=cp.asnumpy(types))
    logging.info(f"최종 결과 저장 완료: results/run_{count}/step_{step+1}.npz")

    total_collisions = int(collision_count.get()[0])
    logging.info(f"전체 충돌 수 : {total_collisions}")

# 메인 함수
if __name__ == '__main__':
    main(num_side = 20, total_time = 1e-13)