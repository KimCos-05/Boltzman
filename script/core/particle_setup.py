import numpy as np
import cupy as cp
import math

# Particle 생성 및 배치 함수.
# Particle의 정보 : 위치(x,y,z), 속도(x,y,z), 물성(r,m), 종류
def setParticle(sx, sy, sz, num, r, m): 
    N = num**3 
    
    # 입자의 배열 간격을 공간의 크기 / 개수로 설정
    dx = sx / num
    dy = sy / num
    dz = sz / num

    # 격자 인덱스 생성 (0, 1, 2 ... num-1)
    gx, gy, gz = np.meshgrid(np.arange(num), np.arange(num), np.arange(num), indexing='ij') 
    gx, gy, gz = gx.ravel(), gy.ravel(), gz.ravel() 

    # 위치 계산 : 격자 인덱스 * 간격 + 반칸 오프셋
    # dx/2.0을 더해주는 이유는 0번 입자가 벽에 딱 붙지 않고, 자기 칸의 중앙에 오게 하기 위함.
    
    # + 약간의 무작위성(Jitter) 추가. 기체 상태에서는 결정성이 없으므로
    rng = np.random.default_rng()
    jitter_scale = 0.2 # 자기 구역의 20% 정도만 흔들림 (충돌 방지하며 랜덤화)
    
    jitter_x = (rng.random(N) - 0.5) * (dx * jitter_scale)
    jitter_y = (rng.random(N) - 0.5) * (dy * jitter_scale)
    jitter_z = (rng.random(N) - 0.5) * (dz * jitter_scale)

    pos_cpu = np.stack([
        gx * dx + dx/2.0 + jitter_x,
        gy * dy + dy/2.0 + jitter_y,
        gz * dz + dz/2.0 + jitter_z
    ], axis=1).astype(np.float64)
    
    # 속도와 물성 배열 생성(값은 0)
    vel_cpu = np.zeros((N, 3), dtype=np.float64)
    prop_cpu = np.zeros((N, 2), dtype=np.float64)
    
    # 입자의 종류 배열 생성
    type_cpu = np.zeros((N,), dtype=np.int32)
    
    rng = np.random.default_rng()
    rand_vals = rng.random(N)

    # 입자 종류 배정
    type_cpu[rand_vals < 0.5] = 1 # 입자 A
    type_cpu[rand_vals >= 0.5] = 2 # 입자 B

    # 물성치 r,m 넣음
    prop_cpu[:, 0] = r
    prop_cpu[:, 1] = m
    
    return pos_cpu, vel_cpu, prop_cpu, type_cpu

# particle의 초기 속도 설정 함수
def setParticleSpeed(velocities, properties, T, m):

    k_B = 1.380649e-23

    v = math.sqrt((k_B * 3 * T) / m)

    # velocities 배열에 v를 매개로 한 무작위 값 넣음. velocities의 dtype을 가지며, velocities와 동일한 형태를 가지는 새 배열 생성.
    # 2를 곱하고 1을 빼는 것으로 값을 [0, 1]의 값을 [-1, 1]의 배열들로 변경. 이후 상수 v 곱해 스케일을 크게 만듬.
    # velocities[:] : 전체 배열을 의미. 새로 생성한 배열을 덮어쓰기함
    rng = cp.random.default_rng()
    velocities[:] = (rng.random(velocities.shape, dtype=velocities.dtype) * 2 - 1) * v # 속도 난수 형성. 

    # 계의 전체 운동량 계산. Total 운동량을 0으로 만들기 위한 보정작업. 이론상으로는 무작위이므로 전체 속도의 합이 0일 것이지만, 큰 수의 법칙이 적용되지 않는 영역에서는 보정이 필요.
    # 질량의 [N*1] 행렬과 속도의 [N*3]의 브로드캐스팅 이후 각 방향 (x,y,z)로의 합을 더해 각 성분의 운동량 값을 각각 계산.
    mass = properties[:, 1].reshape(-1, 1) # 입자의 모든 질량값을 가져옴. velocities와 계산하기 위해 [N*1] 벡터로 저장. (브로드캐스팅 연산을 수행하기 위함)
    total_mom = (velocities * mass).sum(axis=0, keepdims=True) # 각 입자의 속도 벡터에 질량을 모두 곱해 운동량 계산. .sum(axis=0, keepdims=True)를 통해 열 방향(각 차원의 운동량 방향)의 값을 모두 더하고 [1*3] 행렬의 형태로 만듬
    total_mass = mass.sum()

    # !브로드캐스팅 : [N*3] 행렬과 [N*1] 행렬이 있을 때, [N*1] 행렬을 가로로 3칸 복제해 [N*3] 행렬로 만들고, 같은 위치의 두 행렬의 원소를 서로 곱해주는 연산

    # 전체 운동량의 합계가 0보다 크다면 편차만큼을 전체 속도에서 빼줌
    if total_mass > 0:
        velocities -= total_mom / total_mass

    return velocities

if __name__ == '__main__':

    pos_cpu, vel_cpu, prop_cpu = setParticle(6.7e-08, 6.7e-08, 6.7e-08, 20, 1.88e-10, 6.63e-26)
    vel_cpu = setParticleSpeed(vel_cpu, prop_cpu, 298.15, 6.63e-26)

    print(vel_cpu)
