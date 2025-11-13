//CUDA의 C 코드에 사용되는 부분. 병렬 연산 시행

// 입자에 격자를 배정하는 커널
extern "C" __global__ // GPU 커널함수임을 명시. C언어 style로 호출
void build_linked_list(const double* pos, int* head, int* nxt,
                       const int N, const double cell_size,
                       const int nx, const int ny, const int nz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; // 코어에 할당된 스레드의 ID. 1개의 thread당 1개의 입자 연산을 수행
    if (i >= N) return; // 입자의 개수를 초과하면 연산을 할 필요가 없음 (오류만 발생)

    // 할당된 입자의 위치 정보 수집
    // pos 배열은 pos[0]-[1]-[2]... 이며 각각의 [0]-[1]-[2]는 [x,y,z]의 쌍으로 구성되어 있음.
    // 따라서, 3*i로 입자의 ID를 얻고, 0을 더하면 x, 1을 더하면 y, 2를 더하면 z 좌표를 얻을 수 있음. 
    double x = pos[3*i + 0];
    double y = pos[3*i + 1];
    double z = pos[3*i + 2];

    // 현재 입자가 몇번째 격자에 속해있는지 계산
    int ix = floor(x / cell_size);
    int iy = floor(y / cell_size);
    int iz = floor(z / cell_size);

    // 입자가 불가능한 위치(0번 격자와 마지막 격자의 밖)에 있다면 보정
    if (ix < 0) ix = 0; if (iy < 0) iy = 0; if (iz < 0) iz = 0;
    if (ix >= nx) ix = nx - 1; if (iy >= ny) iy = ny - 1; if (iz >= nz) iz = nz - 1;

    // linked list의 형성.
    // ex) head[3] = 46, nxt[46] = 1, nxt[1] = -1이면 3번 cell 안에 46번 입자와 1번 입자가 존재함을 의미.

    // cell의 3차원 좌표를 1차원 ID로 변환. 변환된 모든 1차원 index는 고유함
    // 변환 과정의 index = ix + iy * nx + iz * (nx * ny)는 nx,ny,nz 혼합진법이며, 10진법의 64를 6*10+4*1로 변환한것과 같음
    int cell = ix + iy * nx + iz * (nx * ny);

    // 이전 격자의 linked list의 head. 원자적 교환을 사용해 old에 head를 저장하고 head값에 i를 넣음.
    // head는 python의 main함수에서 cell의 개수만큼 정의된 1차원 배열.
    int old = atomicExch(&head[cell], i);
    nxt[i] = old; //이전의 head값이 뒤로 밀려나 chain 형성
}

// 입자의 충돌을 정의 및 계산하는 커널
extern "C" __global__ // GPU 커널함수임을 명시. C언어 style로 호출
void collide_particles(double* pos, double* vel, const double* props,
                       int* head, const int* nxt, const int N,
                       const double cell_size, const int nx, const int ny, const int nz)
{
    // 1. 연산의 시행 여부 결정

    int i = blockDim.x * blockIdx.x + threadIdx.x; // 코어에 할당된 스레드의 ID. 1개의 thread당 1개의 입자 연산을 수행
    if (i >= N) return; // 입자의 개수를 초과하면 연산을 할 필요가 없음 (오류만 발생)

    // 2. 입자의 정보 수집

    // 할당된 입자의 위치 정보 수집
    // pos 배열은 pos[0]-[1]-[2]... 이며 각각의 [0]-[1]-[2]는 [x,y,z]의 쌍으로 구성되어 있음.
    // 따라서, 3*i로 입자의 ID를 얻고, 0을 더하면 x, 1을 더하면 y, 2를 더하면 z 좌표를 얻을 수 있음. 
    double xi = pos[3*i+0], yi = pos[3*i+1], zi = pos[3*i+2];

    // 입자의 물성 정보 수집. 이도 마찬가지로 props 배열이 [r,m]의 쌍으로 구성되어 있기 때문임
    double m_i = props[2*i + 1];
    double r_i = props[2*i + 0];

    // 현재 입자가 몇번째 격자에 속해있는지 계산
    int ix = floor(xi / cell_size);
    int iy = floor(yi / cell_size);
    int iz = floor(zi / cell_size);

    // 입자가 불가능한 위치에 존재한다면 보정
    if (ix < 0) ix = 0; if (iy < 0) iy = 0; if (iz < 0) iz = 0;
    if (ix >= nx) ix = nx - 1; if (iy >= ny) iy = ny - 1; if (iz >= nz) iz = nz - 1;

    // 3. 입자가 존재하는 격자와 충돌 연산을 수행할 격자의 판별 (x,y,z로 -1, +1에 해당하는 입자와만 연산. 총 27개)

    // dz : -1, 0, 1의 값을 가지며 반복문 수행. z축 방향으로 현재 격자의 위치가 0이라 할 때 양 옆의 격자를 고려하기 위함.
    for (int dz = -1; dz <= 1; ++dz) {
        int nz_c = iz + dz; //연산을 수행중인 격자의 z축 방향 좌표.
        if (nz_c < 0 || nz_c >= nz) continue; //격자가 공간을 초과한다면 계산할 필요가 없음. 예를 들어 벽에 붙어있는 격자의 경우 벽 밖은 계산 x
        for (int dy = -1; dy <= 1; ++dy) {
            int ny_c = iy + dy;
            if (ny_c < 0 || ny_c >= ny) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx_c = ix + dx;
                if (nx_c < 0 || nx_c >= nx) continue;

                // 4. 판별한 격자와의 충돌 연산을 시행할 격자 내 입자 가져오기

                // 양 옆 격자에 대한 판별 종료 및 격자의 index 구하기
                // cell의 3차원 좌표를 1차원 ID로 변환. 변환된 모든 1차원 index는 고유함
                // 변환 과정의 index = ix + iy * nx + iz * (nx * ny)는 nx,ny,nz 혼합진법이며, 10진법의 64를 6*10+4*1로 변환한것과 같음
                int cell = nx_c + ny_c * nx + nz_c * (nx * ny);

                int j = head[cell]; // 해당 cell에 존재하는 입자의 linked list의 맨 앞 입자를 불러옴

                // link가 끝날 때 까지(nxt[A] = -1이면 A에서 멈춤) 반복 시행. 즉, 해당 격자 내 모든 입자에 대한 연산을 수행
                while (j != -1) {

                    // 5. 입자와의 충돌 여부 판단

                    // 충돌의 중복 연산 방지. A와 B의 충돌 event는 A가 B에 충돌과 B가 A에 충돌 2가지로 count될 수 있음
                    // 번호가 더 작은 쪽만 충돌 계산을 수행하는 것으로 이중계산을 방지함
                    // 또한 i가 i에 대한 충돌 연산을 수행하는 것도 방지
                    if (j > i) {
                        double xj = pos[3*j+0], yj = pos[3*j+1], zj = pos[3*j+2]; //입자 j의 위치 데이터 가져옴

                        // 입자 i와 j 사이의 입체적 거리(정확히는 거리의 제곱) 계산
                        // 충돌을 검사할 때 sqrt를 하지 않고 바로 검사하는 것이 유리하기 때문에 이대로 내버려둠
                        double dxij = xi - xj;
                        double dyij = yi - yj;
                        double dzij = zi - zj;
                        double dist2 = dxij*dxij + dyij*dyij + dzij*dzij;

                        // 충돌의 판별. 두 입자의 반지름의 합보다 입자 사이의 거리가 작으면 충돌으로 판단
                        double r_j = props[2*j + 0]; // 입자 j의 반지름 데이터 가져옴. 2*j로 [r,m]에서 2칸씩 점프하며 가져옴
                        double rsum = r_i + r_j; // 두 입자의 반지름의 합 계산

                        // dist2 > 1e-12 : dist2의 값이 0이 되지 않도록(메모리에 0000...으로 저장되지 않을 최소한의 값) 보장. 오류 방지
                        // dist2 < rsum^2 : 충돌할만큼 거리가 가까운지 판별
                        if (dist2 > 1e-12 && dist2 < rsum*rsum) {

                            // 6. 충돌 계산 수행

                            // 두 입자의 속도 데이터 받아오고, 상대속도 계산하는 연산 수행. 내적 사용
                            double dist = sqrt(dist2);
                            double nxu = dxij / dist;
                            double nyu = dyij / dist;
                            double nzu = dzij / dist;

                            double vix = vel[3*i+0], viy = vel[3*i+1], viz = vel[3*i+2];
                            double vjx = vel[3*j+0], vjy = vel[3*j+1], vjz = vel[3*j+2];
                            double rvx = vix - vjx;
                            double rvy = viy - vjy;
                            double rvz = viz - vjz;

                            double v_rel_n = rvx*nxu + rvy*nyu + rvz*nzu;

                            // 입자가 서로 가까워지고 있을 때. 즉, 상대속도가 음수일 때만 연산을 시행
                            if (v_rel_n < 0.0) {
                                double m_j = props[2*j + 1]; // 입자 j의 질량 데이터 가져오기
                                double e = 1.0; // 완전 탄성 충돌을 가정
                                double impulse_j = -(1.0 + e) * v_rel_n / (1.0/m_i + 1.0/m_j); // 충돌 시의 충격량 계산

                                // 각 성분마다의 충격량 계산
                                double ix_imp = impulse_j * nxu;
                                double iy_imp = impulse_j * nyu;
                                double iz_imp = impulse_j * nzu;

                                // 원자적 수준에서 vel에 구한 dv 더해 계산
                                atomicAdd(&vel[3*i+0], -ix_imp / m_i);
                                atomicAdd(&vel[3*i+1], -iy_imp / m_i);
                                atomicAdd(&vel[3*i+2], -iz_imp / m_i);
                                atomicAdd(&vel[3*j+0], ix_imp / m_j);
                                atomicAdd(&vel[3*j+1], iy_imp / m_j);
                                atomicAdd(&vel[3*j+2], iz_imp / m_j);
                            }
                        }
                    }

                    // linked list에서 다음 입자 불러옴
                    j = nxt[j];
                }
            }
        }
    }
}