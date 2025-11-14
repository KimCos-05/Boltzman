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