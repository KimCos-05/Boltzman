# CUDA 기반 3D 분자 동역학 시뮬레이션

본 프로젝트는 **GPU 가속(CUDA)**을 활용하여 이상 기체 입자들의 탄성 충돌을 시뮬레이션하고, 시스템이 **맥스웰-볼츠만 분포(Maxwell-Boltzmann Distribution)**에 도달하는 과정을 검증하는 분자 동역학(MD) 시뮬레이션입니다.

본 시뮬레이션은 이상 기체(Ideal Gas)의 Hard Sphere 모델에 기반합니다.

---

## 주요 기능 (Key Features)

* **GPU 가속 연산** : Python(`cupy`)와 C++(`CUDA Kernel`) 하이브리드 구조로 대규모 입자 연산을 실시간 처리합니다.
* **공간 분할 알고리즘 (Spatial Partitioning)** : Cell Linked-List 방식을 구현하여 충돌 감지 복잡도를 $O(N^2)$에서 $O(N)$으로 최적화했습니다.
* **물리적 정합성 검증** : 에너지 보존 법칙 및 운동량 보존 법칙 준수를 검증합니다.
* **데이터 시각화** : 3D 입자 거동 애니메이션 및 속력 분포 히스토그램 분석 도구를 포함합니다.

---

## 벤치마크 및 성능 (Benchmark & Perfomance)

실제 하드웨어 환경에서 수행한 성능 테스트 결과입니다.

### 테스트 환경 (Test Environment)
| 항목 | 사양 |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 3050 (8GB) |
| **입자 수 ($N$)** | 50,653개 (`num_side` = 37) |
| **공간 크기** | $(6.7 \times 10^{-8})^3$ $m^3$ |
| **시간 간격 ($dt$)** | $5.0 \times 10^{-16}$ $s$ |

### 성능 측정 결과 (Execution Time)

| 진행 단계 | 구간 (Step) | 소요 시간 (평균) | 상태 설명 |
| :--- | :--- | :--- | :--- |
| **초기 (Initial)** | 100 ~ 1,000 | **4.06초** | 고밀도 격자 배치 (Dense Lattice) |
| **후기 (Equilibrium)** | 300,000 ~ 600,000 | **0.24초** | 균일 분포 평형 상태 (Homogeneous) |

![alt text](step_duration_plot.png)

> **Note**: 소요 시간은 데이터 저장 간격(`dlog = 100step`) 당 측정된 시간입니다.

> **Note**: 상세 분석은 ``시뮬레이션 특성 및 주의사항``을 참고해 주세요.

### 시뮬레이션 결과 (Simulation result)

![alt text](momentum_distribution_inital.png)
![alt text](momentum_distribution_final.png)

<video controls src="simulation_12_2-1.mp4" title="Title"></video>

---

## 설치 및 환경 설정 (Installation)

이 프로젝트를 실행하기 위해서는 **NVIDIA GPU**와 **CUDA Toolkit**이 필요합니다.

## 사용법 (Usage)

1. **파라미터 설정 (``simulation_config.json``)**

입자 수, 온도, 시간 간격 등은 JSON 파일에서 수정할 수 있습니다.
```JSON
{
    "num_side": 37,                // 한 변의 입자 수, 총 입자 수 = N^3
    "dt": 5e-16,                   // 시간 간격 (초)
    "total_time": 3e-10,           // 총 시뮬레이션 시간
    "dlog": 10000,                 // 데이터 저장 간격(step)
    "file_base_path": "results",   // 결과 저장 파일 이름
    "physical_params": {
        "sx": 6.7e-08,             // 시뮬레이션 공간(x)
        "sy": 6.7e-08,             // 시뮬레이션 공간(y)
        "sz": 6.7e-08,             // 시뮬레이션 공간(z)
        "r": 1.88e-10,             // 입자 반지름
        "m": 6.63e-26,             // 입자 질량
        "T": 298.15                // 온도
    }
}
```

2. **시뮬레이션 실행**

기본 설정으로 시뮬레이션을 시작합니다. 자동으로 `results/run_N` 폴더가 생성되며 실행 번호가 매겨집니다.

```bash
python main.py
```

3. **시뮬레이션 종료**

시뮬레이션이 종료되면 다음 파일이 저장됨을 확인할 수 있습니다.

```bash
simulation.log : results\run_N\                                 // 시뮬레이션 로그
config.json : results\run_N                                     // 시뮬레이션 초기 설정
step_N.npz : results\run_N\step                                 // dlog 간격으로 저장된 해당 step의 위치, 속도, 물성 데이터
momentum_distribution_initial.png : results\run_N\distribution   // 초기 상태의 운동량 분포
momentum_distribution_final.png : results\run_N\distribution    // 마지막 상태의 운동량 분포
```

4. **시뮬레이션 결과 시각화**

``main.py`` 실행 외에도 각각의 파일을 직접 실행하는 것으로 다음의 작업을 수행할 수 있습니다.

* ``plot_momentum.py``: 특정 step에서의 운동량 분포 시각화.
* ``visualize.py`` : 처음 상태부터 최종 상태까지의 npz 데이터를 활용한 mp4 미디어 생성

---

## ``visualize.py`` 사용법

저장된 시뮬레이션 결과를 3D 애니메이션 비디오로 생성합니다.

#### 기본 명령어

```shell
python visualize.py --run <실행_번호> [옵션]
```

#### 옵션 (인자)

| 인자 (`Argument`) | 설명                                                                                               | 예시                               |
| ----------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `--run` (필수)    | 시각화할 시뮬레이션의 실행 번호(`N`)를 지정합니다.                                                 | `--run 12`                         |
| `--output`        | 저장될 비디오 파일의 이름을 지정합니다. 지정하지 않으면 `simulation_run_N.mp4`로 자동 설정됩니다. | `--output my_animation.mp4`        |
| `--skip`          | N개 중 1개의 입자만 시각화하여 렌더링 속도를 높입니다. 값이 작을수록 많은 입자가 보입니다.         | `--skip 5`                         |

#### 사용 예시

```shell
# 12번 실행 결과를 기본 설정으로 시각화
python visualize.py --run 12

# 15번 실행 결과를 'run15.mp4'라는 이름으로, 20개 중 1개 입자만 그리며 시각화
python visualize.py --run 15 --output run15.mp4 --skip 20
```

---

## 물리 모델 및 알고리즘 (Physics & Algorithms)

### 초기화

1. **격자 기반 배치 (Lattice Placement)**

* 시뮬레이션 공간(Box) 내에 N개의 입자를 균일한 간격으로 배치하여 초기 곂침(Overlapping)을 방지합니다.

2. **무작위 속도 부여 (Random Velocity Assignment)**

* 설정된 온도 T에 상응하는 운동 에너지를 갖도록 각 입자에 무작위 속도 벡터($v_x$, $v_y$, $v_z$)를 부여합니다.

3. **운동량 제로 보정 (Zero-Momentum Correction)**

* 초기 무작위 생성으로 인해 발생하는 전체 시스템의 미세한 표류(Drift)를 방지하기 위해, 계의 총 운동량을 계산하여 0으로 보정합니다.

$$v_i = v_i - \frac{\sum {v}_{total}}{N}$$

### 충돌 처리

1. **벽 - 입자 충돌**
* 입자의 위치가 설정한 경계 밖에 있는지 확인하고, 경계 밖에 존재한다면 위치를 보정하고 속도의 방향을 바꾸는 방식으로 구현하였습니다.

2. **입자 - 입자 충돌**
* 두 입자 사이의 거리와 반지름의 합을 비교하고, 반지름의 합보다 작다면 곂친 길이만큼 위치를 보정하고 속도의 방향을 바꾸는 방식으로 구현하였습니다.

### 알고리즘

* Cell Linked-List : 전체 공간을 입자 크기 기반의 격자(Cell)로 나누고, 인접한 27개 격자 내의 입자만 검사하여 연산 속도를 극대화했습니다.
* Atomic Operation : GPU 병렬 연산 시 발생하는 데이터 경쟁을 방지하기 위해 atomicAdd, atomicExch를 사용하였습니다.

---

## 시뮬레이션 특성 및 주의사항

### 1. 시뮬레이션 초기 진행 속도

``물리 모델 및 알고리즘``의 ``격자 기반 배치`` 내용에 의해 시뮬레이션 초반의 step당 시행 시간이 느린 경향을 가지고 있습니다.

* 시뮬레이션 초기에는 입자 사이의 거리가 작습니다. (``particle_setup``의 초기 상태 기준으로 두 입자 사이의 빈 공간이 0.1d가 되도록 배치)
* 한 격자의 간격이 입자 지름의 7배이므로(``particle_setup``의 초기 상태 기준) 시뮬레이션 초기에는 각 격자당 최대 257개의 입자가 배치될 수 있습니다.

$$\text{Count}_{1D} = \frac{\text{Cell Size}}{\text{Particle Spacing}} = \frac{7d}{1.1d} \approx 6.36$$

* 즉, 한 줄에 약 6.36개의 입자가 들어가며 3차원 부피로 계산하면 :

$$\text{Count}_{3D} = (6.36)^3 \approx 257.5$$

* 따라서, 초기 밀집 상태에서 격자 당 입자 수는 약 257개입니다.

* 충분한 시간이 지나 계가 열역학적 평형에 도달하면 입자들은 공간 전체에 균일하게 분포됩니다. 이때 하나의 격자(Cell) 내에 존재하는 평균 입자 수는 다음과 같습니다.

$$\langle N_{cell} \rangle = \frac{N_{total}}{N_{grid}} \approx \rho \, l_{cell}^3$$

* 실제 시뮬레이션 데이터를 사용하여 계산하면 다음과 같습니다.

1. 전체 입자 수 ($N$):num_side = 37 이므로, $37^3 = 50,653$개

2. 입자 지름 ($d$):반지름 $r = 1.88 \times 10^{-10}$ m 이므로, $d = 3.76 \times 10^{-10}$ m

3. 격자 크기 ($L_{cell}$):cell_size = $7d$ 이므로, 약 $2.632 \times 10^{-9}$ m

4. 공간 크기 ($L_{box}$):$sx = 6.7 \times 10^{-8}$ m

5. 격자 당 평균 입자 수 계산한 축당 격자의 개수 ($n_x$):$$n_x = \text{int}\left(\frac{L_{box}}{L_{cell}}\right) = \text{int}\left(\frac{6.7 \times 10^{-8}}{2.632 \times 10^{-9}}\right) \approx \text{int}(25.45) = 25 \text{개}$$ 따라서 공간은 $25 \times 25 \times 25$ 개의 격자로 나뉩니다

6. 전체 격자의 개수 ($N_{cells}$):$$N_{cells} = 25^3 = 15,625 \text{개}$$

7. 격자 당 평균 입자 수 ($\langle N_{cell} \rangle$):$$\langle N_{cell} \rangle = \frac{N_{total}}{N_{cells}} = \frac{50,653}{15,625} \approx \mathbf{3.24} \text{개}$$

### 2. 시뮬레이션 간격에 따른 결과의 왜곡

``시뮬레이션 간격 (dt)``이 증가하면 시뮬레이션 결과의 왜곡도가 커질 수 있습니다.

* 본 시뮬레이션은 이상 기체(Ideal Gas)의 Hard Sphere 모델에 기반하고 있습니다.
* 입자의 평균 자유 행로보다 각 step당 이동하는 거리가 길면 두 입자가 곂치거나, 지나쳐 충돌수가 왜곡될 수 있습니다.
* 입자의 평균 자유 행로는 다음과 같이 계산할 수 있습니다.

$$\lambda = \frac{1}{4 \sqrt{2} \pi r^2 n_V}$$

변수 설명

* $\lambda$ (Lambda): 평균 자유 행로 (단위: $m$)
* $n_V$ (Number Density): 수밀도, 즉 단위 부피당 입자의 개수 ($\frac{N}{V}$) (단위: $m^{-3}$)
* $d$ (Diameter): 입자의 지름 (단위: $m$). (2r)
* $\pi d^2$ (Collision Cross-section, $\sigma$): 충돌 단면적. 입자가 충돌을 일으키는 유효 면적입니다.
* $\sqrt{2}$: 입자들이 모두 움직이고 있기 때문에 들어가는 보정값입니다.만약 다른 입자들이 정지해 있고 하나만 움직인다면 $\sqrt{2}$가 빠지지만, 실제로는 상대 속도($\bar{v}_{rel} = \sqrt{2}\bar{v}$)를 고려해야 하므로 분모에 $\sqrt{2}$가 붙습니다.

### 3. 부동소수점 계산에 의한 에너지의 변화

``부동소수점 연산``에 의해 에너지가 미세하게 변화할 수 있습니다.

* 통계역학적인 계산은 계의 에너지를 무한소수로 처리하지만, 컴퓨터는 이산적으로 처리함에 따라 시뮬레이션 결과 계의 온도가 소수점 첫째자리에 해당하는 변동이 생길 수 있습니다.

---

## Acknowledgments

* 본 프로젝트의 일부 코드 최적화 및 문서화 과정에서 **Gemini 2.5 Pro** 및 **Gemini CLI**의 도움을 받았습니다.
