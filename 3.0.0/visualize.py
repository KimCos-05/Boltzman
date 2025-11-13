import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
RESULTS_DIR = 'results/run_4/step'
# 시뮬레이션 박스 크기 (boltzman.py와 일치시켜야 함)
BOX_DIMS = [6.7e-8, 6.7e-8, 6.7e-8] 
OUTPUT_FILENAME = 'simulation.gif'
# 애니메이션 속도 및 품질 제어
ANIM_INTERVAL = 30  # 프레임 간 간격 (ms)
ANIM_DPI = 80       # 출력 GIF의 DPI (해상도)

def main():
    print("시각화 스크립트 시작...")

    # 저장된 모든 스텝 파일 찾기
    file_pattern = os.path.join(RESULTS_DIR, 'step_*.npz')
    # 파일 이름의 숫자 부분을 기준으로 정확하게 정렬
    step_files = sorted(glob.glob(file_pattern), key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

    if not step_files:
        print(f"오류: '{RESULTS_DIR}' 디렉토리에서 npz 파일을 찾을 수 없습니다.")
        print("먼저 boltzman.py 시뮬레이션을 실행하여 결과 파일을 생성해야 합니다.")
        return

    print(f"총 {len(step_files)}개의 프레임을 찾았습니다.")

    # 3D 플롯 설정
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 첫 번째 프레임 데이터로 scatter 객체 초기화
    try:
        initial_data = np.load(step_files[0])['positions']
    except KeyError:
        print(f"오류: {step_files[0]} 파일에 'positions' 데이터가 없습니다.")
        print("boltzman.py에서 데이터를 올바르게 저장했는지 확인해주세요.")
        return
        
    scatter = ax.scatter(initial_data[:, 0], initial_data[:, 1], initial_data[:, 2], s=2) # 입자 크기 조절

    def update(frame_num):
        filepath = step_files[frame_num]
        try:
            data = np.load(filepath)
            positions = data['positions']
        except (FileNotFoundError, KeyError) as e:
            print(f"프레임 {frame_num} 로딩 중 오류 발생: {e}")
            # 오류 발생 시 빈 데이터로 처리
            positions = np.array([[], [], []]).T

        # scatter 데이터 업데이트 (매번 새로 그리는 것보다 효율적)
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        
        step_number = os.path.basename(filepath).split('_')[1].split('.')[0]
        ax.set_title(f'Simulation Step: {step_number}')
        
        if (frame_num + 1) % 10 == 0:
            print(f"  - 프레임 {frame_num + 1}/{len(step_files)} 렌더링 중...")
        return scatter,

    # 플롯의 고정 속성 설정
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, BOX_DIMS[0])
    ax.set_ylim(0, BOX_DIMS[1])
    ax.set_zlim(0, BOX_DIMS[2])

    # 애니메이션 생성
    print("애니메이션 생성 중... (프레임 수에 따라 시간이 걸릴 수 있습니다)")
    ani = FuncAnimation(fig, update, frames=len(step_files),
                        interval=ANIM_INTERVAL, blit=True, repeat=False)

    # GIF 파일로 저장
    print(f"'{OUTPUT_FILENAME}' 파일로 저장 중...")
    try:
        ani.save(OUTPUT_FILENAME, writer='pillow', dpi=ANIM_DPI)
        print("-----------------------------------------")
        print(f"성공! '{OUTPUT_FILENAME}' 파일이 생성되었습니다.")
        print("-----------------------------------------")
    except Exception as e:
        print(f"파일 저장 중 오류가 발생했습니다: {e}")
        print("'pillow' 라이터에 문제가 있을 수 있습니다. 'pip install pillow'를 시도해보세요.")

if __name__ == '__main__':
    try:
        import matplotlib
    except ImportError:
        print("오류: 'matplotlib' 라이브러리가 설치되지 않았습니다.")
        print("시각화를 위해 다음 명령어를 실행하여 설치해주세요:")
        print("pip install matplotlib")
    else:
        main()
