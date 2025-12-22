import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def main(run_dir, output_filename, box_dims, skip_particle, anim_interval, anim_dpi):
    """
    시뮬레이션 결과를 바탕으로 애니메이션을 생성하고 저장합니다.

    Args:
        run_dir (str): 'step' 파일들이 포함된 디렉터리 경로.
        output_filename (str): 저장될 mp4 파일의 이름.
        box_dims (list): 시뮬레이션 박스의 [X, Y, Z] 크기.
        skip_particle (int): 시각화할 입자의 샘플링 간격 (N개 중 1개).
        anim_interval (int): 애니메이션 프레임 간 간격 (ms).
        anim_dpi (int): 출력 mp4 파일의 DPI.
    """
    print("시각화 스크립트 시작...")

    # 저장된 모든 스텝 파일 찾기
    file_pattern = os.path.join(run_dir, 'step_*.npz')
    step_files = sorted(glob.glob(file_pattern), key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

    if not step_files:
        print(f"오류: '{run_dir}' 디렉토리에서 npz 파일을 찾을 수 없습니다.")
        print("먼저 시뮬레이션을 실행하여 결과 파일을 생성해야 합니다.")
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
        return
        
    scatter = ax.scatter(initial_data[::skip_particle, 0], initial_data[::skip_particle, 1], initial_data[::skip_particle, 2], s=0.5, alpha=0.8, color='royalblue')

    def update(frame_num):
        filepath = step_files[frame_num]
        try:
            data = np.load(filepath)
            positions = data['positions']
        except (FileNotFoundError, KeyError) as e:
            print(f"프레임 {frame_num} 로딩 중 오류 발생: {e}")
            positions = np.array([[], [], []]).T

        # scatter 데이터 업데이트 (매번 새로 그리는 것보다 효율적)
        scatter._offsets3d = (positions[::skip_particle, 0], positions[::skip_particle, 1], positions[::skip_particle, 2])
        
        step_number = os.path.basename(filepath).split('_')[1].split('.')[0]
        ax.set_title(f'Simulation Step: {step_number}')
        
        if (frame_num + 1) % 10 == 0:
            print(f"  - 프레임 {frame_num + 1}/{len(step_files)} 렌더링 중...")
        return scatter,

    # 플롯의 고정 속성 설정
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, box_dims[0])
    ax.set_ylim(0, box_dims[1])
    ax.set_zlim(0, box_dims[2])

    # 애니메이션 생성
    print("애니메이션 생성 중... (프레임 수에 따라 시간이 걸릴 수 있습니다)")
    ani = FuncAnimation(fig, update, frames=len(step_files),
                        interval=anim_interval, blit=True, repeat=False)

    # mp4 파일로 저장
    print(f"'{output_filename}' 파일로 저장 중...")
    try:
        ani.save(output_filename, writer='ffmpeg', dpi=anim_dpi)
        print("-----------------------------------------")
        print(f"성공! '{output_filename}' 파일이 생성되었습니다.")
        print("-----------------------------------------")
    except Exception as e:
        print(f"파일 저장 중 오류가 발생했습니다: {e}")
        print("'ffmpeg' 라이터에 문제가 있을 수 있습니다. 'pip install ffmpeg-python'를 시도해보세요.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Boltzman Simulation Visualizer")
    parser.add_argument('--run', type=int, required=True, help="시각화할 시뮬레이션의 실행 번호 (예: 12)")
    parser.add_argument('--output', type=str, help="저장할 mp4 파일 이름. 지정하지 않으면 config 파일의 설정을 따릅니다.")
    parser.add_argument('--skip', type=int, help="N개 중 1개의 입자만 시각화합니다. 지정하지 않으면 config 파일의 설정을 따릅니다.")
    
    args = parser.parse_args()

    # --- 설정 로드 ---
    base_dir = "results"
    run_path = os.path.join(base_dir, f'run_{args.run}')
    step_dir = os.path.join(run_path, 'step')
    config_path = os.path.join(os.path.dirname(__file__), 'simulation_config.json')

    if not os.path.exists(run_path):
        print(f"오류: '{run_path}' 디렉터리를 찾을 수 없습니다.")
        exit(1)

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        vis_params = config.get('visualization_params', {})
    except FileNotFoundError:
        print(f"오류: 설정 파일 '{config_path}'를 찾을 수 없습니다.")
        vis_params = {} # 기본값 사용

    # --- 파라미터 결정 (명령줄 인자 > config 파일 > 기본값) ---
    output_file = args.output or vis_params.get('output_filename', f'simulation_run_{args.run}.mp4')
    # 출력 파일이 폴더 경로를 포함하지 않도록 이름만 추출
    output_file = os.path.basename(output_file)
    # 결과를 해당 run 폴더에 저장
    output_path = os.path.join(run_path, output_file)

    skip = args.skip or vis_params.get('skip_particle', 10)
    box_dims = vis_params.get('box_dims', [1e-7, 1e-7, 1e-7]) # config에 없을 경우를 대비한 기본값
    
    # 애니메이션 품질 설정 (하드코딩 또는 config 파일에서 로드)
    anim_interval = 17
    anim_dpi = 150

    try:
        import matplotlib
        main(
            run_dir=step_dir,
            output_filename=output_path,
            box_dims=box_dims,
            skip_particle=skip,
            anim_interval=anim_interval,
            anim_dpi=anim_dpi
        )
    except ImportError:
        print("오류: 'matplotlib' 라이브러리가 설치되지 않았습니다.")
        print("시각화를 위해 다음 명령어를 실행하여 설치해주세요:")
        print("pip install matplotlib")
