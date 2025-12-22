import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def main(run_dir, output_filename, box_dims, skip_particle, anim_interval, anim_dpi):
    print(f"반응 시각화 시작: {run_dir}")

    # 1. 파일 리스트 가져오기
    file_pattern = os.path.join(run_dir, 'step_*.npz')
    step_files = sorted(glob.glob(file_pattern), key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

    if not step_files:
        print("오류: 데이터 파일이 없습니다.")
        return

    print(f"총 {len(step_files)}개의 프레임을 로드합니다.")

    # 2. 그래프 설정
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 축 범위 고정 (박스 크기)
    ax.set_xlim(0, box_dims[0])
    ax.set_ylim(0, box_dims[1])
    ax.set_zlim(0, box_dims[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 색상 매핑 함수
    def get_colors(types):
        # 기본값: 투명한 회색 (배경 입자 Type 0)
        # RGBA: [R, G, B, Alpha]
        colors = np.full((len(types), 4), [0.8, 0.8, 0.8, 0.1]) 
        
        # Type 1 (반응물): 파란색 (진하게)
        mask_reactant = (types == 1)
        colors[mask_reactant] = [0.0, 0.0, 1.0, 0.8]
        
        # Type 2 이상 (생성물): 빨간색 (진하게)
        mask_product = (types >= 2)
        colors[mask_product] = [1.0, 0.0, 0.0, 0.8]
        
        return colors

    # 3. 첫 프레임 로딩 및 초기화
    data = np.load(step_files[0])
    pos = data['positions'][::skip_particle]
    
    # types 데이터가 없으면 모두 0으로 처리
    if 'types' in data:
        types = data['types'][::skip_particle]
    else:
        types = np.zeros(len(pos), dtype=int)

    colors = get_colors(types)

    # 산점도 그리기
    scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
                         c=colors, s=1.0, alpha=None) # alpha는 c에 포함됨

    # 4. 애니메이션 업데이트 함수
    def update(frame_num):
        filepath = step_files[frame_num]
        try:
            data = np.load(filepath)
            positions = data['positions'][::skip_particle]
            
            if 'types' in data:
                current_types = data['types'][::skip_particle]
            else:
                current_types = np.zeros(len(positions), dtype=int)
                
        except Exception as e:
            print(f"프레임 오류: {e}")
            return scatter,

        # 위치 업데이트
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        
        # 색상 업데이트
        new_colors = get_colors(current_types)
        scatter.set_color(new_colors) # 3D scatter 색상 변경

        step_number = os.path.basename(filepath).split('_')[1].split('.')[0]
        ax.set_title(f'Step: {step_number} (Blue: Reactant, Red: Product)')
        
        if frame_num % 10 == 0:
            print(f"렌더링 중... {frame_num}/{len(step_files)}")
            
        return scatter,

    # 5. 애니메이션 저장
    ani = FuncAnimation(fig, update, frames=len(step_files),
                        interval=anim_interval, blit=False) # blit=False 권장 (색상 변경 시)

    print(f"영상 저장 중... ({output_filename})")
    ani.save(output_filename, writer='ffmpeg', dpi=anim_dpi)
    print("저장 완료!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, required=True, help="Run number")
    parser.add_argument('--skip', type=int, default=5, help="Skip N particles")
    args = parser.parse_args()

    # 1. 현재 파일의 위치 파악 (script/analysis/visualize_reaction.py)
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path) # script/analysis

    # 2. results 폴더가 있을만한 곳을 모두 뒤짐 (우선순위: 상위 폴더들)
    possible_base_dirs = [
        os.path.join(current_dir, 'results'),             # script/analysis/results
        os.path.join(current_dir, '..', 'results'),       # script/results
        os.path.join(current_dir, '..', '..', 'results'), # boltzman_V2/results (프로젝트 루트)
        "results"                                         # 현재 실행 위치 기준
    ]

    base_dir = None
    for p in possible_base_dirs:
        # 경로를 절대 경로로 변환해서 체크 (더 안전함)
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            base_dir = abs_p
            print(f"✅ 결과 폴더를 찾았습니다: {base_dir}")
            break

    if base_dir is None:
        print("❌ 오류: 'results' 폴더를 찾을 수 없습니다.")
        print(f"탐색한 경로들: {possible_base_dirs}")
        exit(1)

    # 3. 경로 조립
    run_path = os.path.join(base_dir, f'run_{args.run}')
    step_dir = os.path.join(run_path, 'step')
    config_path = os.path.join(run_path, 'config.json')

    # run 폴더 확인
    if not os.path.exists(run_path):
        print(f"❌ 오류: 해당 실행 번호의 폴더가 없습니다 ({run_path})")
        exit(1)

    # config 파일 확인 및 로드
    if not os.path.exists(config_path):
        print(f"❌ 오류: 설정 파일이 없습니다 ({config_path})")
        exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    sx = config['physical_params']['sx']
    sy = config['physical_params']['sy']
    sz = config['physical_params']['sz']
    
    output_file = os.path.join(run_path, f'reaction_movie_{args.run}.mp4')
    
    # 메인 함수 실행
    main(step_dir, output_file, [sx, sy, sz], args.skip, 30, 150)