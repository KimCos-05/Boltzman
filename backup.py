def setParticle(sx, sy, sz, num, r, m): #Space 안에 Particle이 위치해야 하므로 공간의 크기 가져옴.
    N = num**3 
    spacing = (r + r*0.1) * 2.0 #입자의 배치간격 결정. 지름 d의 1.1배 간격으로 배열. (두자마자 충돌 방지)

    # for 연산이 아닌 행렬 연산을 수행하기 위해 격자를 만들어 계산
    gx, gy, gz = np.meshgrid(np.arange(num), np.arange(num), np.arange(num), indexing='ij') #3차원 격자 형성.
    gx, gy, gz = gx.ravel(), gy.ravel(), gz.ravel() #3차원 격자를 1차원 배열로 변환

    # 시작지점 : 공간의 중심 좌표에서 배수 * 간격의 절반만큼 (공간의 중심에 배열하기 위함)
    x_start = (sx / 2.0) - ((num - 1) * spacing) / 2.0
    y_start = (sy / 2.0) - ((num - 1) * spacing) / 2.0
    z_start = (sz / 2.0) - ((num - 1) * spacing) / 2.0
    
    # 3차원 위치 배열 만들어 계산
    pos_cpu = np.stack([
        x_start + gx * spacing,
        y_start + gy * spacing,
        z_start + gz * spacing
    ], axis=1).astype(np.float64)