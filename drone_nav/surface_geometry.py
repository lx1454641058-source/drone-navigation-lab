"""来源：本项目原创。共享的三参数地面平面拟合，无第三方数值依赖。"""

from math import isfinite


def fit_plane(samples):
    """对 (x,y,z) 点拟合 z=ax+by+c；退化或非法点集拒绝求解。"""
    points = list(samples)
    if len(points)<3 or any(len(p)!=3 or any(type(v) not in (int,float) or not isfinite(v) for v in p) for p in points):
        raise ValueError('at least three finite 3D points required')
    basis = [(x,y,1.0) for x,y,_ in points]
    matrix = [[sum(row[i]*row[j] for row in basis) for j in range(3)] for i in range(3)]
    target = [sum((x,y,1.0)[i]*z for x,y,z in points) for i in range(3)]
    aug = [matrix[i]+[target[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col,3),key=lambda row:abs(aug[row][col]))
        aug[col],aug[pivot] = aug[pivot],aug[col]
        if abs(aug[col][col])<1e-10:
            raise ValueError('surface points do not span a plane')
        factor = aug[col][col]
        aug[col] = [v/factor for v in aug[col]]
        for row in range(3):
            if row!=col:
                factor = aug[row][col]
                aug[row] = [a-factor*b for a,b in zip(aug[row],aug[col])]
    return tuple(row[3] for row in aug)
