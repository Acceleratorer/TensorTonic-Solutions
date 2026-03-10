def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Apply isotonic regression calibration.
    """
    # 1) sort calibration data by predicted probability
    pairs = sorted(zip(cal_probs, cal_labels))
    x = [p for p, _ in pairs]
    y = [float(label) for _, label in pairs]

    # 2) fit isotonic regression with PAV
    # each block: [start_idx, end_idx, sum_y, count, mean]
    blocks = []
    for i, val in enumerate(y):
        blocks.append([i, i, val, 1, val])

        while len(blocks) >= 2 and blocks[-2][4] > blocks[-1][4]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            s = b1[2] + b2[2]
            c = b1[3] + b2[3]
            m = s / c
            blocks.append([b1[0], b2[1], s, c, m])

    calibrated = [0.0] * len(y)
    for start, end, _, _, mean in blocks:
        for i in range(start, end + 1):
            calibrated[i] = mean

    # 3) interpolate new probabilities
    result = []
    n = len(x)

    for q in new_probs:
        if q <= x[0]:
            result.append(calibrated[0])
            continue
        if q >= x[-1]:
            result.append(calibrated[-1])
            continue

        # find interval x[i] <= q <= x[i+1]
        lo, hi = 0, n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if x[mid] <= q:
                lo = mid + 1
            else:
                hi = mid - 1

        i = hi
        x0, x1 = x[i], x[i + 1]
        c0, c1 = calibrated[i], calibrated[i + 1]

        if x1 == x0:
            result.append(c0)
        else:
            t = (q - x0) / (x1 - x0)
            result.append(c0 + t * (c1 - c0))

    return result