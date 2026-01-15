# -*- coding: utf-8 -*-
import argparse, subprocess, sys

def run(cmd):
    print(">>", " ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        sys.exit(ret)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--sigma2", type=float, default=1e-9)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--fc_GHz", type=float, default=3.5)
    args = ap.parse_args()

    # P1 (with exhaustive over U<=2 on small-scale case)
    run(["python","-m","scripts.p1",
         "--K", str(args.K), "--M", str(args.M),
         "--sigma2", str(args.sigma2), "--samples", str(args.samples),
         "--exhaustive_Umax","2",
         "--out_a","fig_P1a.png","--out_b","fig_P1b.png"])
    # P2 (small-scale)
    run(["python","-m","scripts.p2",
         "--K", str(args.K), "--M", str(args.M),
         "--sigma2", str(args.sigma2), "--samples", str(args.samples),
         "--Gamma_dB","-5","0","5","10","15",
         "--out","fig_P2_small.png"])
    # P2 (large+small)
    run(["python","-m","scripts.p2",
         "--K", str(args.K), "--M", str(args.M),
         "--sigma2", str(args.sigma2), "--samples", str(args.samples),
         "--Gamma_dB","-5","0","5","10","15",
         "--include_large","--out","fig_P2_large.png"])
    # P3 (small-scale, M=K for classic WMMSE comparison)
    run(["python","-m","scripts.p3",
         "--K","4","--M","4",
         "--sigma2", str(args.sigma2), "--samples", str(args.samples),
         "--out","fig_P3_small.png"])
    # P3 (large+small)
    run(["python","-m","scripts.p3",
         "--K","4","--M","4",
         "--sigma2", str(args.sigma2), "--samples", str(args.samples),
         "--include_large", "--out","fig_P3_large.png"])

if __name__ == "__main__":
    main()
