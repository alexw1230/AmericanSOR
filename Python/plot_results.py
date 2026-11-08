import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    df = df[df["P"] != "P"]  #Drop repeated header lines if files were merged
    for col in df.columns:
        if col not in ("type", "hash"):
            df[col] = pd.to_numeric(df[col])
    return df


def check_bitwise(df):
    #Every P and rep at a mesh size should give the exact same grid
    lines = []
    all_ok = True
    for N, grp in df.groupby("N"):
        hashes = grp["hash"].unique()
        prices = grp["price"].unique()
        iters = grp["iters"].unique()
        ok = len(hashes) == 1 and len(prices) == 1 and len(iters) == 1
        all_ok = all_ok and ok
        lines.append(f"N = {N:5d}: {'PASS' if ok else 'FAIL'}  "
                     f"hash {hashes[0]}  price {prices[0]:.10f}  "
                     f"iterations {iters[0]}")
    return all_ok, lines


def summarize(df):
    #Median over reps for each (N, P)
    med = (df.groupby(["N", "P"])
             .agg(px=("px", "first"), py=("py", "first"),
                  omega=("omega", "first"), iters=("iters", "first"),
                  price=("price", "first"),
                  t_total=("t_total", "median"),
                  t_per_iter=("t_per_iter", "median"),
                  t_comm=("t_comm_avg", "median"),
                  t_comp_max=("t_comp_max", "median"),
                  t_comp_avg=("t_comp_avg", "median"))
             .reset_index())

    #Speedup against the P = 1 run of the same code
    t1 = med[med["P"] == 1].set_index("N")["t_total"]
    med["speedup"] = med.apply(lambda r: t1[r["N"]] / r["t_total"], axis=1)
    med["efficiency"] = med["speedup"] / med["P"]
    med["comm_frac"] = med["t_comm"] / med["t_total"]
    med["imbalance"] = med["t_comp_max"] / med["t_comp_avg"]
    return med


def line_plot(med, col, ylabel, title, path, ideal=None, log_y=False, skip_p1=False):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = med[med["P"] > 1] if skip_p1 else med
    Ps = sorted(data["P"].unique())

    for N, grp in data.groupby("N"):
        ax.plot(grp["P"], grp[col], "o-", label=f"{N}x{N}")
    if ideal == "linear":
        ax.plot(Ps, Ps, "k--", linewidth=1, label="ideal")
    elif ideal == "one":
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1)

    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log", base=2)
    ax.set_xticks(Ps)
    ax.set_xticklabels(Ps)
    ax.set_xlabel("cores P")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(med, check_lines, euro, path):
    rows = ["| N | P | grid | iterations | T (s) | T/iter (s) | speedup "
            "| efficiency | comm % | imbalance |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in med.itertuples(index=False):
        rows.append(f"| {r.N} | {r.P} | {r.px}x{r.py} | {r.iters} "
                    f"| {r.t_total:.4f} | {r.t_per_iter:.2e} "
                    f"| {r.speedup:.2f} | {r.efficiency:.2f} "
                    f"| {100 * r.comm_frac:.1f} | {r.imbalance:.3f} |")

    text = "## Timing (median of reps)\n\n" + "\n".join(rows) + "\n\n"
    text += "## Bitwise check across P\n\n```\n" + "\n".join(check_lines) + "\n```\n\n"

    if euro is not None and len(euro) > 0:
        text += ("## European check vs. closed form\n\n"
                 "| N | M | numerical | closed form | abs error |\n"
                 "|---|---|---|---|---|\n")
        for r in euro.itertuples(index=False):
            text += (f"| {r.N} | {r.M} | {r.price:.6f} "
                     f"| {r.euro_exact:.6f} | {r.abs_err:.2e} |\n")

    with open(path, "w") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("res", help="results/<jobid> folder from Bash/run.sh")
    parser.add_argument("--img", default="Images", help="where to save plots")
    args = parser.parse_args()

    df = load(os.path.join(args.res, "american.csv"))
    euro_path = os.path.join(args.res, "european.csv")
    euro = load(euro_path) if os.path.exists(euro_path) else None

    ok, lines = check_bitwise(df)
    print("\n".join(lines))
    print("bitwise check:", "PASS" if ok else "FAIL")

    med = summarize(df)
    os.makedirs(args.img, exist_ok=True)
    line_plot(med, "speedup", "speedup T(1) / T(P)", "PSOR speedup, 2-asset American put",
              os.path.join(args.img, "speedup.png"), ideal="linear", log_y=True)
    line_plot(med, "efficiency", "efficiency S(P) / P", "Parallel efficiency",
              os.path.join(args.img, "efficiency.png"), ideal="one")

    #Comm share only makes sense with more than one rank
    med["comm_pct"] = 100 * med["comm_frac"]
    line_plot(med, "comm_pct", "communication time (% of total)",
              "Halo exchange + Allreduce share",
              os.path.join(args.img, "comm_fraction.png"), skip_p1=True)

    write_summary(med, lines, euro, os.path.join(args.res, "summary.md"))
    print(f"wrote plots to {args.img}/ and {args.res}/summary.md")


if __name__ == "__main__":
    main()
