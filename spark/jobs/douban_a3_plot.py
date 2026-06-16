import argparse
import csv
from pathlib import Path


def write_svg(output_path: Path, labels, values):
    width = 820
    height = 500
    margin_left = 110
    margin_bottom = 90
    chart_top = 70
    chart_height = 310
    chart_width = 620
    max_value = max(values)
    bar_width = 120
    gap = 70
    colors = ["#2563eb", "#16a34a", "#f97316"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="410" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">A3 Performance Comparison</text>',
        f'<line x1="{margin_left}" y1="{chart_top + chart_height}" x2="{margin_left + chart_width}" y2="{chart_top + chart_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{chart_top}" x2="{margin_left}" y2="{chart_top + chart_height}" stroke="#111827" stroke-width="1"/>',
    ]

    for i in range(6):
        y = chart_top + chart_height - chart_height * i / 5
        tick = max_value * i / 5
        parts.append(f'<line x1="{margin_left - 6}" y1="{y:.1f}" x2="{margin_left + chart_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>')

    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + 70 + index * (bar_width + gap)
        bar_height = chart_height * value / max_value if max_value else 0
        y = chart_top + chart_height - bar_height
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[index]}"/>')
        parts.append(f'<text x="{x + bar_width / 2}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700">{value:.3f}s</text>')
        parts.append(f'<text x="{x + bar_width / 2}" y="{chart_top + chart_height + 30}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>')

    parts.append('<text x="26" y="235" transform="rotate(-90 26 235)" text-anchor="middle" font-family="Arial" font-size="14">Elapsed seconds</text>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create A3 performance comparison chart.")
    parser.add_argument("--pandas", type=float, required=True, help="Pandas elapsed seconds")
    parser.add_argument("--spark1", type=float, required=True, help="PySpark elapsed seconds with one executor")
    parser.add_argument("--spark2", type=float, required=True, help="PySpark elapsed seconds with two executors")
    parser.add_argument("--output", default="spark/a3_performance_comparison.svg", help="Output SVG path")
    args = parser.parse_args()

    labels = ["Pandas", "Spark 1 executor", "Spark 2 executors"]
    values = [args.pandas, args.spark1, args.spark2]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_svg(output_path, labels, values)

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "elapsed_seconds"])
        writer.writerows(zip(labels, values))

    speedup = args.spark1 / args.spark2 if args.spark2 else 0
    print("A3 chart:", output_path)
    print("A3 data:", csv_path)
    print(f"SPARK_2_EXECUTOR_SPEEDUP_OVER_1_EXECUTOR: {speedup:.4f}")


if __name__ == "__main__":
    main()
