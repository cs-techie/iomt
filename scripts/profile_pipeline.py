import sys
import time
import statistics
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import BioTwin6GPipeline


def run_once(pipeline, message):
    start = time.perf_counter()
    result = pipeline.process(message, application="critical")
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def main():
    print("=" * 60)
    print("BIO-TWIN 6G END-TO-END LATENCY PROFILING")
    print("=" * 60)

    message = {
        "heart_rate": 72,
        "glucose": 118,
        "spo2": 98,
        "packet_rate": 12,
        "auth_failures": 0,
        "jitter_ms": 3,
        "neural_score": 0.08
    }

    print("\nInitializing pipeline...")
    pipeline = BioTwin6GPipeline()

    # ---------------------------------------------------------
    # Warm-up run
    # ---------------------------------------------------------
    print("Running warm-up...")
    warm_result, warm_latency = run_once(pipeline, message)

    print(f"Warm-up latency : {warm_latency:.3f} ms")
    print(f"Warm-up decision: {warm_result.ids_decision}")

    # ---------------------------------------------------------
    # Repeated measurements
    # ---------------------------------------------------------
    RUNS = 30
    latencies = []

    print(f"\nRunning {RUNS} measured iterations...")

    for i in range(RUNS):
        result, latency = run_once(pipeline, message)
        latencies.append(latency)

        print(
            f"Run {i + 1:02d}: "
            f"{latency:10.3f} ms | "
            f"decision={result.ids_decision} | "
            f"authenticated={result.authenticated}"
        )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    mean_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    if len(latencies) > 1:
        std_latency = statistics.stdev(latencies)
    else:
        std_latency = 0.0

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("LATENCY STATISTICS")
    print("=" * 60)

    print(f"Runs                  : {RUNS}")
    print(f"Mean latency          : {mean_latency:.3f} ms")
    print(f"Median latency        : {median_latency:.3f} ms")
    print(f"Minimum latency       : {min_latency:.3f} ms")
    print(f"Maximum latency       : {max_latency:.3f} ms")
    print(f"Standard deviation    : {std_latency:.3f} ms")

    # Approximate throughput based on mean end-to-end latency
    if mean_latency > 0:
        throughput = 1000.0 / mean_latency
    else:
        throughput = 0.0

    print(f"Approx. transactions/s: {throughput:.3f}")

    print("\n" + "=" * 60)
    print("PIPELINE OUTPUT")
    print("=" * 60)

    print(f"Authenticated         : {result.authenticated}")
    print(f"IDS decision          : {result.ids_decision}")
    print(f"Slice                 : {result.slice_name}")
    print(f"Slice accepted        : {result.slice_ok}")
    print(f"Phase-2 bytes         : {result.phase2_bytes}")
    print(f"Final encrypted bytes : {result.final_bytes}")
    print(f"Chaos entropy         : {result.entropy:.6f} bits/byte")

    print("\n" + "=" * 60)
    print("CRITICAL IoMT BUDGET")
    print("=" * 60)

    from src.config import CONFIG

    budget = CONFIG.critical_iomt_budget_ms

    print(f"Configured budget     : {budget:.3f} ms")
    print(f"Mean latency          : {mean_latency:.3f} ms")

    if mean_latency <= budget:
        print("Budget status         : PASS")
    else:
        print("Budget status         : FAIL")

    print("=" * 60)


if __name__ == "__main__":
    main()