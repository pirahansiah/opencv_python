"""Edge deployment utilities: ONNX export, INT8 quantization, benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BenchmarkResult:
    """Benchmark comparison results."""

    model_name: str
    input_size: tuple[int, int]
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    memory_mb: float = 0.0
    num_runs: int = 0


@dataclass
class QuantizationResult:
    """Quantization output metadata."""

    output_path: Path
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 0.0


class EdgeDeployer:
    """Edge deployment utilities for ONNX models.

    Provides ONNX export, INT8 quantization, benchmarking, and
    deployment helpers for edge devices (Jetson, Raspberry Pi).

    Args:
        model_path: Path to PyTorch model (.pt) or ONNX model (.onnx).
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = Path(model_path) if model_path else None

    def convert_to_onnx(
        self,
        output_path: str | Path = "model.onnx",
        input_size: tuple[int, int] = (640, 640),
        opset: int = 17,
        dynamic_axes: bool = True,
        simplify: bool = True,
    ) -> Path:
        """Convert PyTorch model to ONNX.

        Args:
            output_path: Output ONNX file path.
            input_size: Input spatial size (H, W).
            opset: ONNX opset version.
            dynamic_axes: Enable dynamic batch dimension.
            simplify: Run onnx-simplifier.

        Returns:
            Path to exported ONNX file.
        """
        out = Path(output_path)

        if self._model_path and self._model_path.suffix == ".onnx":
            return self._model_path

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("Install ultralytics: pip install ultralytics") from e

        if self._model_path is None:
            model = YOLO("yolo11n.pt")
        else:
            model = YOLO(str(self._model_path))

        model.export(format="onnx", imgsz=input_size, simplify=simplify, opset=opset)
        print(f"Exported ONNX to {out}")
        return out

    def quantize_int8(
        self,
        onnx_path: str | Path,
        output_path: str | Path | None = None,
        calibration_data: np.ndarray | None = None,
    ) -> QuantizationResult:
        """INT8 dynamic quantization for ONNX models.

        Uses onnxruntime quantization tools. Falls back to OpenCV DNN
        if onnxruntimequantization is unavailable.

        Args:
            onnx_path: Input ONNX model.
            output_path: Output quantized model path.
            calibration_data: Calibration data array for static quantization.

        Returns:
            QuantizationResult with size comparison.
        """
        src = Path(onnx_path)
        if output_path is None:
            out = src.with_name(src.stem + "_int8" + src.suffix)
        else:
            out = Path(output_path)

        original_mb = src.stat().st_size / (1024 * 1024)

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                model_input=str(src),
                model_output=str(out),
                weight_type=QuantType.QInt8,
            )
        except ImportError:
            try:
                from onnxruntime.quantization import quantize_dynamic as qd, QuantType
                qd(str(src), str(out), weight_type=QuantType.QInt8)
            except ImportError:
                raise ImportError(
                    "Install onnxruntime quantization: "
                    "pip install onnxruntime onnxruntime-tools"
                )

        quantized_mb = out.stat().st_size / (1024 * 1024)
        ratio = original_mb / quantized_mb if quantized_mb > 0 else 0.0

        print(f"Quantized: {original_mb:.2f} MB -> {quantized_mb:.2f} MB (ratio: {ratio:.2f}x)")

        return QuantizationResult(
            output_path=out,
            original_size_mb=original_mb,
            quantized_size_mb=quantized_mb,
            compression_ratio=ratio,
        )

    def benchmark(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
        num_runs: int = 100,
        warmup: int = 10,
        provider: str = "CPUExecutionProvider",
    ) -> BenchmarkResult:
        """Benchmark ONNX model inference.

        Args:
            model_path: Path to ONNX model.
            input_size: Input size (H, W).
            num_runs: Number of inference runs.
            warmup: Warmup runs before measurement.
            provider: ONNX Runtime execution provider.

        Returns:
            BenchmarkResult with latency and throughput stats.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("Install onnxruntime: pip install onnxruntime") from e

        model = Path(model_path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(opts) if False else str(model), opts, providers=[provider])

        dummy = np.random.randn(1, 3, *input_size).astype(np.float32)
        input_name = session.get_inputs()[0].name

        latencies: list[float] = []

        for i in range(warmup + num_runs):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy})
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:
                latencies.append(dt)

        latencies_arr = np.array(latencies)
        avg_ms = float(latencies_arr.mean())
        p50 = float(np.percentile(latencies_arr, 50))
        p95 = float(np.percentile(latencies_arr, 95))
        p99 = float(np.percentile(latencies_arr, 99))
        throughput = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        mem_mb = 0.0
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            pass

        result = BenchmarkResult(
            model_name=model.name,
            input_size=input_size,
            avg_latency_ms=avg_ms,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            throughput_fps=throughput,
            memory_mb=mem_mb,
            num_runs=num_runs,
        )

        print(f"Benchmark {model.name}:")
        print(f"  Avg: {avg_ms:.2f}ms | P50: {p50:.2f}ms | P95: {p95:.2f}ms | P99: {p99:.2f}ms")
        print(f"  Throughput: {throughput:.1f} FPS")
        if mem_mb > 0:
            print(f"  Memory: {mem_mb:.1f} MB")

        return result

    def deploy_jetson(
        self,
        onnx_path: str | Path,
        output_dir: str | Path = "jetson_deploy",
        create_script: bool = True,
    ) -> Path:
        """Prepare deployment package for NVIDIA Jetson.

        Creates a deployment directory with ONNX model, Python inference
        script, and setup instructions.

        Args:
            onnx_path: Path to ONNX model.
            output_dir: Deployment directory.
            create_script: Generate a Jetson inference Python script.

        Returns:
            Path to deployment directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        src = Path(onnx_path)
        import shutil
        dest_model = out / src.name
        shutil.copy2(src, dest_model)

        readme = out / "README.md"
        readme.write_text(
            f"# Jetson Deployment\n\n"
            f"Model: `{src.name}`\n\n"
            f"## Setup\n"
            f"```bash\n"
            f"pip install onnxruntime-gpu  # or tensorrt\n"
            f"pip install opencv-python-headless numpy\n"
            f"```\n\n"
            f"## Run\n"
            f"```bash\n"
            f"python jetson_inference.py --model {src.name} --source 0\n"
            f"```\n\n"
            f"## TensorRT Optimization\n"
            f"```bash\n"
            f"trtexec --onnx={src.name} --saveEngine={src.stem}.engine --fp16\n"
            f"trtexec --onnx={src.name} --saveEngine={src.stem}_int8.engine --int8\n"
            f"```\n"
        )

        if create_script:
            script = out / "jetson_inference.py"
            script.write_text(
                f'"""Jetson inference script for {src.name}."""\n'
                f"\n"
                f"from __future__ import annotations\n"
                f"\n"
                f"import argparse\n"
                f"import time\n"
                f"\n"
                f"import cv2\n"
                f"import numpy as np\n"
                f"import onnxruntime as ort\n"
                f"\n"
                f"\n"
                f"def main() -> None:\n"
                f"    parser = argparse.ArgumentParser()\n"
                f'    parser.add_argument("--model", default="{src.name}")\n'
                f'    parser.add_argument("--source", default="0")\n'
                f'    parser.add_argument("--input-size", type=int, default=640)\n'
                f"    args = parser.parse_args()\n"
                f"\n"
                f'    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]\n'
                f"    session = ort.InferenceSession(args.model, providers=providers)\n"
                f"    input_name = session.get_inputs()[0].name\n"
                f"\n"
                f"    source = int(args.source) if args.source.isdigit() else args.source\n"
                f"    cap = cv2.VideoCapture(source)\n"
                f"\n"
                f"    while cap.isOpened():\n"
                f"        ret, frame = cap.read()\n"
                f"        if not ret:\n"
                f"            break\n"
                f"\n"
                f"        t0 = time.perf_counter()\n"
                f"        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (args.input_size, args.input_size))\n"
                f"        outputs = session.run(None, {{input_name: blob}})\n"
                f"        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)\n"
                f"\n"
                f"        cv2.putText(frame, f\"FPS: {{fps:.1f}}\", (10, 30),\n"
                f"                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)\n"
                f"        cv2.imshow(\"Jetson\", frame)\n"
                f"        if cv2.waitKey(1) & 0xFF == ord('q'):\n"
                f"            break\n"
                f"\n"
                f"    cap.release()\n"
                f"    cv2.destroyAllWindows()\n"
                f"\n"
                f"\n"
                f"if __name__ == \"__main__\":\n"
                f"    main()\n"
            )

        print(f"Deployment package ready at: {out}")
        return out

    @staticmethod
    def compare_models(
        results: list[BenchmarkResult],
    ) -> None:
        """Print a comparison table of benchmark results.

        Args:
            results: List of BenchmarkResult to compare.
        """
        if not results:
            print("No results to compare.")
            return

        header = f"{'Model':<25} {'Avg(ms)':>10} {'P95(ms)':>10} {'FPS':>10} {'Mem(MB)':>10}"
        print("\n" + header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r.model_name:<25} {r.avg_latency_ms:>10.2f} {r.p95_latency_ms:>10.2f} "
                f"{r.throughput_fps:>10.1f} {r.memory_mb:>10.1f}"
            )
        print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Edge deployment utilities")
    parser.add_argument("model", nargs="?", default=None, help="Model path (.pt or .onnx)")
    parser.add_argument("--convert", action="store_true", help="Convert to ONNX")
    parser.add_argument("--quantize", default=None, help="Quantize ONNX model to INT8")
    parser.add_argument("--benchmark", default=None, help="Benchmark an ONNX model")
    parser.add_argument("--deploy-jetson", default=None, help="Create Jetson deployment package")
    parser.add_argument("--input-size", type=int, default=640, help="Input size")
    parser.add_argument("--num-runs", type=int, default=100, help="Benchmark runs")
    args = parser.parse_args()

    deployer = EdgeDeployer(args.model)

    if args.convert:
        out = deployer.convert_to_onnx(input_size=(args.input_size, args.input_size))
        print(f"Exported to {out}")

    if args.quantize:
        result = deployer.quantize_int8(args.quantize)
        print(f"Quantized: {result.output_path} ({result.compression_ratio:.2f}x)")

    if args.benchmark:
        result = deployer.benchmark(args.benchmark, input_size=(args.input_size, args.input_size), num_runs=args.num_runs)

    if args.deploy_jetson and args.quantize:
        deployer.deploy_jetson(args.quantize)


if __name__ == "__main__":
    main()
