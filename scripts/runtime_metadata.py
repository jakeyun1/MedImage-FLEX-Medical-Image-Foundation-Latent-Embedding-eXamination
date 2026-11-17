"""Rootless hardware, device, and runtime metadata collection."""

import math
import os
import platform
import resource
import subprocess


HARDWARE_SCHEMA_VERSION = 1
EXECUTION_SCHEMA_VERSION = 1


def _read_first_value(path, key=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if key is None:
                    return line.strip()
                name, separator, value = line.partition(":")
                if separator and name.strip() == key:
                    return value.strip()
    except (OSError, UnicodeError):
        return None
    return None


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _cpu_model():
    model = _read_first_value("/proc/cpuinfo", "model name")
    return model or platform.processor() or platform.machine() or None


def _available_cpu_count():
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = None
    if affinity:
        return len(affinity), "process_affinity"
    return _positive_int(os.cpu_count()), "os_cpu_count"


def _physical_memory_bytes():
    mem_total_kib = _read_first_value("/proc/meminfo", "MemTotal")
    if mem_total_kib:
        value = _positive_int(mem_total_kib.split()[0])
        if value is not None:
            return value * 1024
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return _positive_int(page_size * page_count)


def _cgroup_memory_limit_bytes():
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        raw = _read_first_value(path)
        if raw and raw != "max":
            value = _positive_int(raw)
            if value is not None:
                return value
    return None


def _available_memory_bytes():
    physical = _physical_memory_bytes()
    cgroup = _cgroup_memory_limit_bytes()
    candidates = [value for value in (physical, cgroup) if value is not None]
    if not candidates:
        return None, "unavailable"
    available = min(candidates)
    source = "cgroup_limit" if cgroup is not None and available == cgroup else "host_visible"
    return available, source


def _nvidia_inventory():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except FileNotFoundError:
        return {"status": "unavailable", "reason": "executable_not_found", "gpus": []}
    except subprocess.TimeoutExpired:
        return {"status": "unavailable", "reason": "timeout", "gpus": []}
    except (OSError, subprocess.CalledProcessError):
        return {"status": "unavailable", "reason": "query_failed", "gpus": []}

    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not line.strip():
            continue
        if len(fields) != 5:
            return {"status": "unavailable", "reason": "malformed_output", "gpus": []}
        memory_mib = _positive_int(fields[3])
        devices.append({
            "index": _positive_int(fields[0]) if fields[0] != "0" else 0,
            "name": fields[1] or None,
            "uuid": fields[2] or None,
            "memory_bytes": memory_mib * 1024 * 1024 if memory_mib else None,
            "driver_version": fields[4] or None,
        })
    return {
        "status": "available" if devices else "unavailable",
        "reason": None if devices else "no_visible_devices",
        "gpus": devices,
    }


def _validate_hardware(metadata):
    required = {"schema_version", "cpu", "memory", "accelerators"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Hardware metadata is missing fields: {missing}")
    if metadata["schema_version"] != HARDWARE_SCHEMA_VERSION:
        raise ValueError("Hardware metadata schema version is invalid.")
    for path, value in (
        ("cpu.available_logical_cores", metadata["cpu"]["available_logical_cores"]),
        ("memory.available_bytes", metadata["memory"]["available_bytes"]),
    ):
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{path} must be a positive integer or null.")
    return metadata


def collect_hardware_metadata():
    """Collect paper-reportable hardware facts without privileged commands."""
    cpu_count, cpu_source = _available_cpu_count()
    memory_bytes, memory_source = _available_memory_bytes()
    metadata = {
        "schema_version": HARDWARE_SCHEMA_VERSION,
        "cpu": {
            "model": _cpu_model(),
            "architecture": platform.machine() or None,
            "available_logical_cores": cpu_count,
            "count_source": cpu_source,
        },
        "memory": {
            "available_bytes": memory_bytes,
            "source": memory_source,
        },
        "accelerators": _nvidia_inventory(),
    }
    return _validate_hardware(metadata)


def _torch_execution_metadata(backend):
    try:
        import torch
    except ImportError:
        return None

    device = getattr(backend, "device", None)
    parameter_device = None
    model = getattr(backend, "model", None)
    if model is not None and hasattr(model, "parameters"):
        try:
            parameter_device = str(next(model.parameters()).device)
        except (StopIteration, TypeError):
            parameter_device = None

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append({
                "index": index,
                "name": properties.name,
                "memory_bytes": int(properties.total_memory),
                "compute_capability": f"{properties.major}.{properties.minor}",
            })
    return {
        "version": str(torch.__version__),
        "cuda_build_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_cuda_devices": cuda_devices,
        "backend_device": str(device) if device is not None else None,
        "model_parameter_device": parameter_device,
    }


def _tensorflow_execution_metadata():
    try:
        import tensorflow as tf
    except ImportError:
        return None

    def names(devices):
        return [str(device.name) for device in devices]

    try:
        physical_cpus = names(tf.config.list_physical_devices("CPU"))
        physical_gpus = names(tf.config.list_physical_devices("GPU"))
        logical_cpus = names(tf.config.list_logical_devices("CPU"))
        logical_gpus = names(tf.config.list_logical_devices("GPU"))
    except (RuntimeError, ValueError):
        physical_cpus = physical_gpus = logical_cpus = logical_gpus = []

    try:
        built_with_cuda = bool(tf.test.is_built_with_cuda())
    except (AttributeError, RuntimeError):
        built_with_cuda = None

    gpu_memory = None
    if logical_gpus:
        try:
            memory = tf.config.experimental.get_memory_info("GPU:0")
            gpu_memory = {
                "current_bytes": int(memory["current"]),
                "peak_bytes": int(memory["peak"]),
            }
        except (AttributeError, KeyError, RuntimeError, ValueError):
            gpu_memory = None

    if not physical_gpus:
        observed_device = "cpu"
        evidence = "no_tensorflow_gpu_visible"
    elif gpu_memory is not None and gpu_memory["peak_bytes"] > 0:
        observed_device = "gpu"
        evidence = "tensorflow_gpu_memory_allocated"
    else:
        observed_device = "unknown"
        evidence = "gpu_visible_without_observed_memory_allocation"

    return {
        "version": str(tf.__version__),
        "built_with_cuda": built_with_cuda,
        "physical_cpus": physical_cpus,
        "physical_gpus": physical_gpus,
        "logical_cpus": logical_cpus,
        "logical_gpus": logical_gpus,
        "gpu_memory": gpu_memory,
        "observed_device": observed_device,
        "device_evidence": evidence,
    }


def collect_backend_execution_metadata(backend):
    """Describe the backend and framework-visible execution device."""
    framework = str(backend.preprocessing_spec.get("framework", "unknown"))
    metadata = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "framework": framework,
        "backend_device": str(getattr(backend, "device", "unknown")),
        "observed_output_device": getattr(backend, "observed_output_device", None),
        "torch": _torch_execution_metadata(backend),
        "tensorflow": (
            _tensorflow_execution_metadata()
            if framework.startswith("tensorflow") else None
        ),
    }
    if metadata["torch"] is not None:
        parameter_device = metadata["torch"]["model_parameter_device"]
        backend_device = metadata["torch"]["backend_device"]
        if parameter_device is not None:
            metadata["observed_device"] = parameter_device
            metadata["device_evidence"] = "model_parameter_device"
        elif framework.startswith("tensorflow"):
            metadata["observed_device"] = metadata["tensorflow"]["observed_device"]
            metadata["device_evidence"] = metadata["tensorflow"]["device_evidence"]
        else:
            metadata["observed_device"] = backend_device
            metadata["device_evidence"] = "backend_device"
    else:
        metadata["observed_device"] = (
            metadata["tensorflow"]["observed_device"]
            if metadata["tensorflow"] else "unknown"
        )
        metadata["device_evidence"] = (
            metadata["tensorflow"]["device_evidence"]
            if metadata["tensorflow"] else "unavailable"
        )
    return metadata


def synchronize_backend(backend):
    """Synchronize accelerator work before reading a wall-clock timer."""
    try:
        import torch
    except ImportError:
        return
    device = getattr(backend, "device", None)
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_peak_memory(backend):
    try:
        import torch
    except ImportError:
        return
    device = getattr(backend, "device", None)
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _process_peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1 if platform.system() == "Darwin" else 1024
    return int(peak * multiplier) if peak >= 0 else None


def collect_peak_memory(backend):
    torch_allocated = torch_reserved = None
    try:
        import torch

        device = getattr(backend, "device", None)
        if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
            torch_allocated = int(torch.cuda.max_memory_allocated(device))
            torch_reserved = int(torch.cuda.max_memory_reserved(device))
    except ImportError:
        pass

    tensorflow = None
    framework = str(backend.preprocessing_spec.get("framework", ""))
    if framework.startswith("tensorflow"):
        tensorflow_metadata = _tensorflow_execution_metadata()
        if tensorflow_metadata is not None:
            tensorflow = tensorflow_metadata["gpu_memory"]

    return {
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "torch_peak_allocated_bytes": torch_allocated,
        "torch_peak_reserved_bytes": torch_reserved,
        "tensorflow_gpu_memory": tensorflow,
    }


def build_embedding_runtime(seconds, n_images, batch_size, num_workers, source,
                            timing_components=None):
    """Build and strictly validate one extraction timing record."""
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Embedding duration must be a finite nonnegative number.")
    if not isinstance(n_images, int) or n_images <= 0:
        raise ValueError("Embedding image count must be a positive integer.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Embedding batch size must be a positive integer.")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("DataLoader worker count must be a nonnegative integer.")
    if source not in {"computed", "cache"}:
        raise ValueError("Embedding source must be 'computed' or 'cache'.")
    record = {
        "source": source,
        "fresh_computation": source == "computed",
        "n_images": n_images,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "wall_seconds": float(seconds),
        "images_per_second": float(n_images / seconds) if seconds > 0 else None,
    }
    if timing_components is not None:
        required = {
            "total_seconds", "compute_seconds", "cache_read_seconds",
            "cache_write_seconds",
        }
        if set(timing_components) != required:
            raise ValueError(
                f"Embedding timing components must be exactly {sorted(required)}."
            )
        validated = {}
        for key in sorted(required):
            value = timing_components[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Embedding timing component {key} is invalid.")
            validated[key] = float(value)
        if source == "cache" and validated["compute_seconds"] != 0:
            raise ValueError("Cached embeddings cannot report computation time.")
        if source == "computed" and validated["cache_read_seconds"] != 0:
            raise ValueError("Fresh embeddings cannot report cache-read time.")
        component_sum = sum(
            validated[key] for key in (
                "compute_seconds", "cache_read_seconds", "cache_write_seconds"
            )
        )
        if component_sum > validated["total_seconds"] + 1e-9:
            raise ValueError("Embedding timing components exceed their total duration.")
        validated["overhead_seconds"] = max(
            0.0, validated["total_seconds"] - component_sum
        )
        validated["compute_images_per_second"] = (
            float(n_images / validated["compute_seconds"])
            if validated["compute_seconds"] > 0 else None
        )
        record["timing_components"] = validated
    return record
