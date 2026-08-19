

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def export_traces_jsonl(
    traces: List[Any], output_path: Union[str, Path], compress: bool = False
) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        import gzip

        if not output_path.suffix == ".gz":
            output_path = output_path.with_suffix(output_path.suffix + ".gz")
        open_func = gzip.open
        mode = "wt"
    else:
        open_func = open
        mode = "w"

    with open_func(output_path, mode) as f:
        for trace in traces:
            trace_dict = trace.to_dict() if hasattr(trace, "to_dict") else trace
            json.dump(trace_dict, f)
            f.write("\n")

    logger.info(f"Exported {len(traces)} traces to {output_path}")


def export_traces_json(
    traces: List[Any], output_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)


    trace_dicts = []
    for trace in traces:
        if hasattr(trace, "to_dict"):
            trace_dicts.append(trace.to_dict())
        else:
            trace_dicts.append(trace)


    output_data = {"metadata": metadata or {}, "traces": trace_dicts}


    output_data["metadata"].setdefault("total_traces", len(trace_dicts))
    output_data["metadata"].setdefault("exported_at", time.time())

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Exported {len(traces)} traces to {output_path}")


def export_traces_hdf5(
    traces: List[Any], output_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
) -> None:

    try:
        import h5py
        import numpy as np
    except ImportError:
        logger.error("h5py is required for HDF5 export. Install with: pip install h5py")
        raise ImportError("h5py not installed")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:

        traces_group = f.create_group("traces")
        meta_group = f.create_group("metadata")


        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    meta_group.attrs[key] = value
                else:
                    meta_group.attrs[key] = json.dumps(value)

        meta_group.attrs["total_traces"] = len(traces)
        meta_group.attrs["exported_at"] = time.time()


        for i, trace in enumerate(traces):
            trace_dict = trace.to_dict() if hasattr(trace, "to_dict") else trace
            trace_group = traces_group.create_group(f"trace_{i:06d}")

            for key, value in trace_dict.items():
                if value is None:
                    continue

                if isinstance(value, dict):

                    trace_group.attrs[key] = json.dumps(value)
                elif isinstance(value, list):

                    try:
                        arr = np.array(value)
                        trace_group.create_dataset(key, data=arr)
                    except (ValueError, TypeError):

                        trace_group.attrs[key] = json.dumps(value)
                elif isinstance(value, str):

                    trace_group.attrs[key] = value
                elif isinstance(value, (int, float, bool)):

                    trace_group.attrs[key] = value
                else:

                    trace_group.attrs[key] = json.dumps(value)

    logger.info(f"Exported {len(traces)} traces to {output_path}")


def append_traces_jsonl(
    traces: List[Any],
    output_path: Union[str, Path],
    compress: bool = False,
) -> None:


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        import gzip

        if not output_path.suffix == ".gz":
            output_path = output_path.with_suffix(output_path.suffix + ".gz")
        open_func = gzip.open
        mode = "at"
    else:
        open_func = open
        mode = "a"

    with open_func(output_path, mode) as handle:
        for trace in traces:
            trace_dict = trace.to_dict() if hasattr(trace, "to_dict") else trace
            json.dump(trace_dict, handle)
            handle.write("\n")


def append_trace_jsonl(trace: Any, output_path: Union[str, Path], compress: bool = False) -> None:

    append_traces_jsonl([trace], output_path, compress=compress)


def load_traces_jsonl(input_path: Union[str, Path], compress: bool = False) -> List[Dict[str, Any]]:

    input_path = Path(input_path)

    if compress or input_path.suffix == ".gz":
        import gzip

        open_func = gzip.open
        mode = "rt"
    else:
        open_func = open
        mode = "r"

    traces = []
    with open_func(input_path, mode) as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    return traces


def load_traces_json(input_path: Union[str, Path]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:

    with open(input_path, "r") as f:
        data = json.load(f)

    traces = data.get("traces", [])
    metadata = data.get("metadata", {})

    return traces, metadata


def load_traces_hdf5(input_path: Union[str, Path]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:

    try:
        import h5py
    except ImportError:
        logger.error("h5py is required for HDF5 loading. Install with: pip install h5py")
        raise ImportError("h5py not installed")

    traces = []
    metadata = {}

    with h5py.File(input_path, "r") as f:

        if "metadata" in f:
            meta_group = f["metadata"]
            for key in meta_group.attrs:
                value = meta_group.attrs[key]

                if isinstance(value, str) and value.startswith("{"):
                    try:
                        metadata[key] = json.loads(value)
                    except json.JSONDecodeError:
                        metadata[key] = value
                else:
                    metadata[key] = value


        if "traces" in f:
            traces_group = f["traces"]
            for trace_name in sorted(traces_group.keys()):
                trace_group = traces_group[trace_name]
                trace_dict = {}


                for key in trace_group.attrs:
                    value = trace_group.attrs[key]

                    if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                        try:
                            trace_dict[key] = json.loads(value)
                        except json.JSONDecodeError:
                            trace_dict[key] = value
                    else:
                        trace_dict[key] = value


                for key in trace_group.keys():
                    dataset = trace_group[key]
                    trace_dict[key] = dataset[...].tolist()

                traces.append(trace_dict)

    return traces, metadata


def export_traces(
    traces: List[Any],
    output_path: Union[str, Path],
    format: str = "jsonl",
    compress: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:

    format = format.lower()

    if format == "jsonl":
        export_traces_jsonl(traces, output_path, compress=compress)
    elif format == "json":
        export_traces_json(traces, output_path, metadata=metadata)
    elif format == "hdf5":
        export_traces_hdf5(traces, output_path, metadata=metadata)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'jsonl', 'json', or 'hdf5'")


def load_traces(
    input_path: Union[str, Path], format: Optional[str] = None
) -> Union[List[Dict[str, Any]], tuple[List[Dict[str, Any]], Dict[str, Any]]]:

    input_path = Path(input_path)


    if format is None:
        if input_path.suffix in [".jsonl", ".gz"]:
            format = "jsonl"
        elif input_path.suffix == ".json":
            format = "json"
        elif input_path.suffix in [".h5", ".hdf5"]:
            format = "hdf5"
        else:

            with open(input_path, "rb") as f:
                first_bytes = f.read(10)
                if first_bytes.startswith(b"\x89HDF"):
                    format = "hdf5"
                elif first_bytes.startswith(b"{"):

                    f.seek(0)
                    content = f.read(1000)
                    if b"\n{" in content or b"\n[" in content:
                        format = "jsonl"
                    else:
                        format = "json"
                else:
                    format = "jsonl"

    format = format.lower()

    if format == "jsonl":
        return load_traces_jsonl(input_path, compress=input_path.suffix == ".gz")
    elif format == "json":
        return load_traces_json(input_path)
    elif format == "hdf5":
        return load_traces_hdf5(input_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
