# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA2D
#
#     https://github.com/CNES/Pandora2D
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Contains functions for profiling pandora2d
"""
import csv
import datetime
import logging
import os
import shutil
import time
from dataclasses import dataclass
from functools import wraps
from multiprocessing import Pipe
from pathlib import Path
from threading import Thread
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import psutil
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

THREAD_TIMEOUT = 2


@dataclass
class ExpertModeConfig:
    """
    Expert mode config class
    """

    enable: bool = False


class Data:
    """
    Data class
    """

    def __init__(self) -> None:
        self._data: list[Any] = []
        self.timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")

    def append(self, line):
        self._data.append(line)

    def reset(self) -> None:
        self._data.clear()

    @property
    def timestamp(self) -> str:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = value


expert_mode_config = ExpertModeConfig()
data = Data()


def get_current_memory() -> float:
    """
    Get current memory of process

    :return: memory

    """

    # Use psutil to capture python process memory as well
    process = psutil.Process()
    process_memory = process.memory_info().rss

    # Convert nbytes size for logger (in MiB)
    process_memory = float(process_memory) / 1000000

    return process_memory


class MemProf(Thread):
    """
    MemProf

    Profiling thread with time and memory performances in seconds and  MiB
    """

    def __init__(self, pid, pipe, interval=0.1) -> None:
        """
        Init function of Pandora2dMemProf
        """
        super().__init__()
        self.pipe = pipe
        self.interval = interval
        self.cpu_interval = 0.1
        self.process = psutil.Process(pid)

    def run(self) -> None:
        """
        Run
        """

        try:
            max_mem = 0
            max_cpu = 0

            # tell parent profiling is ready
            self.pipe.send(0)
            stop = False
            while not stop:
                # Get memory
                current_mem = self.process.memory_info().rss
                max_mem = max(max_mem, current_mem)

                # Get cpu max
                current_cpu = self.process.cpu_percent(interval=self.cpu_interval)
                max_cpu = max(max_cpu, int(current_cpu))

                stop = self.pipe.poll(self.interval)

            # Convert nbytes size for logger
            self.pipe.send(float(max_mem) / 1000000)
            self.pipe.send(max_cpu)

        except BrokenPipeError:
            logging.debug("broken pipe error in log wrapper ")


def mem_time_profile(name=None, interval=0.1):
    """
    Pandora2d profiling decorator

    :param: func: function to monitor

    """

    def decorator_generator(func):
        """
        Inner function
        """

        @wraps(func)
        def wrapper_profile(*args, **kwargs):
            """
            Profiling wrapper

            Generate profiling logs of function, run

            :return: func(*args, **kwargs)

            """
            if not expert_mode_config.enable:
                return func(*args, **kwargs)

            # Launch memory profiling thread
            child_pipe, parent_pipe = Pipe()
            thread_monitoring = MemProf(os.getpid(), child_pipe, interval=interval)
            thread_monitoring.start()
            if parent_pipe.poll(THREAD_TIMEOUT):
                parent_pipe.recv()

            start_time = time.perf_counter()
            start_cpu_time = time.process_time()

            memory_start = get_current_memory()

            result = func(*args, **kwargs)

            total_time = time.perf_counter() - start_time
            total_cpu_time = time.process_time() - start_cpu_time

            # end memprofiling monitoring
            parent_pipe.send(0)
            max_memory, max_cpu = None, None
            if parent_pipe.poll(THREAD_TIMEOUT):
                max_memory = parent_pipe.recv()
                max_cpu = parent_pipe.recv()

            memory_end = get_current_memory()

            func_name = func.__name__.capitalize() if name is None else name

            # Prepare data to write to the CSV
            performance_data = [func_name, total_time, total_cpu_time, max_memory, memory_start, memory_end, max_cpu]

            # Check if the file already exists
            file_exists = os.path.exists(f"{data.timestamp}_profiling.csv")

            # Write to CSV using the csv module
            with open(f"{data.timestamp}_profiling.csv", mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                # Write header only if the file does not exist
                if not file_exists:
                    writer.writerow(
                        [
                            "Function_name",
                            "Time (s)",
                            "CPU Time (s)",
                            "Max_Memory (MiB)",
                            "Start_Ram (MiB)",
                            "End_Ram (MiB)",
                            "Max_CPU",
                        ]
                    )

                # Write the performance data
                writer.writerow(performance_data)

            return result

        return wrapper_profile

    return decorator_generator


def generate_barh_figure(series: pd.Series, values: Any, title: str = "") -> Figure:
    """
    Barh figure.

    :param series: Series containing the data
    :param values: Values for bar chart
    :param title: Title of the chart
    :return: Performance graph
    """
    fig = plt.figure(figsize=(12, 12))
    plt.tight_layout()
    hbar = plt.barh(values, series, alpha=0.6)
    small_hbar = [f"{d:.2f}" if d <= (max(series) / 2) else "" for d in series]
    large_hbar = [f"{d:.2f}" if d > (max(series) / 2) else "" for d in series]
    plt.bar_label(hbar, small_hbar, padding=5, fmt="%.2f", color="black")
    plt.bar_label(hbar, large_hbar, padding=-35, fmt="%.2f", color="black")
    plt.title(title)
    return fig


def generate_box_figure(dataframe: pd.DataFrame, title: str = "", xlabel: str = "", ylabel: str = "") -> Figure:
    """
    Box figure.

    :param dataframe: DataFrame containing the data
    :param title: Title of the chart
    :param xlabel: Label for x-axis
    :param ylabel: Label for y-axis
    :return: Performance graph
    """
    fig = plt.figure(figsize=(12, 12))
    plt.tight_layout()
    dataframe.T.boxplot(vert=False, showfliers=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Get median and quartiles
    stats = dataframe.T.describe()
    for idx, col in enumerate(dataframe.T.columns):
        q1 = stats[col]["25%"]
        median = stats[col]["50%"]
        q3 = stats[col]["75%"]
        plt.text(median, idx + 1, f"Med: {median:.2f}", va="center", ha="center", color="black", fontsize=8)
        plt.text(q1, idx + 1, f"Q1: {q1:.2f}", va="center", ha="center", color="blue", fontsize=8)
        plt.text(q3, idx + 1, f"Q3: {q3:.2f}", va="center", ha="center", color="blue", fontsize=8)
    plt.title(title)
    return fig


class PerformanceSummaryItem(TypedDict):
    """Item of a Performance Summary."""

    df: pd.DataFrame
    unit: str


class PerformanceSummary(TypedDict):
    """Performance Summary."""

    Time: PerformanceSummaryItem
    Process_time: PerformanceSummaryItem
    Maximum_memory: PerformanceSummaryItem
    Start_RAM: PerformanceSummaryItem
    End_RAM: PerformanceSummaryItem
    MAX_CPU: PerformanceSummaryItem


def generate_summary(path_output: os.PathLike, expert_mode_cfg: dict) -> None:
    """
    Generate graphs referencing memory management and time for each step.

    :param path_output: output directory
    :param expert_mode_cfg: Dictionary containing expert_mode parameters
    """

    # Copy memory_profiling results in the correct folder
    folder_name = Path(path_output) / expert_mode_cfg.get("folder_name")
    Path.mkdir(folder_name, exist_ok=True)

    csv_data_path = f"{folder_name}/{data.timestamp}_profiling.csv"

    shutil.copy(f"{data.timestamp}_profiling.csv", csv_data_path)
    os.remove(f"{data.timestamp}_profiling.csv")

    # Transform csv to a panda.DataFrame
    resumed_performance_df = pd.read_csv(csv_data_path)
    grouped = resumed_performance_df.groupby("Function_name")

    metrics_list: list[Any] = ["mean", "sum"]  # use Any instead of str because typing of agg method is very annoying

    dict_perf: PerformanceSummary = {
        "Time": {"df": grouped["Time (s)"].agg(metrics_list), "unit": "seconds"},
        "Process_time": {"df": grouped["CPU Time (s)"].agg(metrics_list), "unit": "seconds"},
        "Maximum_memory": {"df": grouped["Max_Memory (MiB)"].agg(metrics_list), "unit": "MiB"},
        "Start_RAM": {"df": grouped["Start_Ram (MiB)"].agg(metrics_list), "unit": "MiB"},
        "End_RAM": {"df": grouped["End_Ram (MiB)"].agg(metrics_list), "unit": "MiB"},
        "MAX_CPU": {"df": grouped["Max_CPU"].agg(metrics_list), "unit": "unit"},
    }

    # Time graphics
    histo_mean_time = generate_barh_figure(
        dict_perf["Time"]["df"]["mean"],
        values=dict_perf["Time"]["df"].index,
        title="Mean time",
    )
    histo_total_time = generate_barh_figure(
        dict_perf["Time"]["df"]["sum"],
        values=dict_perf["Time"]["df"].index,
        title="Total time",
    )
    histo_mean_cpu_time = generate_barh_figure(
        dict_perf["Process_time"]["df"]["mean"],
        values=dict_perf["Process_time"]["df"].index,
        title="Mean CPU time",
    )
    histo_total_cpu_time = generate_barh_figure(
        dict_perf["Process_time"]["df"]["sum"],
        values=dict_perf["Process_time"]["df"].index,
        title="Total CPU time",
    )

    # Memory graphics
    max_cpu = generate_box_figure(
        dict_perf["MAX_CPU"]["df"],
        title="Max CPU",
        xlabel=dict_perf["Maximum_memory"]["unit"],
        ylabel="Function name",
    )

    max_mem = generate_box_figure(
        dict_perf["Maximum_memory"]["df"],
        title="Maximum memory per task",
        xlabel=dict_perf["Maximum_memory"]["unit"],
        ylabel="Function name",
    )

    # Calls graphics
    occurrences = grouped["Function_name"].value_counts().reset_index()
    occ = generate_barh_figure(
        occurrences["count"],
        values=occurrences["Function_name"],
        title="Number of calls",
    )

    # Save all figures in PDF file
    figures = [histo_mean_time, histo_total_time, histo_mean_cpu_time, histo_total_cpu_time, max_cpu, max_mem, occ]
    pdf_filename = f"{folder_name}/{data.timestamp}_graph_perf.pdf"
    with PdfPages(pdf_filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)
