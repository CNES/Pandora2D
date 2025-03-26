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
import csv
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

    def __init__(self):
        self._data = []
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")

    def append(self, line):
        self._data.append(line)

    def reset(self):
        self._data.clear()

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = value


expert_mode_config = ExpertModeConfig()
data = Data()


def get_current_memory():
    """
    Get current memory of process

    :return: memory
    :rtype: float

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

    def __init__(self, pid, pipe, interval=0.1):
        """
        Init function of Pandora2dMemProf
        """
        super().__init__()
        self.pipe = pipe
        self.interval = interval
        self.cpu_interval = 0.1
        self.process = psutil.Process(pid)

    def run(self):
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


def generate_figure(
    fig_type: str,
    dataframe,
    values=None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> Figure:
    """
    Generic function to generate different types of plots.

    :param fig_type: Type of figure ('pie', 'box', 'barh')
    :type fig_type: str
    :param dataframe: DataFrame containing the data
    :type dataframe: pd.DataFrame
    :param values: Values for bar chart

    :param title: Title of the chart
    :type title: str
    :param xlabel: Label for x-axis
    :type xlabel: str
    :param ylabel: Label for y-axis
    :type ylabel: str
    :return: Performance graph
    :rtype: plt.Figure
    """
    fig = plt.figure(figsize=(12, 12))
    plt.tight_layout()

    if fig_type == "box":
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

    elif fig_type == "barh":
        hbar = plt.barh(values, dataframe, alpha=0.6)
        small_hbar = [f"{d:.2f}" if d <= (max(dataframe) / 2) else "" for d in dataframe]
        large_hbar = [f"{d:.2f}" if d > (max(dataframe) / 2) else "" for d in dataframe]
        plt.bar_label(hbar, small_hbar, padding=5, fmt="%.2f", color="black")
        plt.bar_label(hbar, large_hbar, padding=-35, fmt="%.2f", color="black")

    plt.title(title)
    return fig


def generate_summary(path_output: os.PathLike, expert_mode_cfg: dict):
    """
    Generate graphs referencing memory management and time for each step.

    :param path_output: output directory
    :type path_output: str
    :param expert_mode_cfg: Dictionary containing expert_mode parameters
    :type expert_mode_cfg: dict
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

    metrics_list = ["mean", "sum"]

    dict_perf = {
        "Time": {"df": grouped["Time (s)"].agg(metrics_list), "unit": "seconds"},  # type: ignore
        "Process time": {"df": grouped["CPU Time (s)"].agg(metrics_list), "unit": "seconds"},  # type: ignore
        "Maximum_memory": {"df": grouped["Max_Memory (MiB)"].agg(metrics_list), "unit": "MiB"},  # type: ignore
        "Start_RAM": {"df": grouped["Start_Ram (MiB)"].agg(metrics_list), "unit": "MiB"},  # type: ignore
        "End_RAM": {"df": grouped["End_Ram (MiB)"].agg(metrics_list), "unit": "MiB"},  # type: ignore
        "MAX_CPU": {"df": grouped["Max_CPU"].agg(metrics_list), "unit": "unit"},  # type: ignore
    }

    # Time graphics
    histo_mean_time = generate_figure(
        "barh",
        dict_perf["Time"]["df"]["mean"],  # type: ignore
        values=dict_perf["Time"]["df"].index,  # type: ignore
        title="Mean time",
        ylabel="Function name",
    )
    histo_total_time = generate_figure(
        "barh",
        dict_perf["Time"]["df"]["sum"],  # type: ignore
        values=dict_perf["Time"]["df"].index,  # type: ignore
        title="Total time",
        ylabel="Function name",
    )
    histo_mean_cpu_time = generate_figure(
        "barh",
        dict_perf["Process time"]["df"]["mean"],  # type: ignore
        values=dict_perf["Process time"]["df"].index,  # type: ignore
        title="Mean CPU time",
        ylabel="Function name",
    )
    histo_total_cpu_time = generate_figure(
        "barh",
        dict_perf["Process time"]["df"]["sum"],  # type: ignore
        values=dict_perf["Process time"]["df"].index,  # type: ignore
        title="Total CPU time",
        ylabel="Function name",
    )

    # Memory graphics
    max_cpu = generate_figure(
        "box",
        dict_perf["MAX_CPU"]["df"],
        title="Max CPU",
        xlabel=str(dict_perf["Maximum_memory"]["unit"]),
        ylabel="Function name",
    )

    max_mem = generate_figure(
        "box",
        dict_perf["Maximum_memory"]["df"],
        title="Maximum memory per task",
        xlabel=str(dict_perf["Maximum_memory"]["unit"]),
        ylabel="Function name",
    )

    # Calls graphics
    occurrences = grouped["Function_name"].value_counts().reset_index()
    occ = generate_figure(
        "barh",
        occurrences["count"],
        values=occurrences["Function_name"],
        title="Number of calls",
        ylabel="Function name",
    )

    # Save all figures in PDF file
    figures = [histo_mean_time, histo_total_time, histo_mean_cpu_time, histo_total_cpu_time, max_cpu, max_mem, occ]
    pdf_filename = f"{folder_name}/{data.timestamp}_graph_perf.pdf"
    with PdfPages(pdf_filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)
