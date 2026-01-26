"""
Parallel plot generation utilities.

Enables concurrent generation of independent presentation slides.
"""

import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Matplotlib must be configured before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for thread safety


@dataclass
class PlotTask:
    """Definition of a plot generation task."""
    name: str
    func: Callable
    args: Tuple = ()
    kwargs: Dict = None
    output_path: Optional[Path] = None
    priority: int = 0  # Lower = higher priority

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class PlotResult:
    """Result of a plot generation task."""
    name: str
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    result: Any = None


def _execute_task(task: PlotTask) -> PlotResult:
    """
    Execute a single plot task.

    Parameters
    ----------
    task : PlotTask
        Task to execute.

    Returns
    -------
    PlotResult
        Result of the task execution.
    """
    try:
        result = task.func(*task.args, **task.kwargs)
        return PlotResult(
            name=task.name,
            success=True,
            output_path=task.output_path,
            result=result,
        )
    except Exception as e:
        return PlotResult(
            name=task.name,
            success=False,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
        )


def generate_plots_parallel(
    tasks: List[PlotTask],
    max_workers: int = 4,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> List[PlotResult]:
    """
    Generate multiple plots in parallel.

    Parameters
    ----------
    tasks : List[PlotTask]
        List of plot tasks to execute.
    max_workers : int
        Maximum number of concurrent workers.
    progress_callback : Callable, optional
        Callback function called with (task_name, completed, total) for progress reporting.

    Returns
    -------
    List[PlotResult]
        Results for each task.

    Examples
    --------
    >>> tasks = [
    ...     PlotTask("slide1", generate_slide1, (config,), output_path=Path("slide1.png")),
    ...     PlotTask("slide2", generate_slide2, (config,), output_path=Path("slide2.png")),
    ... ]
    >>> results = generate_plots_parallel(tasks, max_workers=4)
    >>> for r in results:
    ...     print(f"{r.name}: {'OK' if r.success else r.error}")
    """
    if not tasks:
        return []

    # Sort by priority
    tasks = sorted(tasks, key=lambda t: t.priority)
    total = len(tasks)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(_execute_task, task): task
            for task in tasks
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results.append(result)
            completed += 1

            if progress_callback:
                progress_callback(task.name, completed, total)

    # Return results in original task order
    result_map = {r.name: r for r in results}
    return [result_map[task.name] for task in tasks]


def generate_plots_sequential(
    tasks: List[PlotTask],
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> List[PlotResult]:
    """
    Generate plots sequentially (for debugging or single-threaded execution).

    Parameters
    ----------
    tasks : List[PlotTask]
        List of plot tasks to execute.
    progress_callback : Callable, optional
        Callback function for progress reporting.

    Returns
    -------
    List[PlotResult]
        Results for each task.
    """
    results = []
    total = len(tasks)

    for i, task in enumerate(tasks):
        result = _execute_task(task)
        results.append(result)

        if progress_callback:
            progress_callback(task.name, i + 1, total)

    return results


def print_progress(task_name: str, completed: int, total: int):
    """Default progress callback that prints to stdout."""
    pct = 100 * completed / total
    print(f"  [{completed}/{total}] ({pct:.0f}%) {task_name}")
