# Copyright (c) 2026 H. L. Nourse
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: H. L. Nourse

"""Functions to split independent tasks over MPI ranks."""

from collections.abc import Mapping
from typing import Any, Protocol


class Comm(Protocol):
    """
    The part of an ``mpi4py`` communicator used in RISB.

    ``mpi4py.MPI.COMM_WORLD`` is one.
    """

    @property
    def rank(self) -> int:
        """The rank of this process."""
        ...

    @property
    def size(self) -> int:
        """The number of processes."""
        ...

    def allgather(self, sendobj: Any) -> list[Any]:
        """Return the objects of every rank, in order of rank, on every rank."""
        ...

    def allreduce(self, sendobj: Any) -> Any:
        """Return the sum of the objects of every rank on every rank."""
        ...


class SerialComm:
    """A communicator of a single process, for when there is no MPI."""

    rank = 0
    size = 1

    def allgather(self, sendobj: Any) -> list[Any]:
        """Return the object of this process in a list."""
        return [sendobj]

    def allreduce(self, sendobj: Any) -> Any:
        """Return the object of this process."""
        return sendobj


def round_robin(comm: Comm, n: int) -> range:
    """
    Return the indices of the tasks of this rank, of ``n`` tasks split over the ranks.

    Parameters
    ----------
    comm : Comm
        Communicator of the ranks.
    n : int
        Number of tasks.

    Returns
    -------
    range
        Indices of the tasks of this rank.

    """
    return range(comm.rank, n, comm.size)


def gather_tasks[T](comm: Comm, results: Mapping[int, T]) -> dict[int, T]:
    """
    Return the results of the tasks of every rank on every rank.

    Parameters
    ----------
    comm : Comm
        Communicator of the ranks.
    results : mapping
        Results of the tasks of this rank, by the index of the task.

    Returns
    -------
    dict
        Results of every task, in order of the index of the task.

    """
    parts = comm.allgather(dict(results))
    return dict(
        sorted((item for part in parts for item in part.items()), key=lambda x: x[0])
    )
