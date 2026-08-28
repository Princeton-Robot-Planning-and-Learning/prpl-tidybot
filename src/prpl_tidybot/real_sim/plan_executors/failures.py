"""Failure raised when a plan executor cannot finish tracking its trajectory."""


class ExecutionFailure(RuntimeError):
    """A plan executor exhausted its tick budget without converging.

    Raised from ``done()`` instead of quietly reporting the segment complete.
    The rest of the plan was refined on the assumption that this segment
    reached its target, so continuing (for example driving the base while
    the arm is still inside a fixture) is not safe; the rollout ends with
    this failure as its finish reason and the robot holds its last commanded
    target.
    """
