"""Client-side interface to the SAM segmentation service (issue #136).

The service runs on the perception laptop's GPU
(``scripts/run_segmentation_server.py``, in its own torch venv) and answers
one request at a time: a wrist frame in, per-instance column extents out.
This package holds the light client (:mod:`client`) and the
:class:`~prpl_tidybot.segmentation.detector.SamEdgeDetector`, which turns the
service's instances into the :class:`CylinderEdges` the visual-servo code
consumes, with the OpenCV detector as the natural fallback when the service
is unreachable.
"""
