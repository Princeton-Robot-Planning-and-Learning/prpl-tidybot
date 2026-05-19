"""Ceiling-camera ArUco marker detector for the TidyBot++ base.

Publishes robot base poses in the map frame over a multiprocessing-connection
socket on `MARKER_DETECTOR_PORT`. The client side lives in
`prpl_tidybot.interfaces.base_interface.RealBaseInterface.get_map_base_state`.

Adapted from `yixuanhuang98/tidybot_server` (server/ subset).
"""
