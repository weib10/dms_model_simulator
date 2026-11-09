"""DMS model-output simulator.

Coordinate conventions (see README):
  vehicle frame : X forward, Y left, Z up, metres, origin = nominal driver eye point.
  camera frame  : OpenCV (x right, y down, z forward).
  yaw/pitch     : computed in a forward-left-up frame; yaw>0 = left, pitch>0 = up.
"""
__version__ = "0.1.0"
