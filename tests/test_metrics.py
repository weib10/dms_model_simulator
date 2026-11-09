"""metrics.summarize: empty input, NaN denominators, config threshold, latency-consistent gaze error."""
import math
import numpy as np
from dms_model_simulator import geometry as G
from dms_model_simulator.metrics import summarize
from dms_model_simulator.simulate import Simulation

CAM = "steering_column"


def test_summarize_empty_and_nan_denominators():
    m = summarize([])
    assert m["frames"] == 0 and all(math.isnan(v) for k, v in m.items() if k != "frames")
    s = Simulation({"name": "closed", "behavior": {"blink": {"rate_per_min": 0}},
                    "segments": [{"target": "road_ahead", "duration": 0.5, "eye_openness": 0.1}]}, CAM, seed=1)
    s.run()
    m = s.summarize()
    assert math.isnan(m["gaze_valid_rate_eyes_open"]) and math.isnan(m["gaze_mean_error_deg"])   # no eyes-open frame


def test_valid_min_openness_comes_from_the_run_config():
    sc = {"name": "half", "behavior": {"blink": {"rate_per_min": 0}},
          "segments": [{"target": "road_ahead", "duration": 1.0, "eye_openness": 0.4}]}
    s = Simulation(sc, CAM, seed=1, emulator_overrides={"gaze": {"valid_min_openness": 0.5}})
    s.run()
    assert math.isnan(s.summarize()["gaze_valid_rate_eyes_open"])      # openness 0.4 is not "open" at 0.5
    s2 = Simulation(sc, CAM, seed=1); s2.run()
    assert s2.summarize()["gaze_valid_rate_eyes_open"] > 0.9


def test_gaze_error_is_against_the_records_own_truth_under_latency():
    over = {"pipeline": {"latency_frames": 10, "drop_frame_prob": 0.0}}
    s = Simulation("mirror_check", CAM, seed=4, emulator_overrides=over); s.run()
    own = [G.angle_between(np.array(r["pred"]["gaze"]["dir_vehicle"]), np.array(r["truth"]["gaze"]["dir_vehicle"]))
           for r in s.records if r["pred"]["gaze"] and r["pred"]["gaze"].get("valid")]
    m = s.summarize()
    assert m["gaze_mean_error_deg"] == np.mean(own)
    s0 = Simulation("mirror_check", CAM, seed=4, emulator_overrides={"pipeline": {"drop_frame_prob": 0.0}}); s0.run()
    assert m["gaze_mean_error_deg"] > 1.5 * s0.summarize()["gaze_mean_error_deg"]   # lag shows up like zone / head yaw
