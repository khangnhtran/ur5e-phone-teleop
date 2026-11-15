"""Run official robomimic script using native MuJoCo, without touching packages."""
import runpy
import sys
import robosuite
import robosuite.utils.binding_utils
# This installation uses native mujoco 3.2.3. robomimic's OPTIONAL import of
# installed mujoco_py raises a configuration Exception (not ImportError).
# Mark that unused legacy backend unavailable only inside this child process.
assert robosuite.__version__=="1.4.1"
assert "mujoco" in sys.modules
sys.modules["mujoco_py"]=None
sys.argv=sys.argv[1:]
runpy.run_module(sys.argv[0],run_name="__main__")
