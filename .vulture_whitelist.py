# Vulture whitelist for false positives
# Usage: vulture fusion/ .vulture_whitelist.py

# Common method names that appear unused but are called dynamically
# These are string patterns that vulture recognizes

# Django-style patterns
setUp = True
tearDown = True

# Common simulation framework methods
run = True
execute = True
start = True
stop = True
init = True
setup = True
cleanup = True
process = True
handle = True
update = True
step = True

# Common class attributes
name = True
id = True
type = True
value = True
config = True
logger = True
version = True

# CLI and entry points
main = True
cli = True
command = True
args = True
parser = True

# Event handlers
on_event = True
handle_event = True
callback_event = True

# Property names
property = True

# Mock fixtures - pytest fixtures used by @patch decorators
mock_validate = True
mock_process_req = True
mock_process_opt = True
mock_cwd = True
mock_extract = True
mock_save_input = True

# Abstract method parameters - required for interface documentation
deterministic = True
total_timesteps = True
next_observation = True
next_state = True
timestep = True

# Protocol parameters - required by Python protocols
exc_type = True
exc_val = True
exc_tb = True

# Tuple unpacking in tests - variables that are unpacked but not all used
high = True
kw = True
