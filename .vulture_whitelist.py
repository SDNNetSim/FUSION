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
