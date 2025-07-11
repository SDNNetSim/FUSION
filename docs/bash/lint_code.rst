Lint Code
================

This script runs `pylint` on all Python files in the repository, excluding specified directories.
It ensures code quality by catching stylistic and functional issues.

Usage
-----

.. code-block:: bash

   ./bash/lint_code.sh

Make sure the script has execute permissions:

.. code-block:: bash

   chmod +x ./bash/lint_code.sh

Behavior
--------

- Navigates to the repository root before running.
- Excludes the following directories from linting:
  - `./bash`
  - `./.venv`
  - `./docs`
- Uses `find` with `-prune` to efficiently skip ignored directories.
- Runs `pylint` on all discovered `.py` files.
- Exits immediately if any `pylint` check fails.

Dependencies
------------

- Requires `pylint` to be installed. If it's missing, the script will exit and prompt you to install it:

.. code-block:: bash

   pip install pylint

Output
------

- Prints the current working directory.
- Lists all Python files that will be linted.
- Runs `pylint` on each file individually.
- Terminates with an error if any file fails linting.

Example Output
--------------

.. code-block:: text

   Current working directory: /path/to/repo
   Identifying Python files...
   Found Python files:
   ./module1/script.py
   ./module2/utility.py
   Running pylint...
   Linting ./module1/script.py
   Linting ./module2/utility.py
   Linting completed successfully!
