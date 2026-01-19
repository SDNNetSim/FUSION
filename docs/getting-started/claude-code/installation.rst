Installation
============

This guide walks you through installing and setting up Claude Code for use with FUSION.

Install Claude Code
-------------------

**Option 1: npm (all platforms)**

.. code-block:: bash

   npm install -g @anthropic-ai/claude-code

**Option 2: Homebrew (macOS)**

.. code-block:: bash

   brew install claude-code

Authenticate
------------

Run the login command to authenticate with your Anthropic account:

.. code-block:: bash

   claude login

This opens a browser window where you can sign in.

Start Using Claude
------------------

Navigate to the FUSION directory and launch Claude:

.. code-block:: bash

   cd path/to/FUSION
   claude

Claude automatically reads the project's ``CLAUDE.md`` file which contains context about
the architecture, coding standards, and key concepts.

Verify Setup
------------

Try a simple command to verify everything is working:

::

   > What is FUSION and what are its main components?

If Claude responds with information about the optical network simulator, you're ready to go.

For complete installation instructions, see the
`official Claude Code documentation <https://docs.anthropic.com/en/docs/claude-code/overview>`_.
