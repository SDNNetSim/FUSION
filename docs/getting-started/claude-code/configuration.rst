Configuration
=============

FUSION includes configuration files that help Claude Code understand the project.

CLAUDE.md
---------

Located in the project root, ``CLAUDE.md`` is automatically read by Claude Code when you
start a session in the FUSION directory. It contains:

- **Project overview**: What FUSION is and its key architectural concepts
- **Code organization**: How modules are structured and naming conventions
- **Domain knowledge**: Optical networking terminology and concepts
- **Development workflow**: Quality tools, testing standards, and contribution guidelines
- **Key files**: Important entry points and where to find specific functionality

You don't need to do anything special to use this file. Claude reads it automatically
and uses the information to provide more accurate and contextual responses.

.claude/settings.json
---------------------

Project-specific Claude Code settings are stored in ``.claude/settings.json``:

.. code-block:: json

   {
     "claude_code_max_output_tokens": 64000
   }

**Settings explained:**

``claude_code_max_output_tokens``
   Maximum length of Claude's responses. Set to 64000 to allow comprehensive explanations
   and longer code generation when needed.

.claudeignore (Optional)
------------------------

You can create a ``.claudeignore`` file to exclude files from Claude's context,
similar to ``.gitignore``. This is useful for large data files or generated content.

Example patterns for FUSION:

.. code-block:: text

   # Large data files
   *.pkl
   *.npy
   data/

   # Build artifacts
   __pycache__/
   *.pyc
   .mypy_cache/

   # Output directories
   output/
   logs/

Customizing for Your Workflow
-----------------------------

If you frequently work on specific parts of FUSION, you can add notes to ``CLAUDE.md``
about your focus areas. For example:

.. code-block:: markdown

   ## Current Development Focus

   Working on the survivability module, specifically:
   - Adding new failure types
   - Improving recovery metrics

This helps Claude provide more relevant suggestions for your current work.
