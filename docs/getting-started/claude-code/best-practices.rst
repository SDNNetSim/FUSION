Best Practices
==============

Tips for getting the most out of Claude Code when working with FUSION.

Be Specific
-----------

Instead of broad questions, provide context about which part of the codebase you're asking about.

**Less effective:**

::

   > How does routing work?

**More effective:**

::

   > How does the KShortestPaths routing algorithm select paths in fusion/modules/routing?

Ask for Context First
---------------------

Before making changes, ask Claude to explain the existing code and patterns:

::

   > Before I modify the spectrum assignment logic, explain how FirstFit currently works and what data structures it uses.

Reference Modules by Name
-------------------------

FUSION has many components. Specify which module you're asking about:

::

   > In the spectrum assignment module, how does the guard band calculation work?

::

   > Show me how the reporting module collects blocking probability metrics.

Request Explanations with Code
------------------------------

Ask Claude to show relevant code snippets when explaining concepts:

::

   > Explain how requests flow through the simulation and show me the relevant code paths.

Iterate on Solutions
--------------------

If Claude's first suggestion doesn't quite fit, provide feedback:

::

   > That's close, but I need it to work with multi-core fibers. Can you adjust the implementation?

Break Down Complex Tasks
------------------------

For large changes, break them into smaller steps:

::

   > First, help me understand the current protection mechanism.
   > Now, what would I need to change to add a new protection scheme?
   > Finally, show me how to register the new scheme with the factory.

Verify Understanding
--------------------

Ask Claude to confirm your understanding:

::

   > Let me make sure I understand: the EngineProps class holds simulation state, and SimParams holds configuration. Is that correct?
