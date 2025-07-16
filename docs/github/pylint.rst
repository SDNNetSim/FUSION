Pylint & UML Diagram Generation
=================================

Our Pylint pipeline plays a crucial role in ensuring the consistency and long-term health of our simulator's code.
It automatically examines our Python code, flagging potential stylistic inconsistencies, enforcing best practices,
and helping prevent subtle errors. This helps us deliver a more robust and maintainable codebase.

To ensure that our code adheres to the established quality standards, our Pylint pipeline requires a successful Pylint run using our custom `.pylintrc` configuration file. This file defines specific coding conventions and style guidelines for our project.

**Example:**

.. code-block:: bash

   pylint --rcfile=./.pylintrc  my_python_script.py other_script.py

* In this example, `./.pylintrc` assumes the configuration file is in the same directory you're running the command. Adjust the path if your `.pylintrc` file is located elsewhere.

For more information on PEP 8 coding style guidelines, see `this resource <https://peps.python.org/pep-0008/>`_.

High Level UML Diagram Generation
----------------------------------
Pylint's Pyreverse is recommended for the automatic generation of UML Class diagrams for packages.

The following Pyreverse command is ran in bash script. `$OUTPUT_DIR` is set to a folder named class_diagram_output, and `$MODULE` is a list of packages in the project.

.. code-block:: bash
   
   pyreverse -ASmy "$MODULE" -o png -p "${MODULE}" -d "$OUTPUT_DIR"  

Graphviz is required for the script to function. Pyreverse alone cannot output formats other than .puml, .dot, .mmd. 

`Graphviz installation <https://graphviz.org/download/>`_.

`Pyreverse usage guide <https://pylint.readthedocs.io/en/latest/additional_tools/pyreverse/configuration.html>`_.

After installing Graphviz, the script `generate_diagram.sh` must be run with git bash to generate UML diagrams for the modules set in `$MODULE`.

