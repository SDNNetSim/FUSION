=================
Testing Standards
=================

Testing guidelines for FUSION development.

.. contents:: Table of Contents
   :local:
   :depth: 2

Running Tests
=============

**All tests:**

.. code-block:: bash

   make test-new

**Specific test file:**

.. code-block:: bash

   pytest tests/test_routing.py

**By keyword:**

.. code-block:: bash

   pytest -k "first_fit"

Writing Tests
=============

**Test structure:**

.. code-block:: python

   def test_feature_name():
       """Test description."""
       # Arrange
       input_data = setup_test_data()
       
       # Act
       result = function_under_test(input_data)
       
       # Assert
       assert result == expected_value

**Use fixtures:**

.. code-block:: python

   @pytest.fixture
   def sample_network():
       return create_test_network()

   def test_routing(sample_network):
       result = route_path(sample_network, src=0, dst=5)
       assert len(result) > 0

Coverage Requirements
=====================

* Aim for >80% overall coverage
* Critical paths must be 100% covered
* Document untested edge cases

See Also
========

* :doc:`workflow` - Development workflow
* :doc:`contributing` - Contribution guidelines
