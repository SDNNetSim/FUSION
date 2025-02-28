Artificial Intelligence
=======================

What is Artificial Intelligence (AI)? What types are used in the Simulator?
---------------------------------------------------------------------------
Artificial Intelligence is a general term for human-like thinking capabilities for computers. It has existed in some
capacities for decades, as there are many different subtypes. The main subtypes are as follows: Machine Learning, Deep
Learning, and Generative AI. Our simulator uses both Machine Learning and Deep Learning (in the form of Reinforcement
Learning).

Machine Learning (ML)
---------------------
While there are many forms of machine learning, the simplest is that in which there are labeled data sets in which the
algorithm trains off of. However, this form of AI has its drawbacks as humans have to manually label the sets of data.
Our simulator can run with supervised machine learning. An example of an algorithm used is the decision tree classifier,
which aids in routing and spectrum assignment. It allows for data optimization by diving the space in order to find the
highest information gain. This allows for multiple different routing strategies to be tried simultaneously.

Reinforcement Learning (RL)
---------------------------
In Reinforcement Learning, the algorithm attempts an action and receives either an award or a negative award. Based on
the results of its actions, it then can decide what action to take next.
Our simulator has many different uses of reinforcement learning algorithms. The algorithms that have been used at length
are Q-learning, epsilon-greedy bandit, and Upper Confidence Bound (UCB). They fall under two types of reinforcement
learning: bandit algorithms (such as epsilon greedy or UCB) and temporal difference method (such as Q-learning).
Similarly to machine learning, they have uses in both routing and spectrum assignment.
Our simulator also uses StableBaselines3, which allows for a user to work with pre-trained agents and instruction on
how to train, increasing reliability of the reinforcement learning function.

Papers Written by Members of this Project
-----------------------------------------
`[Article] Dynamic Crosstalk-Aware Routing, Modulation, Core, and Spectrum Allocation for Sliceable Demands in SDM-EONs
<https://doi.org/10.1109/LANMAN61958.2024.10621885>`_
# TODO: if ANTS papers are accepted, link here
# TODO: add LATINCOM paper when published

How to run a Machine Learning scenario on our simulator
-------------------------------------------------------
To run a machine learning scenario, the following is a config file that works for v5.0.0

.. code-block:: python

    [ml_settings]
    deploy_model = False
    output_train_data = False
    ml_training = True
    ml_model = decision_tree
    train_file_path = Pan-European/0531/22_00_16_630834
    test_size = 0.3

How to run a Reinforcement Learning scenario on our simulator
-------------------------------------------------------------
To run a reinforcement learning scenario, the following is a config file that works for v5.0.0

Note: the values for ``path_model``, ``core_model``, and ``spectrum_model`` are all for a pre-trained model. Due to the
model being pre-trained, ``is_training`` is set to ``false``. If one were to train the model themselves, these
parameters would be different and ``is_training`` would be set to true.

.. code-block:: python
    [rl_settings]
    device = cpu
    optimize = False
    is_training = False
    path_algorithm = ucb_bandit
    path_model = greedy_bandit/NSFNet/0617/16_47_22_694727/state_vals_e750.0_routes_c4.json
    core_algorithm = first_fit
    core_model = greedy_bandit/NSFNet/0617/16_57_13_315030/state_vals_e750.0_cores_c4.json
    spectrum_algorithm = first_fit
    spectrum_model = ppo/NSFNet/0512/12_57_55_484293

For deep reinforcement learning, add:

.. code-block:: python
    render_mode = None
    super_channel_space = 3

For Q-learning, add:

.. code-block:: python

    learn_rate = 0.01
    discount_factor = 0.95
    epsilon_start = 0.2
    epsilon_end = 0.05
    reward = 1
    penalty = -100
    dynamic_reward = False
    path_levels = 2
    decay_factor = 0.01
    core_beta = 0.1
    gamma = 0.1


Additional Resources
--------------------
Artificial Intelligence Overview

`[Article] IBM What is Artificial Intelligence (AI)?
<https://www.ibm.com/topics/artificial-intelligence>`_

Machine Learning

`[Article] McKinsey & Company What is Machine Learning
<https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-machine-learning>`_

`[Article] IBM What is Machine Learning (ML)?
<https://www.ibm.com/topics/machine-learning>`_

Reinforcement Learning

`[Article] Synopsys What is Reinforcement Learning?
<https://www.synopsys.com/glossary/what-is-reinforcement-learning.html>`_

`[Article] MathWorks What is Reinforcement Learning?
<https://www.mathworks.com/discovery/reinforcement-learning.html>`_

`[Article] IBM What is Reinforcement Learning?
<https://www.ibm.com/topics/reinforcement-learning>`_

StableBaselines3

`[Web] Stable-Baseline3 Docs
<https://stable-baselines3.readthedocs.io/en/master/index.html>`_
