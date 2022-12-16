Usage
=====

Installation
------------

To use ISE, first install it using git clone:

.. code-block:: console

   (.venv) $ git clone https://github.com/Brown-SciML/ise

Creating recipes
----------------

To process a list of random ingredients,
you can use the ``ise.get_random_ingredients()`` function:

.. py:function:: lumache.get_random_ingredients(kind=None)

   Return a list of random ingredients as strings.

   :param kind: Optional "kind" of ingredients.
   :type kind: list[str] or None
   :return: The ingredients list.
   :rtype: list[str]