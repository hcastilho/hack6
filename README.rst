===========
Hackathon 6
===========

This is the source code and documentation of the work produced for the
Lisbon Data Science Starters Academy (LDSSA).


Source
--------
* ``src/exploration`` Data exploration and modelling for the first report
* ``src/hack6`` Preparation of the selected model for online deployment
* ``src/verification`` Data exploration and analysis for the final report
* ``src/webapp`` Code for the online app deployed on heroku


Data
----
* ``data/train.csv`` Initial data provided
* ``data/models`` Trained models save directory (best.pkl was the one deployed)
* ``data/heroku_dump`` Backup of the data received in heroku


Documents
---------
* ``doc/report`` Source and pdf for the first report
* ``doc/final`` Source and pdf for the final report
To re-generate the pdf just run ``make report`` or ``make final``


Reproducing Results
-------------------
The names of the python modules where selected to be as self explanatory as
possible.
To run the code this project should be installed in the python environment.
Installing the ``requirements.txt`` file also adds the project.
A running database with the data collected online is required to run the
code in the ``verification`` module to do this run the commands
 ``make run-db`` and ``make restore-db`` the password for the database
is ``bestpwd``.


Tests
-----
There are tests for the online code, install ``dev_requirements.txt`` and
run ``make test``.


Acknowledgments
---------------
A big thanks to everyone involved in the academy. :)
