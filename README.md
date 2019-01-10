# data-science-animation-challenge

This is the submition repo for [Harvard Data Science Animation Contest](https://sites.google.com/view/harvard-data-science-animation/home?authuser=0).

This is an interactive visualization of [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost) inspired by [this video](https://www.youtube.com/watch?v=se_ftkIPru8).

The web app can be found at https://adaboost-visualized.herokuapp.com/.

In case the above deployment is down. The visualization can also be viewed locally. Following are the instructions:
1. Clone this repository by `git clone git@github.com:feiyu-chen96/data-science-animation-challenge.git`
2. Go into the directory by `cd data-science-animation-challenge`
3. Create a virtual environment by `virtualenv venv` (assuming `virtualenv` is installed)
4. Activate the virtual environment by `source venv/bin/activate`
5. Install the dependencies by `pip install -r requirements.txt` (assuming `pip` is installed)
6. Launch the app by `python adaboost_visualized.py`
7. Open a browser and see `http://127.0.0.1:8050/`
8. Close the app by `ctrl+C` and deactive the virtual environment by `deactivate` afterwards