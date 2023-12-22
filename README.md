clasificacion_robos_fge
==============================

Prediction of the variable *delito seguimiento* in robbery dataset. This project will explore the use of transformers in order to make text classification. Dataset is in spanish and is property of Fiscalia General del Estado

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Preguntas

1. Se puede realizar la predicción de $N$ registros en lotes usando las facilidades de la librería dataset de huggingFace? Esto, pienso se logra usando el modelo en formato tensorflow y generando previamente los embedings (encoding) de los datos.... pensaría que esto haría que la predicción se haga mucho más eficiente y en menor tiempo.
2. Cómo se realiza lo anterior usando el pipeline?

## Procedimiento delitos validados

1. Se genera duplicado del datasetgen.ipynb como datasetgen_delitosvalidados.ipynb
2. Se procede a leer la base de datos con relatos Policia y Fiscalía y la base original con datos de la comisión que no ha sido alterada. Se restringe las operaciones conservando las ndds de la comisión
3. Se ajusta las categorías segun lo solicitado por Comisión en un total de 10 categorías más otros robos
4. Se procede a tratar de reutilizar el dataset_trainset_split.ipynb para separar los datos. Sin embargo, se tratará de hacer que se disponga del doble de relatos para que ingresen tanto el relato de policia como el de fiscalia
5. se obtiene los juegos de datos de entrenamiento, validación y testeo y se procede al entrenamiento

