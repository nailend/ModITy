## ModITy - Model for the Identification of Typeregions

This is a Python Package for a Model to Identity Typeregions:

more informations will follow soon:


# German installation informations:

Um die Installation der benötigten Packages zu vereinfachen kann die enivronment.yml Datei
genutzt werden um ein virtual environment mit allen nötigen packages zu installieren.

Um das Environment innerhalb eines Notebook zu nutzen wird folgendes Paket benötigt.

$ conda install ipykernel


Dann kann das Environment erstellt werden. 

$ conda env create -f #path/environment.yml

$ conda activate modity

Dann muss du das environment noch im notebook installieren.

$ ipython kernel install --user --name=modity

Wenn das Jupyter Notebook geöffnet wird, kannst im Reiter „Kernel“ „change Kernel“ das
environment ausgewählt werden.

