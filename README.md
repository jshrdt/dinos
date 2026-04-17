# dinos
Code respository for creating/working with Distributional Noun Structure (DiNoS), supplementing the paper "DiNoS: Creating a Data-Driven German Noun Phrase Lexicon from Universal Dependencies" (SLiDE @ LREC 2026).

DiNoS is a format for data-driven lexica of NP heads, which includes statistical information on the dependents and the morphosyntactic features of their original in-context appearances.

DiNoS is structured as follows:

```
.
├── lemma 1
    |–– count
    |–– gender
    |–– word forms
        |–– form 1
            |–– count
            |–– spec(ifier) types
                |–– det_art_specs (determiner-article-specifiers)
                    |–– count
                    |–– spec_id
                        |–– [id + count]; e.g.: ('das', ('Gender;Neut', 'Case;Nom', 'Number;Sing')) : 2 
                        |–– ...
                |–– gen_art_specs (non-determiner/general-article-specifiers)
                    |–– count
                    |–– spec_id
                        |–– [id + count]; e.g.: ('im', 'APPRART', ('Gender;Masc,Neut', 'Case;Dat', 'Number;Sing')) : 5
                        |–– ...
                |–– other_specs
                    |–– count
                    |–– spec_id
                        |–– [id + count]; e.g.: (('ADJ-ADJA', ('Gender;Neut', 'Case;Acc', 'Number;Sing')),) : 3
                        |–– ...
            |–– feat(ure) types
                |–– [id: count]; e.g.: ('Gender;Neut', 'Case;Dat', 'Number;Sing') : 8
                |–– ...
            |–– dep(endency) rel(ations) types
                |–– [id: count]; e.g.: ('obl', ('Case;Dat', 'Number;Sing')) : 3
                |–– ...
        |–– form 2
            |–– ...
        |–– ...
├── lemma 2
    |–– ...
```

___

# Repository Usage

#### Contents:
```
.
├── code/
    |–– NPs_extractor.py
    |–– build_DiNoS.py
    |–– loader.py
├── data/
    |–– ...
|–– config.json
├── demo.ipynb
```

###  0. (optional) Setup for NP dataset/DiNoS creation:

* Place (German) treebanks in CoNLL-U format into into `./data`
  * See `./data/README.md` for attested treebanks
* Update `./config.json` with filenames

### 1. Creating NP datasets

* From `./dinos`, run:  
    > ```$ python3 code/NPs_extractor.py -v -rfd``` 
* Extracts NPs from CoNLL-U treebanks and creates:
    * `./data/{corpus}_complete.conllu`: A merged version of all specified CoNLL-U files per corpus
    * `./data/{corpus}_nps.conllu`: A CoNLL-U  file consisting of the extracted NPs per corpus
* `-rfd` assign case values (if missing) according to: `deprel=subj`->`Nom`, and `deprel=obj`->`Acc`

### 2. Creating DiNoS-lexica

* From `./dinos`, run:  
    > ```$ python3 code/build_DiNoS.py -v -relem```. 
* `-relem` activates relemmatisation (specific for German data)
* `-demo[some-string]` can be passed to create smaller DiNoS files, limited to lemmas beginning with your string of choice. The file will be named `./data/{corpus}_DiNoS_demo_{str}.json`

### 3. Working with DiNoS

* Use the `DINOS` class defined in `./code/loader.py` and the json package (recommended)
```
import sys
import json
sys.path.append("./code")
from loader import DINOS  
parsed_dinos = DINOS("dinos_filename.json")
```
* Open the demo.ipynb notebook to check how the DiNoS loader class works & how to interact with the data format

___

# Dataset releases

Noun phrase datasets and DiNoS lexica are publihsed on Zenodo under CC BY-SA 4.0:  
* [HDT-NP/-DiNoS](https://doi.org/10.5281/zenodo.19224081)
    * HDT-NP (722,135 NPs, 1.7M tokens)
    * HDT-DiNoS (707,706 NPs; 84,598 unique lemmas; 102,418 unique word forms)
* [GSD-NP/-DiNoS](https://doi.org/10.5281/zenodo.19222243)
    * GSD-NP (49,425 NPs, 119.0k tokens)
    * GSD-DiNoS: (49,416 NPs; 17,433 unique lemmas; 20,190 unique word forms)
___

# Citation

If you use this repository or the associated datasets, pleace cite:

```
@inproceedings{suchardt-laarmannquante-2026-dinos,
    title = "{DiNoS}: Creating a Data-Driven {German} Noun Phrase Lexicon from {Universal} {Dependencies}",
    author = "Suchardt, Jacob Lee and
        Laarmann-Quante, Ronja",
    editor = "Hajič, Jan and
        Hinrichs, Erhard and
        Kübler, Sandra and
        Nivre, Joakim and
        Osenova, Petya and
        Pustejovsky, James",
    booktitle = "Proceedings of the First Workshop on Structured Linguistic Data and Evaluation (SLiDE) @ LREC 2026",
    month = may,
    year = "2026",
    address = "Palma, Mallorca",
    publisher = "TBD",
    url = "TBD",
    pages = "TBD"
}
```
