import json
import re

### DiNoS CLASSES ###
class DINOS:
    """Overarching class for distributional data structure (DiNoS), take and
    laod the corresponding DiNoS json file, parse, and create OOP equivalent
    of DiNoS json structure with shortcuts. Precompute absolute frequencies on
    lemma and corpus bases, as well as relative frequency distributions on
    form (micro), lemma (maicro), and corpus basis (macro) for annotational
    features, dependency relations, specifier types, for unique and over-
    arching types."""
    def __init__(self, dinos_file: str):
        self.all = load_json(dinos_file)  # used to show entire contents
        self.name = dinos_file  # name of file
        # Dictionary of lemma forms: LEMMA object type; iterate over values
        # for access to LEMMA data
        self.lemmas = {lemma: LEMMA(lemma, entry) for
                       lemma, entry in self.all.items()}
        # Corpus size: total amount of NP heads
        self._count = sum([lemma._count for lemma in self.lemmas.values()])

        # Shortcut: unique identifiers of word forms across lemmas: Form
        self.forms = {}
        for lemma in self.lemmas.values():
             for form, entry in lemma.forms.items():
                if form not in self.forms:
                    self.forms[form] = entry
                else:
                    i=0
                    while form in self.forms:
                        form = form+str(i)  # create unique IDs for homographs
                    self.forms[form] = entry

        # Assign distributional attributes (freqs on macro basis across lemmas)
        for attrname in ['specs', 'detarts', 'gendets', 'otherspecs',
                         'features', 'deprels']:
            output = self._get_feat_overview(mode=attrname)
            setattr(self, attrname+'_unique', output[0])
            setattr(self, attrname, output[1])

       # self.spec_feats = ... deprecated
        # self.specs = {lemma: FREQ({spectype: self.lemmas[lemma].specs[spectype].abs
        #                 for spectype in self.lemmas[lemma].specs}, macro=self.count)
        #               for lemma in self.lemmas}


    def _get_feat_overview(self, mode: str|bool = False) -> tuple:
        """Helper function to calculate distributions on various levels."""
        # Depending on attribute (mode), access correct features from LEMMAs
        # to create overarchign distributions; also differentiate between
        # (1) unique IDs (including disambiguation with additional feature
        # tags) and cumulative IDs (only considering highest order categories)
        # e.g.  (1) deprel: nsubj (Nom, Sing) != nsubj (Nom, Plur)
        #       (2) deprel: nsubj (Nom, Sing) & nsubj (Nom, Plur) -> nsubj
        ids_unique = {}
        ids_cumul = ({} if mode!='features' 
                     else {'Gender': {}, 'Case': {}, 'Number':{}})
        for lemma in self.lemmas.values():
            # Iterate over relevant attribute dict
            iterdict = (getattr(lemma, mode).abs if mode!='specs'
                        else getattr(lemma, mode))
            # Depending on mode, iterate to correct depth, extract simplified
            # and unique IDs with values.
            for subdict in iterdict.values():
                if mode=='specs':
                    for specdict_type, subdict in subdict.abs.items():
                        for id, count in subdict.items():
                            # tbs use recstack
                            init_dict(ids_unique, specdict_type)
                            init_dict(ids_cumul, specdict_type)
                            init_dict(ids_unique[specdict_type], id, count)
                            init_dict(ids_cumul[specdict_type], id[0], count)
                else:
                    # different path for specs here: iter through subdicts first,
                    for id, count in subdict.items():
                        init_dict(ids_unique, id, count)
                        if mode=='features':
                            init_dict(ids_cumul['Gender'], id[0], count)
                            init_dict(ids_cumul['Case'], id[1], count)
                            init_dict(ids_cumul['Number'], id[2], count)
                        else:
                            init_dict(ids_cumul, id[0], count)

        # Create distributions, macro for mode!='specs' accessible from
        # summing values.
        macro = self._count if mode=='specs' else False
        ids_cumul = FREQ(ids_cumul, macro)
        ids_unique = FREQ(ids_unique, macro)

        return ids_unique, ids_cumul

    def __getitem__(self, lemma: str):
        return self.lemmas[lemma] 

    def __len__(self):
        return len(set(self.all))


class LEMMA:
    def __init__(self, lemma: str, lemmadict: dict):
        self.all = lemmadict   # used to show entire lemma's json dict
        self.form = lemma  # word form
        self._count = lemmadict['count']  # n occurences of lemma in corpus
        self.gender = (re.findall(r'\w+|\w+,\w+', lemmadict.get('gender', '_'))[0]
                       if type(lemmadict.get('gender', '_'))==str else '_')
        self.forms = {form: WORDFORM(form, entry) 
                      for form, entry in lemmadict['forms'].items()}

        # Assign parallel attributes (freqs on macro basis across forms)
        for attrname in ['counts', 'detarts', 'gendets', 'otherspecs',
                         'features', 'deprels']:
            setattr(self, attrname,
                FREQ({form: (getattr(self.forms[form], attrname).abs
                             if attrname!='counts'
                             else getattr(self.forms[form], '_count'))
                      for form in self.forms}, macro=self._count))

        self.specs = {form: FREQ({spectype: self.forms[form].specs[spectype].abs
                                  for spectype in self.forms[form].specs},
                                 macro=self._count)
                      for form in self.forms}

    def __getitem__(self, idx: int|str):
        if type(idx)==int:
            return list(self.forms.items())[idx]
        else:
            return self.forms[idx]

    def __len__(self):
        return len(self.forms)

class WORDFORM:
    def __init__ (self, form: str, formdict: dict):
        self.all = formdict   # used to show entire form's json dict
        self.form = form
        self._count = formdict['count']
        self.features = FREQ(self.parse_dict('feat_types'))
        self.deprels = FREQ(self.parse_dict('deprel_types'))
        self.detarts = FREQ(self.parse_dict('det_art_specs', mode='specs'))
        self.gendets = FREQ(self.parse_dict('gen_det_specs', mode='specs'))
        self.otherspecs = FREQ(self.parse_dict('other', mode='specs'))
        self.specs = {'det_art_specs': FREQ(self.detarts.abs, macro=self._count),
                      'gen_det_specs': FREQ(self.gendets.abs, macro=self._count),
                      'other': FREQ(self.otherspecs.abs, macro=self._count)}


    def parse_dict(self, dictname: str, mode: str|bool = False) -> dict:
        # Read & parse json dictionaries back into python Syntax, slight
        # differences for splitting/regex depending on type of subdict.
        if mode =='specs':
            parsed = {parse_json_tup(id, how='identifier_strict'): count
                      for id, count
                      in self.all['spec_types'][dictname]['spec_id'].items()}
        else:
            id_type = ('identifier_strict' if dictname =='deprel_types'
                       else 'identifier')
            parsed = {parse_json_tup(id, how=id_type): count
                      for id, count in self.all[dictname].items()}

        return parsed

class FREQ:
    """Contains passed absolute frequency distriution and calculates
    corresponding relative frequency distribution, optional: macro value,
    defaults to sum of values of given absolute distribution."""
    def __init__(self, freqdict: dict, macro: int|bool = False):
        self.abs = freqdict
        self.rel = self._as_rel(freqdict, macro) if self.abs else {}

    def _as_rel(self, freqdict: dict, macro: int|bool = False) -> dict:
        if type(list(freqdict.values())[0])==int:
            # is not nested
            if not macro:
                macro = sum(self.abs.values())
            rel_freqs = dict(sorted(
                                {form: val/macro
                                 for form, val in freqdict.items()}.items(),
                                key=lambda x: x[1], reverse=True))
        else:
            # is nested: macro mandatory
            rel_freqs = {form: dict(sorted(
                            {id: (val/macro if macro else val/sum(entry.values()))
                            for id, val in entry.items()}.items(),
                            key=lambda x: x[1], reverse=True))
                         for form, entry in freqdict.items()}

        return rel_freqs


### assorted help functions ###

def load_json(path: str) -> dict:
    """Return data loaded from specified json file as dict."""
    print(f'Load data from {path}...')
    with open(path) as f:
        data = json.load(f)
    return data

def parse_json_tup(json_tup: str, how: str|bool = False) -> tuple|dict:
    """Safely parse various string tuples from json file to python tuples."""
    if how=='identifier_strict':
        id =  re.findall(r'\w+[:]?\w+|\w+', json_tup.split()[0])[0]
        parsed = tuple((id, tuple([feat_mark.split(';')[1]
                                   for feat_mark in re.findall(r'\w+;\w+',
                                                               json_tup)])))
    elif how=='identifier':
        parsed = tuple([feat_mark.split(';')[1] 
                        for feat_mark in re.findall(r'\w+;\w+,\w+|\w+;\w+',
                                                    json_tup)])

    elif ';' in json_tup:
        parsed = {feat_mark.split(';')[0]: feat_mark.split(';')[1]
                  for feat_mark in re.findall(r'\w+;\w+', json_tup)}

    else: 
        parsed = tuple([feat for feat in re.findall(r'\w', json_tup)])

    return parsed 


def init_dict(mydict: dict, newkey: str, atomic: bool = False) -> dict:
    """Lazy function to create nested dicts/counter value without
    overwriting.

    Args:
        mydict (dict): Outer dictionary to add new key/value to.
        newkey (str): New key to add to outer dictionary.
        atomic (int/float/bool, optional): Value to add to new key
            (int/float-value, True-0). Defaults to False - nested dict.

    Returns:
        _type_: _description_
    """
    if type(atomic)==int or type(atomic)==float:
        mydict[newkey] = mydict.get(newkey, 0) + atomic
    elif atomic:
        mydict[newkey] = mydict.get(newkey, 0)
    else:
        mydict[newkey] = mydict.get(newkey, dict())
    return mydict


if __name__ == '__main__':
    pass