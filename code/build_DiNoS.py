# Script to transform and relemmative Conllu format NP file into DiNoS 
# i.e json compatible dict with various counts for lemmas, forms:
# features, specifiers, word forms, deprel distinctions

# Imports
import argparse
import re
import json

import nltk
from tqdm import tqdm

from loader import init_dict
from NPs_extractor import parse_conllu_files, save_data

parser = argparse.ArgumentParser()
parser.add_argument('-corp', '--corpora', help='source corpus',
                    default='GSD HDT')
parser.add_argument('-v', '--verbosity', help='amount of detail printed',
                    action='store_true', default=False)
parser.add_argument('-relem', '--relemmatise', help='relemmatis entries in DiNoS (recommended)',
                    action='store_true', default=False)
parser.add_argument('-demo', '--create_demo', help='create small DiNoS of lemmas beginning with str.',
                    default=False)
parser.add_argument('-noi', '--noisy', help='create noisy DiNos or drop irreg toks',
                    action='store_true', default=False)

args, unk = parser.parse_known_args()
config = json.load(open('config.json'))


def get_dinos_data(file: list, goal: list = ['Gender', 'Case', 'Number']) -> dict:
    """Transform conllu formatted file(s) of NPs into one DiNos structure.

    Args:
        file (list): List of one/multiple conllu files.
        goal (list, optional): Annotation goal. Defaults to ['Gender',
            'Case', 'Number'].

    Returns:
        dict: DiNos dictionary, json compatible.
    """
    # Parse conllu files
    parsed_nps = parse_conllu_files(file)
    # Initiate dict to collect all DiNos frequency data
    freqs = {}

    print('Transforming data to DiNos')
    for np in tqdm(parsed_nps):
        # Get NP head lemma & form
        head = np.filter(upos='NOUN', xpos='NN',
                         deprel=lambda x: x not in ['compound', 'amod'])[0]
    ## TBD check for preexisting lemma entry; gender match? if not -> new lemma_i
    # messes with relemmatisation though
        lemma = head['lemma']
        form = head['form']
        feat_id = str(tuple([featkey+';'+grab_val(head['feats'],featkey)
                             for featkey in goal]))
        # Get deprel info as (deprel_tag, (Case;val, Number;val))
        deprel_id =  str(((head['deprel'],
                           ('Case;'+grab_val(head['feats'], 'Case'),
                            'Number;'+grab_val(head['feats'], 'Number')))))

        # Quick-build DiNos structure (nested dictionaries), automatically
        # adds +1 to form , feat_type, and deprel_type count
        # strings: init empty nested dictionary
        # int: init 'count' with value +int
        # tuple: init dict key in nested dict to the left, with value +int
        # lists indicate +1 nesting depth; inside dict to left
        # updates recursively; specifiers need to be handled spearately though
        build_dinos_dict(freqs, lemma, form,
                            [1, 'spec_types', ['det_art_specs', [0, 'spec_id' ],
                                              'gen_det_specs', [0, 'spec_id'],
                                              'other', [0, 'spec_id']],
                             'feat_types', [(feat_id, 1)],
                             'deprel_types', [(deprel_id, 1)]])
        freqs[lemma]['count']+=1  # update lemma counter
        # add gender tag (default '_', override with non-_)

        freqs[lemma]['gender'] = freqs[lemma].get('gender', '_')
        if grab_val(head['feats'], 'Gender') != '_':
            freqs[lemma]['gender'] = grab_val(head['feats'], 'Gender')

        # Set shortcut to specs subdictionary to add to 'spec_id' dict
        shorthand = freqs[lemma]['forms'][form]['spec_types']

        ## Get collocations, assignment to categories is hierarchical:
        # highest order: DET/ART token
        if np.filter(xpos='ART'):
            tok = np.filter(xpos='ART')[0]
            # extract feature annotation info as tuple of (feat;val) tuples
            spec_feats = tuple(([fkey+';'+grab_val(tok['feats'], fkey)
                                 for fkey in goal]))
            # add specifier entry as str of tuple: (token form, features)
            spec = str((tok['form'].lower(),  spec_feats))
            # initiate new count tracker entry for spec form
            init_dict(shorthand['det_art_specs']['spec_id'], spec, True)
            # Add counts for general det_art_specs and specific specifier
            shorthand['det_art_specs']['count'] += 1
            shorthand['det_art_specs']['spec_id'][spec] += 1

        # second: other DET (non-ART) tokens
        elif np.filter(upos='DET') or np.filter(xpos='APPRART'):
            tok = np.filter(upos='DET')[0] if np.filter(upos='DET') else np.filter(xpos='APPRART')[0]

           # bypass non-exist/None error
            if tok.get('feats', None)==None: tok['feats'] = dict()

            spec_feats = tuple([fkey+';'+grab_val(tok['feats'], fkey)
                                for fkey in goal]) 
            # aside from token form, add XPOS info to entry to differentiate
            spec = str((tok['form'].lower(), tok['xpos'], spec_feats))
            init_dict(shorthand['gen_det_specs']['spec_id'], spec, True)
            shorthand['gen_det_specs']['spec_id'][spec] += 1
            shorthand['gen_det_specs']['count'] += 1

        else:
            # else/lowest: if other tokens beside head, simplify:
            # append tuple of UPOS/XPOS tuples of all non-DETART tokens
            # since assignment is hierachical: only needs to verify tok is not head
            remainder = [tok for tok in np if tok != head]
            if remainder:
                for tok in remainder:
                    if tok['feats']==None: tok['feats'] = dict()
                    spec = list()
                    tok_spec_feats = tuple([fkey+';'+grab_val(tok['feats'], fkey)
                                            for fkey in goal])
                    tok_spec = (tok['upos']+'-'+tok['xpos'], tok_spec_feats)
                    spec.append(tok_spec)
                spec = str(tuple(spec))
            else:
                # If np consists of only 1 head token: non_spec placeholder
                spec = 'non_spec'
            # update
            init_dict(shorthand['other']['spec_id'], spec, True)
            shorthand['other']['spec_id'][spec] += 1
            shorthand['other']['count'] += 1

    return freqs


def grab_val(mydict, mykey):
    """Get value from dict, default to _ if no value or value None."""
    return mydict.get(mykey, '_') if mydict.get(mykey, '_')!=None else '_'


def build_dinos_dict(mydict: dict, outer: str, nest: str = False,
                   unique: list = False):
    """High level function to quickly build DiNos internal structure per
    lemma entry: determine substructure, allow for passing of counts, 
    differntiate between building new lemma entry or create form entry below
    existing lemma. In-place.

    Args:
        mydict (dict): Dict to build on (DiNos dict, or DiNos lemma dict).
        outer (str): Lemma/form outer dict key to build DiNos into.
        nest (str, optional): Form to pass along with lemma, will build/
            update nested dict. Defaults to False.
        unique (str, optional): Unique structure to build, typically with
            non-zero counter values. Defaults to False.
    """
    # indentation reflects nested structure, str-> dict name; list -> nested
    # dict structure in dict name to the left, int -> 'count': 0
    dinos_structure = [0, 'spec_types',
                            ['det_art_specs', [0, 'spec_id'],
                             'gen_det_specs', [0, 'spec_id'],
                             'other', [0, 'spec_id']],
                        'feat_types',
                        'deprel_types'
                        ] if unique == False else unique
    if nest:
        # Lemma: form: nested dict
        quick_build = [outer, [0, 'forms', [nest, dinos_structure]]]
    else:
        # Form: nested dict
        quick_build = [outer, dinos_structure]

    # Safely build structure without overwriting preexisting values
    build_n_depth_dict(mydict, quick_build)


def build_n_depth_dict(mydict: dict, build_dicts: list):
    # together with build_dinos_dict: recursively create nested dicts,
    # strongly dependent of type checking in structure creation
    for i, dict_id in enumerate(build_dicts):
        if type(dict_id)==str:
            init_dict(mydict, dict_id)
        elif type(dict_id)==int:
            init_dict(mydict, 'count', dict_id)
        elif type(dict_id)==list:
            if type(build_dicts[i-1])==str:
                build_n_depth_dict(mydict[build_dicts[i-1]], dict_id)
            else:
                build_n_depth_dict(mydict[build_dicts[i-2]], dict_id)
        elif type(dict_id)==tuple:
            init_dict(mydict, dict_id[0], dict_id[1])

    return mydict


### RELEMMATISATION ###

def re_lemmatise(dinos_frame: dict, add_unks: bool = False,
                 return_unks: bool = False) -> tuple[dict, dict]:
    """Take a DiNos and apply string based heuristics relemmatise entries.
    Drop completely numeric entries (e.g. 2010). Optional: keep, return 
    separately, or entirely drop entries with unknown lemma.

    Args:
        dinos_frame (dict): lemma/form disributional stats in nested dict format
        add_unks (bool, optional): _description_. Defaults to False.
        return_unks (bool, optional): _description_. Defaults to False.

    Returns:
        dict|tuple[dict, dict]: Relemmatised stats dict; or additionally dict
            of lemma 'Uknown' items.
    """
    # store relemmatised frequency dict, unkown lemmas+entries, digits
    relem, unks, digits = dict(), dict(), dict()
    # Iterate over all lemma entries in DiNos and check compatibility of lemma
    # form with contained word forms on string basis to determine lemma-form
    # matching.
    for lemma, lemma_entry in tqdm(dinos_frame.items()):
        formdicts = lemma_entry['forms']
        lemma_candidates = list()
        lemma_norm = re.sub(r'\W+', '', lemma).lower()
        lemma_norm = re.sub(r'ß', 'ss', lemma_norm)
        # future: might need to determine gender based on distribution across
        # forms: annotation quality seems lacking (<90%)
        gender = lemma_entry['gender']

        # Go throug formdicts sorted alphabetically/by lengths
        # (non-compound/inflected items will be seen first)
        for form, formdict in sorted(formdicts.items(),
                                    key=lambda item: len(item[0])):
            # regularise spelling
            form_norm = re.sub(r'\W+', '', form).lower()
            form_norm = re.sub(r'ß', 'ss', form_norm)

            if form_norm.isdigit():  # skip/drop complete numeric items 
                dict_update(digits, lemma_norm, form_norm, formdict)
                continue

            # 1) Find non-inflected forms (base lemma + compund lemma); 
            # add as lemmas, form entries, save to lemma list
            # e.g. Geschichte, Firmengeschichte
            if form_norm[-len(lemma_norm):] == lemma_norm:
                lemma_candidates.append(form_norm.capitalize())
                relem = dict_update(relem, form_norm, form_norm, formdict)
                relem[form_norm.capitalize()]['gender'] = gender

            # 2) Simple inflected forms (base+compounds) (stem unaltered, suffix)
            # E.g. Geschichten, Firmengeschichten
            elif lemma_norm in form_norm:
                # Try to match base lemma (+left compound noun) as substring -> new lemma
                new_lemma = re.search(r'[a-zA-Z]*'+lemma_norm.lower(),
                                    form_norm.lower()).group().capitalize()
                lemma_candidates.append(new_lemma)
                relem = dict_update(relem, new_lemma, form_norm, formdict)
                relem[new_lemma.capitalize()]['gender'] = gender

            # 3) replace word internal umlauts, repeat 2)
            elif (re.match(r'.*[äöü]', form_norm)
                  and lemma_norm in form_norm.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u')):
                form_norm_re = form_norm.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u')
                # form_norm_re = form_norm.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u')
                # if lemma_norm in form_norm_re:
                new_lemma = re.search(r'[a-zA-Z]*'+lemma_norm.lower(),
                                    form_norm_re.lower()).group().capitalize()
                lemma_candidates.append(new_lemma)
                relem = dict_update(relem, new_lemma, form_norm, formdict)
                relem[new_lemma.capitalize()]['gender'] = gender

            # 4) Simple inflected form (optional with stem change) from og_lemma
            # E.g. Ärzte
            elif str_distance(form_norm, lemma_norm):
                relem = dict_update(relem, lemma_norm, form_norm, formdict)
                relem[lemma_norm.capitalize()]['gender'] = gender
            # Deal with unrecognised inflected compounds
            else:
                # 5) Inflected compounds (with stem change?) & lemma hypothesis exists
                # E.g.: Supermärkte -> check against base & created lemmas from
                # current base lemma 'Markt'; perform distance check -> 'Supermarkt' lemma match
                match_found = False
                for candidate in lemma_candidates:
                    if str_distance(form_norm, candidate.lower()):
                        relem = dict_update(relem, candidate, form_norm, formdict)
                        #relem[candidate]['gender'] = gender
                        match_found = True
                        break
                # 6) slowest step (ca 27s), but sorts another ~8k tokens
                if match_found==False and lemma_norm.capitalize() in relem and lemma!='unknown':
                    # No close pre-existing lemma match was found 
                    # -> check against base lemma inflectional forms to replace matching substring
                    # with non-inflected form (=new lemma), add to current lemma candidates
                    # E.g. 'Leseköpfen' -> iter through forms of 'Kopf' -> match with 'köpfen'
                    # -> replace 'köpfen' in current form with 'kopf' -> 'Lesekopf'
                    for compare_form in relem[lemma_norm.capitalize()]['forms']:
                        if compare_form.lower() in form_norm:
                            new_lemma = (re.split(r''+compare_form.lower(),
                                                form_norm)[0]
                                        + lemma.lower()).capitalize()
                            lemma_candidates.append(new_lemma)
                            relem = dict_update(relem, new_lemma, form_norm,
                                                formdict)
                            relem[new_lemma.capitalize()]['gender'] = lemma_entry['gender']
                            match_found = True
                            break

                # 7) low edit distance: typo
                if match_found==False:
                    if (form_norm[0]==lemma_norm[0]
                        and nltk.edit_distance(form_norm[:len(lemma_norm)],
                                            lemma_norm)<=2):
                        relem = dict_update(relem, lemma_norm, form_norm, formdict)
                        match_found = True

                if match_found==False:
                    unks = dict_update(unks, lemma_norm, form_norm, formdict)

    # second run through unknown lemma items, only check for direct matches of forms
    # therein with lemma keys in relem
    drop_unks = list()
    for lemma in unks:
        for form, formdict in unks[lemma]['forms'].items():
            if form in relem:
                relem = dict_update(relem, form.lower(), form.lower(),
                                    formdict)
                drop_unks.append((lemma, form))
        # drop found forms from unknowns
            elif add_unks:
                relem = dict_update(relem, 'unknown', form.lower(),
                                    formdict)
                drop_unks.append((lemma, form))

    if add_unks:
        # add unknown entry back into DiNos
        for lemma in digits:
            for form, formdict in digits[lemma]['forms'].items():
                relem = dict_update(relem, lemma, form, formdict)

    # remove unknowns that have later been assinged to lemmas
    for lemma, form in drop_unks:
        del unks[lemma]['forms'][form]

    # Print some meta info
    if args.verbosity:
        print('\nLemmatisation failed for {} items across {} lemmas'.format(
            sum([unks[lemma]['count'] for lemma in unks]), len(unks)))
        if not add_unks:
            print('Found {} all-digit item{} across {} lemmas'.format(
                sum([digits[lemma]['count'] for lemma in digits]),
                ('s'if 1-len(digits) else''), len(digits)))

        print('Unknown/all digit items', ('kept in' if add_unks else 'dropped from'),
            'data')

    if return_unks:
        return relem, unks
    else:
        return relem


def dict_update(mydict: dict, key1: str, key2: str, subdict: dict):
    """Shorthand function to initialise & update nested dinos dict safely."""
    # Re-capitalise lemma/form
    lemma = key1.capitalize()
    form = key2.capitalize()
    # Make sure DDs structure around form entry exists, add/update values
    build_dinos_dict(mydict, lemma, form)
    mydict[lemma]['count'] += subdict['count']  # manually udpate lemma count too

    init_dict(mydict[lemma], 'forms')
    init_dict(mydict[lemma]['forms'], form)
    rec_update(mydict[lemma]['forms'][form], subdict)

    return mydict


def rec_update(iterdict: dict, subdict: dict):
    """Recursively merge-update (additive) parallel n_depth nested dicts."""
    for mykey, val in subdict.items():
        if type(val) == int:
            iterdict[mykey] = iterdict.get(mykey, 0) + val
        else:
            init_dict(iterdict, mykey)
            rec_update(iterdict[mykey], val)


def str_distance(form_check, lemma_check):
    """Naively check distance between two strings by character comparison.
    True if two strings have a maximum length distance & difference of 3."""
    if len(form_check)-len(lemma_check)<4:
        mismatch_count = 0
        for char1, char2 in zip(lemma_check, form_check):
            if char1 != char2:
                mismatch_count+=1

        if mismatch_count<4:
            return True

    return False

def meta(dinos):
    # Print some meta info to verify changes from relemmatisation 
    # nr of: unique lemmas, unique forms, forms total (=nr of NPs)
    print('\nu_lemm', 'u_forms', 'n_forms', sep='\t')
    print(len(dinos),
          len(set([w for lemma in dinos for w in list(dinos[lemma]['forms'].keys())])),
          sum([dinos[lemma]['count'] for lemma in dinos]),
          sum([form['count'] for lemma in dinos for form in dinos[lemma]['forms'].values()]),
          sep='\t')
    print()


if __name__=='__main__':
    corpus_names = args.corpora.split() if ' ' in args.corpora else args.corpora
    for corp in corpus_names:
        # get DiNos of nested feature/cooccurence frequency dictionaries for nouns and specifiers
        nps_file = config[corp]['nps'] #if not args.noisy else config[corp]['nps_noisy']
        dinos_raw = get_dinos_data(nps_file) #opt: call Nps extract if doesnt exist ig
        if args.verbosity:
            print('\nPre relemmatisation:')
            meta(dinos_raw)
        # Relemmatise dinos
        dinos_relem, _ = (re_lemmatise(dinos_raw, add_unks=args.noisy,
                                       return_unks=True)
                          if args.relemmatise else (dinos_raw, []))
        if args.verbosity:
            print('\nPost relemmatisation:')
            meta(dinos_relem)
        # Sort dinos lemmas alphabetically
        dinos_sorted = dict(sorted(dinos_relem.items()))
        dinos_file =  config[corp]['dinos'] if not args.noisy else config[corp]['dinos_noisy']
        save_data(dinos_sorted,dinos_file, overwrite=True)

        if args.create_demo:
            save_data({lemma: entry for lemma, entry in dinos_sorted.items() 
                       if lemma.startswith(args.create_demo)}, 
                      'data/{}_DiNoS_demo_{}.json'.format(corp, args.create_demo),
                      overwrite=True)

