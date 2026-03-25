## NPs ##
# script to extract nps from file, print eval & save regular nps to file
import argparse
import os

import conllu
import json 
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-corp', '--corpora', help='source corpus',
                    default='GSD HDT')
parser.add_argument('-sets', '--data_sets', help='corpus\' data sets to load',
                    default='train dev test')
parser.add_argument('-v', '--verbosity', help='amount of detail printed',
                    action='store_true', default=False)
parser.add_argument('-rfd', '--restore_from_deprel', help='guess case from deprel',
                    action='store_true', default=False)
parser.add_argument('-noi', '--noisy', help='block item pruning',
                    action='store_true', default=False)

args, unk = parser.parse_known_args()
config = json.load(open('config.json'))
corpus_names = args.corpora.split(' ')
corpus_sets = args.data_sets.split(' ')


def parse_conllu_files(files: str|list, save_to: str|bool = False) -> list:
    """Parse one or more Conllu files, optionally save parsed content of
    multiple passed files to one single file.

    Args:
        files (str | list): Source file(s) in Conllu format.
        save_to (str | bool, optional): Set whether to save parsed content
            to single Conllu file. Defaults to False.

    Raises:
        FileNotFoundError: Raised when 1+ source files cannot be found.

    Returns:
        list: Parsed content of source files.
    """

    # First verify that single file/list of files can be found
    if type(files)==str and os.path.isfile(files):
        parse_files = files

    elif type(files)==list:
        files_check = list()
        for fname in files:
            if type(fname)==str: files_check.append(fname)
            else: [files_check.append(nested_file) for nested_file in fname]
        # check that all files in list exist
        if sum([1 for fname in files_check if os.path.isfile(fname)])==len(files_check):
            parse_files = files_check
        else:
            raise FileNotFoundError ('One or more files from corpus config could not be found')

    # file existence verified: parse single conllu file, or..
    if type(parse_files)==str:
        with open(parse_files, 'r', encoding='utf-8') as f:
            print(f'Parsing {parse_files} file...')
            parsed = conllu.parse(f.read())

    else:
        # parse & merge multiple files, optional: save to one big merged_file
        parsed = list()
        for fname in parse_files:
            with open(fname, 'r', encoding='utf-8') as f:
                print(f'Parsing {fname} file...')
                parsed += conllu.parse(f.read())

    if save_to:
        save_data(parsed, save_to, overwrite=True)

    return parsed

def extract_nps(conllu_sents: conllu.SentenceList,
                dep_rm_upos: list = ['VERB', 'AUX', 'PRON', 'PUNCT', 'CCONJ',
                                     'SCONJ', 'PROPN', 'X'],
                head_rm_deprels: list = ['compound', 'amod'],
                noisy: bool = False,
                drop_irregs: bool = True,
                drop_deprecated: bool = False,
                verbose: bool = False,
                return_pruned: bool = False) -> list:
    """Returns a list of TokenList noun phrases extracted from a parsed
    Conllu file. Restriced to normal nouns (UPOS NOUN/XPOS NN). Attempts
    to restore feature annotation of nominal heads and determiners from 
    their dependents to include information on gender, case, and number.
    Takes arguments for tokens with a given upos tags to be excluded
    from dependents & tokens (NOUN/NN) with a given dependency relation
    tag to be excluded from becoming NP heads.

    Args:
        conllu_sents (conllu.SentenceList): Parsed sentences from Conllu file.
        dep_rm_upos (list, optional): _description_. Defaults to ['VERB',
        'AUX', 'PRON', 'PUNCT', 'CCONJ', 'SCONJ', 'PROPN', 'X'].
        head_rm_deprels (list, optional): _description_. Defaults to
            ['compound', 'amod'].
        noisy (bool, optional): Decides whether to prune output. On True does
            not prune XPOS=TRUNC and len(wordform)<2 tokens, overwrites
            drop_irregs/drop_deprecated. Defaults to False.
        drop_irregs (bool, optional): Decide whether to drop irregular token, 
            i.e. those not fulfilling the filter. Defaults to True.
        drop_deprecated (bool, optional): Decide whether to drop deprecated
            items, i.e. feature annotations not complete for gender/case/
            number. Defaults to False.
        verbose (bool, optional): On True, prints information about data and
            data handling. Defaults to False.
        return_pruned (bool, optional):  On True, return list of pruned 
            (non-restorable) NPs. Defaults to False.

    Returns:
        list: List of NPs extacted from file.
    """
    print('Extract NPs from corpus...')
    # Store NPs
    pruned_nps, nps = list(), list()
    # Store counts regarding feature annotation completeness & restoration
    tracker = {categ: {'nn': 0, 'det': 0, 'apprart': 0}
               for categ in ['compl', 'incompl', 'restored']}
    tracker['dropped'] = 0

    for sent in tqdm(conllu_sents[:]):
        # on sentence level: gather tokens with odd tuple ids (apprart)
        # add 0 as status: update to 1 once apprart is successfully added to np
        ambig = [[tok, 0] for tok in sent if type(tok['id'])==tuple]

        # iterate over each nouns in sentence
        # strict regular noun filtering, and needs to be suited as phrase head
        for noun in sent.filter(upos='NOUN', xpos='NN',
                                deprel=lambda x: x not in head_rm_deprels):
            # get np: noun + all depdendents in grammatical agreement
            dependents = list()
            for tok in sent:
                # token belongs to current head & has eligible UPOS tag (e.g. not a verb)
                if tok['head']==noun['id'] and tok['upos'] not in dep_rm_upos:
                    # if token is a NOUN -> more strict criteria:
                    if tok['upos']=='NOUN':
                        if tok['deprel'] in head_rm_deprels:
                            if noisy:
                                dependents.append(tok)
                            # for noun toks: must be:
                            # xpos NOT truncation, 
                            # deprel IS amod/compound
                            # len IS more than 1 character
                            elif (tok['xpos']!='TRUNC'
                                and len(tok['form'])>1):
                                 dependents.append(tok)
                    else:
                        dependents.append(tok) 
            # Join nominal head + dependents as np
            np = [noun] + dependents

            # check if np head feature annotation meets gold standard, i.e.
            # fully specified for gender/case/number
            if np[0]['feats']!=None and check_feat_annotation(np[0]['feats']):
                tracker['compl']['nn']+=1
            else:
                # log as incomplete in corpus; attempt to restore annotation.
                tracker['incompl']['nn']+=1
                restore_N_head_annotation(np)
                # if np head now fulfills gold standard, log as restored.
                if check_feat_annotation(np[0]['feats']):
                    tracker['restored']['nn']+=1


            # similarly: check feature annotation on DET/ART
            for tok in np:
                det_idx = False
                if tok['upos']=='DET' and tok['xpos']=='ART':
                    det_idx = np.index(tok)
                    if check_feat_annotation(tok['feats']):
                        # eature annotation on det complete
                        tracker['compl']['det']+=1
                    else:
                        # annotation incomplete, attempt to restore
                        tracker['incompl']['det']+=1
                        restore_DET_annotation(np, det_idx)
                        if check_feat_annotation(tok['feats']):
                            tracker['restored']['det']+=1


            # try to infer case from deprel, only done when unambiguous
            # i.e. nsubj -> Nom, obj -> Acc
            if args.restore_from_deprel:
                missing_det_case = False
                try:
                    if det_idx and np[det_idx]['feats']['Case'] == None:
                        missing_det_case = True
                except:
                    pass

                if np[0]['feats']['Case']==None :
                    if np[0]['deprel'] == 'nsubj':
                        np[0]['feats']['Case'] = 'Nom'
                        if det_idx and missing_det_case:
                            np[det_idx]['feats']['Case'] = 'Nom'
                    elif np[0]['deprel'] == 'obj':
                        np[0]['feats']['Case'] = 'Acc'
                        if det_idx and missing_det_case:
                            np[det_idx]['feats']['Case'] = 'Acc'
                    else:
                        ## TBD any way to restore more cases? - unlikely
                        pass

                    # log for evaluation
                    if check_feat_annotation(np[0]):
                        tracker['restored']['nn']+=1
                    if det_idx and missing_det_case:
                        if check_feat_annotation(np[det_idx]):
                            tracker['restored']['det']+=1

            # see if sentence has unassigned apprart items left (ambig)
            # & if they belong to the current np; skip once all statuses = 1
            if ambig and sum([status for tok, status in ambig]) != len(ambig):
                for i, (ambig_tok, status) in enumerate(ambig):
                    if (status == 0
                        and (ambig_tok['id'][0] in [tok['id'] for tok in np]) 
                        and (ambig_tok['id'][2] in [tok['id'] for tok in np])):  #id format '2-3'
                            # update status of ambiguous token: origin np found
                            ambig[i][1] = 1
                            # create new annotation for apprart, drop adp+det toks
                            np = rebuild_apprart(ambig_tok, np)
                            tracker['restored']['apprart'] += 1

            ## Compile list of NPs to return ##
            # decide which NPs to keep depending on keyword & annotation status
            if (not drop_irregs and not drop_deprecated) or noisy:
                # keep all
                nps.append(sort_np(np))

            elif drop_irregs:
                # double check against filters/minimum length
                if (np[0]['xpos']=='NN'
                    and np[0]['deprel'] not in head_rm_deprels
                    and len(np[0]['form'])>1
                    ):
                    # check feature annotation quality
                    if drop_deprecated:
                        if check_feat_annotation(np[0]['feats'])==True:
                            nps.append(sort_np(np))
                        else:
                            tracker['dropped'] += 1
                            pruned_nps.append(sort_np(np))
                    else:
                        nps.append(sort_np(np))
                else:
                    tracker['dropped'] += 1
                    pruned_nps.append(sort_np(np))
            # Keep irregular items, but only those with complete feature anno
            elif drop_deprecated and check_feat_annotation(np[0]['feats'])==True:
                nps.append(sort_np(np))
            # deprecated to be dropped
            else:
                tracker['dropped'] += 1
                pruned_nps.append(sort_np(np))

    # print meta info and return nps
    if verbose:
        print_restore_eval(tracker, drop_irregs, drop_deprecated)

    print('_'*80)
    print('Returned NPs:\n\t', len(nps),
          as_percent(len(nps),(tracker['compl']['nn']+tracker['incompl']['nn'])))

    if return_pruned:
        return nps, pruned_nps
    else:
        return nps


def check_feat_annotation(tok_feats: dict) -> bool:
    """Checks if feature dict has annotation for gender, number, and case."""
    if ((sum([1 if (fkey in tok_feats.keys() and tok_feats[fkey] != None)
              else 0 for fkey in ['Gender', 'Number', 'Case']])
         == 3)):
        return True
    else:
        return False


def restore_N_head_annotation(np: list[conllu.Token]):
    """Attempt to restore feature annotation of noun head (first item in
    list) using information on head's dependents concerning gender, case,
    and number, happens in-place.

    Args:
        np: List of conllu.models.Token comprising a noun phrase.

    """
    # List of dict keys wanted in np head annotation.
    goal = ['Gender', 'Number', 'Case']
    # if np has no features at all: create empty dict
    if np[0]['feats'] == None: np[0]['feats'] = dict()

    # create underspecified annotation for missing features
    for fkey in goal:
        if fkey not in np[0]['feats'].keys():
            np[0]['feats'][fkey] = None  # alt: '_'

    # try to restore annotation from dependents
    for tok in np[1:]:
        # get list of missing annotation keys in loop to avoid conflicts
        # during iteration; updated whenever all features of a single dependent
        # have been checked.
        missing_keys = [fkey for fkey in goal if np[0]['feats'][fkey] == None]
        # check if missing feature is annotated in dependents, add to head feats.
        for fkey in missing_keys:
            if (tok['upos'] != 'NOUN' and tok['feats'] and fkey in tok['feats']):
                np[0]['feats'][fkey] = tok['feats'][fkey]


def restore_DET_annotation(np: list[conllu.Token], det_idx: int):
    """Attempt to restore feature annotation of determiner in NP using
    information from np head's annotation concerning gender, case,
    and number. NP head has previously inherited all possible relevant
    annotations from other dependents. Happens in-place.

    Args:
        np (list[conllu.Token]): NP, list of Conllu tokens.
        det_idx (int): Index position of NPs determiner.
    """
    for feat in ['Gender', 'Case', 'Number']:
        if feat not in np[det_idx]['feats'] or np[det_idx]['feats'][feat]==None:
            if np[0]['feats'][feat] != None:  # inherit from noun/dependents
                np[det_idx]['feats'][feat] = np[0]['feats'][feat]


def rebuild_apprart(apprart: conllu.Token, np: list) -> list:
    """Rebuild a fused APPRART token from an ADP and DET token,
    inheriting their features and replacing them.

    Args:
        apprart: APPRART token with tuple id and form annotation.
        np: Noun phrase (list of tokens) APPRART originally belonged to.

    Returns:
        list: List of tokens, NP with former DET and ADP replaced by APPRART.
    """
    head = np[0]
    old_id = apprart['id']  # save old id

    apprart['id'] = old_id[0]
    # apprart['form'] = ...  # keep original form
    apprart['lemma'] = None  # ?
    apprart['upos'] = None  # ?
    apprart['xpos'] = 'APPRART'
    apprart['feats'] = dict()  # inherit from det/adp
    apprart['head'] = head['id']
    # apprart['deprel'] = ...  # case or det? ? indef
    # apprart['deps'] = ...  # keep as None?
    # apprart['misc'] = ...  # keep as None?

    # inherit features from adp+det
    for tok in np:
        # in GSD only one of the divided forms is annotated, in HDT annotation
        # is split across both tokens
        tok['feats'] = tok.get('feats', dict())
        if tok['id'] in [old_id[0], old_id[2]] and tok['feats']:
            for feat_key, value in tok['feats'].items():
                apprart['feats'][feat_key] = value
    # drop adp+det from np and replace with apprart; keep head in initial
    # position to be indexable at [0].
    rebuild_np = ([head, apprart] + 
                  [tok for tok in np if tok['id'] not in [old_id[0], old_id[2],
                                                          head['id']]])
    return rebuild_np


def sort_np(np: list) -> conllu.TokenList:
    """Returns NP as TokenList with tokens in order of original sentence."""
    return conllu.TokenList(sorted(np, key=lambda tok: tok['id']))


def print_restore_eval(tracker: dict, drop_irregs: bool = False,
                       drop_deprecated: bool = False):
    # Print meta info; mainly strings and simple computation with tracker
    # logs; other keywords determine info printed
    nn_total = tracker['compl']['nn']+tracker['incompl']['nn']
    det_total = tracker['compl']['det']+tracker['incompl']['det']

    print('_'*80)
    print('Extracted NPs total:\t', nn_total,
            '\t\tIncluded DETs:\t', det_total)
    print('_'*80)
    print(f'Feature annotation in raw data on...',
        '\nNP heads\t\t\t\tDETs'
        '\n  complete:'.ljust(32), tracker['compl']['nn'],
        as_percent(tracker['compl']['nn'], nn_total).ljust(20),
        'complete:'.ljust(12), tracker['compl']['det'],
        as_percent(tracker['compl']['det'], det_total),
        '\n  incomplete:'.ljust(15), tracker['incompl']['nn'],
        as_percent(tracker['incompl']['nn'], nn_total).ljust(20),
        'incomplete:'.ljust(12), tracker['incompl']['det'],
        as_percent(tracker['incompl']['det'], det_total))
    print('_'*80)
    print(f'Feature annotation restoration using dependents on incomplete... ',
        '\nNP heads\t\t\t\tDETs'
        '\n  success:'.ljust(32), tracker['restored']['nn'],
        as_percent(tracker['restored']['nn'], tracker['incompl']['nn']).ljust(22),
        'success:'.ljust(12), tracker['restored']['det'],
        as_percent(tracker['restored']['det'], tracker['incompl']['det']),
        '\n  failure:'.ljust(15),
        tracker['incompl']['nn']-tracker['restored']['nn'],
        as_percent((tracker['incompl']['nn']-tracker['restored']['nn']),
                    tracker['incompl']['nn']).ljust(20),
        'failure:'.ljust(12),
        tracker['incompl']['det']-tracker['restored']['det'],
        as_percent((tracker['incompl']['det']-tracker['restored']['det']),
                    tracker['incompl']['det']))
    print('_'*80)

    if drop_irregs and  drop_deprecated:
        print('Dropped NPs with len(head)<2 and non-restorable annotation.')
    elif drop_irregs and not drop_deprecated:
        print('Dropped NPs with len(head)<2. Returned Complete, restored and non-restorable NPs.')
    elif not drop_irregs and drop_deprecated:
        print('Dropped NPs with non-restorable feature annotation.')
    else:
        print('Returned all NPs. Complete, restored and non-restorable.')
    print('_'*80)
    print('Reconstructed APPRARTS:\t', tracker['restored']['apprart'])
    print('Nps dropped due to insufficient tags/len/non-alpha:\t', tracker['dropped'],
          as_percent(tracker['dropped'], nn_total))


def as_percent(a: int|float, b: int|float) -> str:
    """Returns result of a/b as percentage in f-string format."""
    return f'({a/b:.2%})'


def save_data(data: list|dict, fname: str, overwrite: bool = False):
    """Saves data to conllu or json file."""
    if os.path.isfile(fname) and overwrite==False:
        if input(f'File {fname} already exists, overwrite file? (Y/n) ') == 'Y':
            overwrite=True

    if not os.path.isfile(fname) or overwrite==True:
        if fname.endswith('conllu'):
            with open(fname, 'w') as f:
                f.writelines([entry.serialize() for entry in data])
            print(f'\nSaved data to {fname}\n')

        elif fname.endswith('json'):
            with open(fname, 'w', encoding='utf8') as f:
                json.dump(data, f, ensure_ascii=False)
            print(f'\nSaved data to {fname}\n')

    elif not fname.endswith('conllu') and not fname.endswith('json'):
        raise ValueError('Unable to save file, specify filename with either conllu or json ending')
    else:
        pass

## not actively in use; call on noisy nps to check upos NOUN !& xpos NN items
def inspect_nn(nps, return_reg_nps=False):
    truncs, ne, unk = list(), list(), list()
    for np in nps:
        for tok in np:
            if tok['upos'] == 'NOUN' and not tok['xpos'] == 'NN':
                if tok['xpos'] == 'TRUNC': truncs.append(tok)
                elif tok['xpos'] == 'NE': ne.append(tok)
                else: unk.append(tok)

    print('Fringe cases where upos==NOUN & xpos!=NN, TBD')
    print('1. trunc:\t', len(truncs))
    print('2. ? NE:\t', len(ne))
    print('4. ? unk:\t', len(unk))
    print('-'*20)
    print(f'total\t{sum([len(truncs), len(ne), len(unk)])}')

    if return_reg_nps:
        new_nps = [conllu.TokenList([tok for tok in np]) for np in nps
            if np[np.index([tok for tok in np
                            if tok['upos']=='NOUN'][0])]['upos']=='NOUN'
            and np[np.index([tok for tok in np
                                if tok['upos']=='NOUN'][0])]['xpos']=='NN']
        print('\nRegular NPs with head upos=Noun & xpos=NN returned:', len(new_nps))
        return new_nps
    else:
        return ne


if __name__=="__main__":
    for corpus in corpus_names:
        # load all corpus files, parse & save merged version to one big file
        load_files = [config[corpus][dataset] for dataset in corpus_sets]
        save_file = config[corpus]['complete'] if len(corpus_sets)==3 else (
            f'../{corpus}_compl_{"".join(corpus_sets)}.conllu')
        data = parse_conllu_files(load_files, save_to=save_file)
        print(f'Number of extracted sentences from {corpus}:', len(data))

        # Get NPs, optional: print info, save to file 
        # TBD: drop non-restorable? drop irrg UPOS/XPOS?
        nps = extract_nps(data, verbose=args.verbosity, drop_irregs=True, noisy=args.noisy)
        if args.noisy:
            np_file = config[corpus]['nps_noisy']
        else:
            np_file = config[corpus]['nps'] if len(corpus_sets)==3 else (
                f'../{corpus}_nps_{"".join(corpus_sets)}.conllu')
        save_data(nps, np_file, overwrite=True)

    ## to view amount of token with upos NOUN but NOT xpos NN: ##
        # on eligible NP heads: config[corpus]['nps_noisy']; in corpus total: config[corpus]['complete']
        # npdata = parse_conllu_files(config[corpus]['nps_noisy'])
        # inspect_nn(npdata)

