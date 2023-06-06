'''
Adapted from: https://github.com/Eleanor-H/DAGN
'''

from transformers import RobertaTokenizer
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer("english")
from svo_extraction import SVOsextractor

def token_stem(token):
    return stemmer.stem(token)


def arg_tokenizer(text_a, text_b, tokenizer, stopwords, relations:dict, punctuations:list,
                  max_gram:int, max_length:int, do_lower_case:bool=False):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _is_stopwords(word, stopwords):
        if word in stopwords:
            is_stopwords_flag = True
        else:
            is_stopwords_flag = False
        return is_stopwords_flag

    def _head_tail_is_stopwords(span, stopwords):
        if span[0] in stopwords or span[-1] in stopwords:
            return True
        else:
            return False

    def _with_septoken(ngram, tokenizer):
        if tokenizer.bos_token in ngram or tokenizer.sep_token in ngram or tokenizer.eos_token in ngram:
            flag = True
        else:
            flag = False
        return flag

    def _is_argument_words(seq, argument_words):
        pattern = None
        arg_words = list(argument_words.keys())
        if seq.strip() in arg_words:
            pattern = argument_words[seq.strip()]
        return pattern

    def _is_exist(exists:list, start:int, end:int):
        flag = False
        for estart, eend in exists:
            if estart <= start and eend >= end:
                flag = True
                break
        return flag

    def _find_punct(tokens, punctuations):
        punct_ids = [0] * len(tokens)
        for i, token in enumerate(tokens):
            if token in punctuations:
                punct_ids[i] = 1
        return punct_ids

    def is_uselessword(token_list):
        arg_words = list(argument_words.keys())
        for t in token_list:
            if _is_stopwords(t,stopwords):
                return True
            if t in arg_words:
                return True
            else:
                return False


    def idf_ngram(token_list, max_n=3):
        idfs_ngram = {}
        stemmed_tokens = [token_stem(token) for token in token_list]
        n_tokens = len(token_list)
        for n in range(max_n, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.
                window_start, window_end = i, i + n
                stemmed_span = stemmed_tokens[window_start:window_end]
                stemmed_ngram = " ".join(stemmed_span)
                orin_span = token_list[window_start:window_end]
                orin_ngram = " ".join(orin_span)

                if is_uselessword(stemmed_span): continue

                if orin_ngram in idfs_ngram:
                    idfs_ngram[orin_ngram] += 1
                if not orin_ngram in idfs_ngram:
                    idfs_ngram[orin_ngram] = 1


        idfs_ngram = sorted(idfs_ngram.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
        key_tokens = [i[0] for i in idfs_ngram]
        key_tokens_ = []
        for t in key_tokens:
            else_span = key_tokens
            else_span.remove(t)
            else_gram = " ".join(else_span)
            if not t in else_gram:
                key_tokens_.append(t)
        return idfs_ngram, key_tokens_[:5]

    def _find_arg_ngrams(tokens, max_gram):
        '''
        return:
            argument_words: dict, key = ngrams, value =(window_start, window_end)
            argument_ids: list, value = {0,1} 1 means argument word
        '''
        n_tokens = len(tokens)
        global_arg_start_end = []
        argument_words = {}
        argument_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.
                window_start, window_end = i, i + n
                ngram = " ".join(tokens[window_start:window_end])
                pattern = _is_argument_words(ngram, relations)
                if pattern:
                    if not _is_exist(global_arg_start_end, window_start, window_end):
                        global_arg_start_end.append((window_start, window_end))
                        argument_ids[window_start:window_end] = [pattern] * (window_end - window_start)
                        argument_words[ngram] = (window_start, window_end)

        return argument_words, argument_ids

    def _find_dom_ngrams_2(tokens, max_gram): 
        stemmed_tokens = [token_stem(token) for token in tokens]

        ''' 1 & 2'''
        n_tokens = len(tokens)
        d_ngram = {}
        domain_words_stemmed = {}
        domain_words_orin = {}
        domain_ids = [0] * n_tokens

        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.

                window_start, window_end = i, i+n
                stemmed_span = stemmed_tokens[window_start:window_end]
                stemmed_ngram = " ".join(stemmed_span)
                orin_span = tokens[window_start:window_end]
                orin_ngram = " ".join(orin_span)

                if is_uselessword(stemmed_span): continue
                if not stemmed_ngram in d_ngram:
                    d_ngram[stemmed_ngram] = []
                d_ngram[stemmed_ngram].append((window_start, window_end))

        ''' 3 '''
        d_ngram = dict(filter(lambda e: len(e[1]) > 1, d_ngram.items())) 
        raw_domain_words = list(d_ngram.keys())
        raw_domain_words.sort(key=lambda s: len(s), reverse=True) 
        domain_words_to_remove = []
        for i in range(0, len(d_ngram)):
            for j in range(i+1, len(d_ngram)):
                if raw_domain_words[j] in raw_domain_words[i]:
                    domain_words_to_remove.append(raw_domain_words[j])
        for r in domain_words_to_remove:
            try:
                del d_ngram[r]
            except:
                pass
        ''' 4 '''
        d_id = 0
        for stemmed_ngram, start_end_list in d_ngram.items():
            d_id += 1
            for start, end in start_end_list:
                domain_ids[start:end] = [d_id] * (end - start)
                rebuilt_orin_ngram = " ".join(tokens[start: end]) 
                if not stemmed_ngram in domain_words_stemmed:
                    domain_words_stemmed[stemmed_ngram] = []
                if not rebuilt_orin_ngram in domain_words_orin:
                    domain_words_orin[rebuilt_orin_ngram] = []
                domain_words_stemmed[stemmed_ngram] += [(start, end)]
                domain_words_orin[rebuilt_orin_ngram] += [(start, end)]


        return domain_words_stemmed, domain_words_orin, domain_ids




    ''' start '''
    bpe_tokens_a = tokenizer.tokenize(text_a)
    bpe_tokens_b = tokenizer.tokenize(text_b)
    bpe_tokens = [tokenizer.bos_token] + bpe_tokens_a + [tokenizer.sep_token]*2 + \
                 bpe_tokens_b + [tokenizer.eos_token]
    a_mask = [1] * (len(bpe_tokens_a) + 2) + [0] * (max_length - (len(bpe_tokens_a) + 2))
    b_mask = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 2) + [0] * (max_length - len(bpe_tokens))
    a_mask = a_mask[:max_length]
    b_mask = b_mask[:max_length]
    assert len(a_mask) == max_length, 'len_a_mask={}, max_len={}'.format(len(a_mask), max_length)
    assert len(b_mask) == max_length, 'len_b_mask={}, max_len={}'.format(len(b_mask), max_length)

    assert isinstance(bpe_tokens, list)
    bare_tokens = []
    for token in bpe_tokens:
        if token is not None:
            if "Ġ" in token:
                bare_tokens.append(token[1:])
            else:
                bare_tokens.append(token)
    argument_words, argument_space_ids = _find_arg_ngrams(bare_tokens, max_gram=max_gram)
    domain_words_stemmed, domain_words_orin, domain_space_ids = _find_dom_ngrams_2(bare_tokens, max_gram=max_gram)
    punct_space_ids = _find_punct(bare_tokens, punctuations)  

    argument_bpe_ids = argument_space_ids
    domain_bpe_ids = domain_space_ids 
    punct_bpe_ids = punct_space_ids 

    ''' output items '''
    input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)

    padding = [0] * (max_length - len(input_ids))
    padding_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids))
    arg_dom_padding_ids = [-1] * (max_length - len(input_ids))
    input_ids += padding_ids
    argument_bpe_ids += arg_dom_padding_ids  #[0,0,0,4,4,0,4,4,4,0,....-1,-1,...]
    domain_bpe_ids += arg_dom_padding_ids  #[0,0,1,1,1,2,2,3,3,3,3,1,1,1,....-1,-1,-1....]
    punct_bpe_ids += arg_dom_padding_ids  #[0,0,0,0,0....1,0,0,...,1,0,0,.....-1,-1,-1,-1...]
    input_mask += padding   #[1,1,1,1,...0,0,0....]
    segment_ids += padding #[0,0,0,0,....1,1,1,....0,0,0,...]

    input_ids = input_ids[:max_length]
    input_mask = input_mask[:max_length]
    segment_ids = segment_ids[:max_length]
    argument_bpe_ids = argument_bpe_ids[:max_length]
    domain_bpe_ids = domain_bpe_ids[:max_length]
    punct_bpe_ids = punct_bpe_ids[:max_length]

    assert len(input_ids) <= max_length, 'len_input_ids={}, max_length={}'.format(len(input_ids), max_length)
    assert len(input_mask) <= max_length, 'len_input_mask={}, max_length={}'.format(len(input_mask), max_length)
    assert len(segment_ids) <= max_length, 'len_segment_ids={}, max_length={}'.format(len(segment_ids), max_length)
    assert len(argument_bpe_ids) <= max_length, 'len_argument_bpe_ids={}, max_length={}'.format(
        len(argument_bpe_ids), max_length)
    assert len(domain_bpe_ids) <= max_length, 'len_domain_bpe_ids={}, max_length={}'.format(
        len(domain_bpe_ids), max_length)
    assert len(punct_bpe_ids) <= max_length, 'len_punct_bpe_ids={}, max_length={}'.format(
        len(punct_bpe_ids), max_length)

    #idfs, key_words_ = idf_ngram(bare_tokens)
    keywords_tokens_list = list(domain_words_orin.values())
    keytokens_ids = []
    for i in keywords_tokens_list:
        for (start, end) in i:
            tokens = input_ids[start:end]
            if not tokens in keytokens_ids and len(tokens) > 0:
                keytokens_ids.append(tokens)

    # print(domain_words_orin.keys())

    svo_token_ids = SVOsextractor(text_a,text_b,tokenizer)
    output = {}
    output["input_tokens"] = bpe_tokens
    output["input_ids"] = input_ids
    output["attention_mask"] = input_mask
    output["token_type_ids"] = segment_ids
    output["argument_bpe_ids"] = argument_bpe_ids
    output["domain_bpe_ids"] = domain_bpe_ids
    output["punct_bpe_ids"] = punct_bpe_ids
    output["a_mask"] = a_mask
    output["b_mask"] = b_mask
    output["keywords_ids"] = keytokens_ids
    output["SVOlist"] = svo_token_ids

    return output

def main(text, option):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    stopwords = list(gensim.parsing.preprocessing.STOPWORDS) + punctuations

    inputs = arg_tokenizer(text, option, tokenizer, stopwords, relations, punctuations, 5, 256)


    ''' print '''
    p = []
    for token, arg, dom, pun in zip(inputs["input_tokens"], inputs["argument_bpe_ids"], inputs["domain_bpe_ids"],
                                    inputs["punct_bpe_ids"]):
        p.append((token, arg, dom, pun))
    print(p)
    print('input_tokens\n{}'.format(inputs["input_tokens"]))
    print('input_ids\n{}, size={}'.format(inputs["input_ids"], len(inputs["input_ids"])))
    print('attention_mask\n{}'.format(inputs["attention_mask"]))
    print('token_type_ids\n{}'.format(inputs["token_type_ids"]))
    print('argument_bpe_ids\n{}'.format(inputs["argument_bpe_ids"]))
    print('domain_bpe_ids\n{}, size={}'.format(inputs["domain_bpe_ids"], len(inputs["domain_bpe_ids"])))
    print('punct_bpe_ids\n{}'.format(inputs["punct_bpe_ids"]))
    print('keywords_ids\n{}'.format(inputs['keywords_ids']))
    # print('idfs\n{}'.format(inputs['idfs']))


if __name__ == '__main__':

    import json
    from graph_building_blocks.argument_set_punctuation_v4 import punctuations
    with open('./graph_building_blocks/explicit_arg_set_v4.json', 'r') as f:
        relations = json.load(f)  # key: relations, value: ignore

    text = "Outsourcing is the practice of obtaining from an independent supplier a product or service that a company has previously" \
           " provided for itself. Vernon, Inc. , a small manufacturing company that has in recent years experienced a decline in its profits, " \
           "plans to boost its profits by outsourcing those parts of its business that independent suppliers can provide at lower cost than " \
           "Vernon can itself."
    option = "Vernon plans to select the independent suppliers it will use on the basis of submitted bids."


    main(text, option)
