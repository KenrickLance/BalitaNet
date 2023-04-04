import json

from collections import Sequence

from django.conf import settings

from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    StoppingCriteriaList,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    MaxLengthCriteria,
    BeamSearchScorer,
    NoBadWordsLogitsProcessor,
    InfNanRemoveLogitsProcessor
)

from api.ML import dataset
from api.apps import ApiConfig

with open(f'{ApiConfig.checkpoint_name}/vocab.json', 'r', encoding='utf-8') as f:
  encoder = json.load(f)
with open(f'{ApiConfig.checkpoint_name}/added_tokens.json', 'r', encoding='utf-8') as f:
  added_tokens = json.load(f)
  added_tokens['<|endoftext|>'] = 0
  added_tokens_decoder = {v:k for k,v in added_tokens.items()}
decoder = {v:k for k,v in encoder.items()}

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

def get_utf_type(num):
  binary = str(bin(num))[2:]
  if len(binary) != 8:
    return 0
  if binary[:2] == '10':
    return 1
  if binary[:3] == '110':
    return 2
  if binary[:4] == '1110':
    return 3
  if binary[:5] == '11110':
    return 4
  return 0

def makeChar2Token(output):
    tokens = output.sequences[0].tolist()
    char2token = []
    skip = 0
    remark = None
    for i, token in enumerate(tokens):
        if token in added_tokens_decoder.keys():
            char2token.append({'char': added_tokens_decoder[token], 'tokens': [i], 'remark': 'special'})
            if added_tokens_decoder[token] == '<|title|>':
                remark = 'title'
            if added_tokens_decoder[token] == '<|body|>':
                remark = 'body'
            if added_tokens_decoder[token] == '<|category|>':
                remark = 'category'
            # print(added_tokens_decoder[token], end='')
            continue
        byte_string = decoder[token]
        for j, char in enumerate(byte_string):
            if skip:
                skip = skip - 1
                continue
            encoded_char = byte_decoder[char]
            utf_type = get_utf_type(encoded_char)
            if utf_type == 0:
                char = bytearray([encoded_char]).decode('utf-8')
                char2token.append({'char': char, 'tokens': [i], 'remark': remark})
                # print(char, end='')
            else:
                # depending on type: get the rest of the chars in the utf sequence

                # if continuation byte, handle error
                if utf_type == 1:
                    #char2token.append({'char': '�', 'tokens': [i]})
                    print("UTF ERROR")
                    continue

                # get rest of chars from the next three tokens plus this token, stop when you see a special token
                complete_utf_sequence = [encoded_char]
                four_tokens_old = tokens[i:i + 4]
                four_tokens = []
                for token in four_tokens_old:
                    if token in added_tokens_decoder.keys():
                        break
                    four_tokens.append(token)

                # get the corresponding tokens for this utf sequence
                four_tokens_lengths = [len(decoder[token]) for token in four_tokens]
                tokens_of_utf = []
                counter = 0
                for k, val in enumerate(four_tokens_lengths):
                    if counter < utf_type:
                        tokens_of_utf.append(i + k)
                        counter += val

                # if sequence is valid utf-8: decode it and skip the sequence
                # else: handle error (replace or ignore) then proceed to next char as normal
                text = ''.join([decoder[token] for token in four_tokens])[:utf_type]
                try:
                    text = bytearray([byte_decoder[c] for c in text]).decode('utf-8')
                    # print(text, end='')
                    char2token.append({'char': text, 'tokens': tokens_of_utf, 'remark': remark})
                    skip = utf_type - 1
                except UnicodeDecodeError:
                    print("UTF ERROR")

    for i, val in enumerate(char2token):
        if val['char'] == '<|body|>':
            body_ind = i
    try:
        char2token.insert(body_ind, {'char': '<|image|>', 'remark': 'image', 'tokens': [-1]})
    except NameError:
        pass

    return char2token

def makeScores(output, limit=10):
    scores = []
    for token in output.scores:
        values, indices = torch.nn.functional.softmax(token[0], dim=-1).sort(descending=True)
        scores.append({"values": values[:limit].tolist(), "indices": indices[:limit].tolist()})

    return scores

def recursive_map(seq, func):
    for item in seq:
        if isinstance(item, Sequence):
            yield type(item)(recursive_map(item, func))
        else:
            yield func(item)

def makeAttentions(output, k=20, image=True):
    attentions = []
    for token in output.attentions:
        all = torch.stack(token, 0)
        if all.size(3) != 1:
            all = torch.squeeze(all[:, :, :, -1, :])
        else:
            all = torch.squeeze(torch.squeeze(all, 3), 1)

        if image:
            image_avg = torch.sum(all[:, :, :49], dim=-1, keepdim=True)
            all = all[:, :, 49:]
            all = torch.cat([image_avg, all], dim=-1)

        real_k = min(k, all.size(2))
        layer_means = torch.mean(all, 0)
        head_means = torch.mean(all, 1)
        all_means = torch.mean(head_means, 0)

        layer_val, layer_idx = torch.topk(layer_means, real_k, -1)
        head_val, head_idx = torch.topk(head_means, real_k, -1)
        all_val, all_idx = torch.topk(all_means, real_k, -1)

        attentions.append(
            {
                "layer": {
                    "val": list(recursive_map(layer_val.tolist(), lambda x: round(x, 2))),
                    "idx": list(recursive_map((layer_idx - 1).tolist(), lambda x: round(x, 2))),
                },
                "head": {
                    "val": list(recursive_map(head_val.tolist(), lambda x: round(x, 2))),
                    "idx": list(recursive_map((head_idx - 1).tolist(), lambda x: round(x, 2))),
                },
                "all": {
                    "val": list(recursive_map(all_val.tolist(), lambda x: round(x, 2))),
                    "idx": list(recursive_map((all_idx - 1).tolist(), lambda x: round(x, 2))),
                }
            }
        )
    return attentions

def generateArticle(input_prompt, image=None, config={}):
    model = ApiConfig.model
    tokenizer = ApiConfig.tokenizer

    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(settings.PYTORCH_DEVICE)


    model_kwargs = {}
    if image is not None:
        model_kwargs["ctx"] = torch.unsqueeze(dataset.process_image(image), 0).to(settings.PYTORCH_DEVICE)

    stopping_criteria = StoppingCriteriaList(
        [MaxLengthCriteria(300)]
    )
    logits_processor = LogitsProcessorList(
        [InfNanRemoveLogitsProcessor(),
         TopKLogitsWarper(100),
         TopPLogitsWarper(.95),
         RepetitionPenaltyLogitsProcessor(1.2),
         #MinLengthLogitsProcessor(175, model.config.eos_token_id),
         InfNanRemoveLogitsProcessor()]
    )

    logits_warper = LogitsProcessorList(
        [InfNanRemoveLogitsProcessor(),
         TemperatureLogitsWarper(float(config.get('temperature', .8))),
         InfNanRemoveLogitsProcessor()]
    )
    with torch.no_grad():
        output = model.sample(input_ids,
                              logits_processor=logits_processor,
                              stopping_criteria=stopping_criteria,
                              logits_warper=logits_warper,
                              output_attentions=True,
                              output_scores=True,
                              return_dict_in_generate=True,
                              **model_kwargs)
    #text = tokenizer.decode(output.sequences[0])
    return output, input_ids

def makeGenerationResponse(output, input_ids):
    return {'char2token': makeChar2Token(output),
            'scores': makeScores(output),
            'attentions': makeAttentions(output),
            'inputLength': len(input_ids[0])}

def makeGenerationResponseMobile(output, input_ids):
    return {'char2token': makeChar2Token(output),
            'inputLength': len(input_ids[0])}