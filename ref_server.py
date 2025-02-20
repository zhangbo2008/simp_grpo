
import json, os, shutil, re, random, io
import torch

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b))

def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()

def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

if __name__ == '__main__':   
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn as nn

    from bottle import request
    import bottle, threading, queue
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    model_path = "/data2/Qwen/Qwen2.5-7B"

    ref_model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    ref_model.eval()
    ref_model.requires_grad_(False)

    def get_per_token_logps(input_ids):# 计算整个句子的每个位置的生成概率.
        logits = ref_model(input_ids).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred # 去除最后一个eos的概率.
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):#这样遍历每一个tokne的生成概率.
            log_probs = logits_row.log_softmax(dim=-1)#概率归一化.在vocab_size维度上.
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1) #然后收集我们句子token的概率. 整体计算跟dpo算句子的概率完全一样.
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps) # 输出,  [seq_len,1]

    raw_queue = queue.Queue()
    result_queue = queue.Queue()

    app = bottle.Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
        data = {'base': json.loads(dd[0])} 
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        raw_queue.put(data)
        print('receive', data['inputs'].shape, data['rewards'])

    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty(): return b'empty'
        return result_queue.get()

    def run_server(): bottle.run(app, host='0.0.0.0', port=59875, server='tornado')
    threading.Thread(target=run_server, daemon=False).start()

    while True:
        d = raw_queue.get() # 循环进行处理, 只要queue里面有东西进来, 就进行处理. 处理完的放入result_queue
        prompt_length = d['base']['plen']
        with torch.inference_mode():
            per_token_logps = get_per_token_logps(d['inputs'].to(ref_model.device))
        per_token_logps = per_token_logps[:,prompt_length-1:] # 注意这里面的生成概率我们只计算我们生成句子的概率, 不计算prompt部分. 假设prompt 有2个token, 那么prompt的索引是0和1,我们需要计算的是从2,3,4....之后的每个token的概率.所以
        xdata = make_bytes_list([json.dumps(d['base']).encode(), 
                                 tensor_to_bytes(d['inputs']), 
                                 tensor_to_bytes(d['rewards']),
                                 tensor_to_bytes(per_token_logps)])
        result_queue.put(xdata) # 返回数据是 提示词长度,  问答句子,  奖励分数,  回答的每个token概率.

    