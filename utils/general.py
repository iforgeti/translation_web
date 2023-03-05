from torch import nn
import torch
import torchtext
from utils.structure import  Encoder, Decoder,Seq2SeqPackedAttention,Attention
import pickle
from pythainlp import word_tokenize
# models architect



def Load_model(device = torch.device('cpu'), save_path = "models/Seq2SeqPackedAttention.pt", params_path = "models/params.pt"):

    params = torch.load(params_path)

    attn = Attention(params['hid_dim'])
    enc  = Encoder(params['output_dim'], params['emb_dim'],  params['hid_dim'], params['dropout'])
    dec  = Decoder(params['input_dim'], params['emb_dim'],  params['hid_dim'], params['dropout'], attn)
    model = Seq2SeqPackedAttention(enc, dec, params['SRC_PAD_IDX'], device)#.to(device)

    model.load_state_dict(torch.load(save_path))

    return model


def translate(text,vocab_th,vocab_en, model,src_tokenizer):
    src_text = torch.tensor([vocab_th[token] for token in src_tokenizer(text)])
    src_text = src_text.reshape(-1, 1)
    text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)

    index2eng = vocab_en.get_itos()

    with torch.no_grad():
        output, attentions = model(src_text, text_length, src_text, 0) #turn off teacher forcing

    output = output.squeeze(1)
    output = output[1:]
    output_max = output.argmax(1)

    list_answer = [index2eng[int(i)] for i in output_max]
    
    output_max.detach()

    text_trg = ' '.join(list_answer)

    return text_trg


if __name__ == "__main__":
    
    device = torch.device('cpu')
    tokenizer = word_tokenize
    model = Load_model(device)  
    model.eval()

    with open("models/vocab_en.pkl",'rb') as f:
        vocab_en = pickle.load(f)

    with open("models/vocab_th.pkl",'rb') as f:
        vocab_th = pickle.load(f)

    # SRC_LANGUAGE = 'th'
    # TRG_LANGUAGE = 'en'

    text = "อยากไปนอนแล้วบาย"

    print(translate(text,vocab_th,vocab_en, model,tokenizer))










