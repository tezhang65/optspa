from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import os

def assign_bin_to_layer(n, m):
    '''
    example: n=11, m=5
    return [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    To assign each layer to a bin. Assign the mod to the first bin
    '''
    bin_size = n // m
    extra = n % m
    sequence = [i // (bin_size + 1) if i < bin_size * extra + extra else (i - extra) // bin_size for i in range(n)]
    return sequence

def get_llava(model_path):  
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    model.seqlen = model.config.max_position_embeddings
    return model, tokenizer

