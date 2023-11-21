from .preprocess_cumsum import PreprocessCumSum
from .chunk_scan_triton_full import Chunk_memory_update

def inter_chunk_onc(query, key, value, gk, gv, normalizer_gk=16, normalizer_gv=16):
    
    g_key_cumsum, g_value_cumsum, reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = PreprocessCumSum.apply(query, key, value, gk, gv, normalizer_gk, normalizer_gv)

    to_add = reduce_key.transpose(-1, -2) @ reduce_value 

    memory_cache = Chunk_memory_update.apply(g_key_last_exp, g_value_last_exp, to_add)
    
    inter_chunk_contribution = ((q_exp) @ memory_cache) * g_value_cumsum_exp

    return g_key_cumsum, g_value_cumsum, inter_chunk_contribution




