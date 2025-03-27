"""
draw_curve.py绘制的foorline模型曲线中, 横坐标为token_length, 我现在把横坐标修正为算数强度, 看看会有什么变化。

"""
import matplotlib.pyplot as plt
import numpy as np

def TL2AI(TL):
    """
    Description: 
        Token length和AI (arithmetic intensity) 的转换。
        load_store的数据量.包括权重和激活值的load。不包括激活值和kv cache的store(其实我感觉store也占时间呀,为什么不包括呢)。
        embedding层的计算时间就先不管了。计算时间比较短。
    """
    dim = 4096
    inter = 11008
    weight = 32 * (dim * dim * 4 + dim * inter * 3) + dim * 32000
    weight *= 2     # Byte 

    layer_act = TL * dim            # layernorm
    layer_act += TL * dim * 3       # qkv
    layer_act += TL * dim * 2       # qkt
    layer_act += TL * TL            # softmax
    layer_act += (TL * TL + TL * dim)       # pv
    layer_act += TL * dim           # o_proj
    layer_act += TL * dim           # layernorm
    layer_act += TL * dim * 2       # up, gate
    layer_act += TL * inter * 2     # up + gate
    layer_act += TL * inter         # down
    act = layer_act * 32 + TL * dim * 2 # 32 layers, final_norm, vocab_size
    act *= 2                        # Byte

    # print(f"weight:{weight/1e9} GB.")
    # print(f"act:{act/1e9} GB.")
    load = weight + act

    comp = TL * dim * dim * 3       # qkv
    comp += TL * TL * dim * 2       # qkt, pv
    comp += TL * dim * dim          # o_proj
    comp += TL * dim * inter * 3    # fc block
    comp *= 32
    comp += TL * dim * 32000
    comp *= 2                       # 1 MAC = 2 flops
    # print(f"computation: {comp/1e9} GFLOPs")

    AI = comp / load
    # print(f"Arithmetic intensity: {AI}")
    return AI

def TL2AI_update(TL):
    """
    Description:
        算数强度的计算应该更新, 存储部分不仅仅要描述对act的load, 也要描述对act的store。
        因为我在绘制GPU的Roofline模型时, 它的带宽可是将load和store包含在一起了。
    """
    dim = 4096
    inter = 11008
    weight = 32 * (dim * dim * 4 + dim * inter * 3) + dim * 32000
    weight *= 2     # Byte 

    layer_act = TL * dim            # layernorm
    layer_act += TL * dim * 3       # qkv
    layer_act += TL * dim * 2       # qkt
    layer_act += TL * TL            # softmax
    layer_act += (TL * TL + TL * dim)       # pv
    layer_act += TL * dim           # o_proj
    layer_act += TL * dim           # layernorm
    layer_act += TL * dim * 2       # up, gate
    layer_act += TL * inter * 2     # up + gate
    layer_act += TL * inter         # down
    act = layer_act * 32 + TL * dim * 2 # 32 layers, final_norm, vocab_size
    act *= 2                        # Byte

    # print(f"weight:{weight/1e9} GB.")
    # print(f"act:{act/1e9} GB.")
    load = weight + act

    # 添加store
    act = TL * dim                  # layernorm
    layer_act += TL * dim * 3       # qkv
    layer_act += TL * TL            # qkt
    layer_act += TL * TL            # softmax
    layer_act += TL * dim           # pv
    layer_act += TL * dim           # o_proj
    layer_act += TL * dim           # layernorm
    layer_act += TL * inter * 2     # up, gate
    layer_act += TL * inter         # up + gate
    layer_act += TL * dim           # down
    act = layer_act * 32 + TL * dim + TL * 32000 # 32 layers, final_norm, vocab_size
    act *= 2                        # Byte
    store = act

    load_store = load + store

    comp = TL * dim * dim * 3       # qkv
    comp += TL * TL * dim * 2       # qkt, pv
    comp += TL * dim * dim          # o_proj
    comp += TL * dim * inter * 3    # fc block
    comp *= 32
    comp += TL * dim * 32000
    comp *= 2                       # 1 MAC = 2 flops
    # print(f"computation: {comp/1e9} GFLOPs")

    AI = comp / load_store
    # print(f"Arithmetic intensity: {AI}")
    return AI


if __name__ == "__main__":
    TLs = [5,10,20,30,40,50,100,200,250,300,350,400,450,500,550,600,800,1000,1200,1400,1600,1800,2000]
    times = [12.76,12.67,13.01,13.14,13.54,13.59,16.50,24.05,24.55,30.84,31.68,40.47,42.16,43.90,50.95,
             53.14,70.56,85.74,99.77,109.8,129.42,149.78,160.98,]
    AIs = []
    for TL in TLs:
        # AI = TL2AI(TL)
        AI = TL2AI_update(TL)
        AIs.append(AI)
    print(AIs)
    
    performance = np.array(TLs) / np.array(times) * 1000 # tokens/s

    plt.figure(figsize=(10,6))
    plt.plot(AIs, performance, marker='o', linestyle='-')
    plt.xlabel('Arithmeitc Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (tokens / s)')
    plt.title('Tested Roofline model')
    plt.grid(True)
    plt.show()
    # plt.savefig("roofline_model_v2.png")
    plt.savefig("roofline_model_v3.png")
