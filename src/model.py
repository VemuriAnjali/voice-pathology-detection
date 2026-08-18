import numpy as np
import keras
from keras import layers, Model, Input

def residual_block(x,units,drop=.3):
    h=layers.Dense(units,activation='relu')(x); h=layers.BatchNormalization()(h); h=layers.Dropout(drop)(h); h=layers.Dense(units)(h); h=layers.BatchNormalization()(h)
    if x.shape[-1]!=units: x=layers.Dense(units)(x)
    return layers.Activation('relu')(layers.Add()([x,h]))

def sparse_transformer(x,embed_dim,num_heads=4,keep_ratio=.7):
    seq=layers.Reshape((1,embed_dim))(x)
    attn=layers.MultiHeadAttention(num_heads=num_heads,key_dim=max(1,embed_dim//num_heads),dropout=round(1.0-keep_ratio,2))(seq,seq)
    attn=layers.LayerNormalization(epsilon=1e-6)(layers.Add()([seq,attn])); ff=layers.Dense(embed_dim*2,activation='gelu')(attn); ff=layers.Dense(embed_dim)(ff)
    return layers.Flatten()(layers.LayerNormalization(epsilon=1e-6)(layers.Add()([attn,ff])))

def build_msrspT152(input_dim,num_classes,embed_dim=128):
    inp=Input(shape=(input_dim,),name='msrsp_in'); s1=residual_block(inp,256); s1=residual_block(s1,128); s1=residual_block(s1,64)
    half=input_dim//2; s2=layers.Lambda(lambda t:t[:,:half],name='stream2_slice')(inp); s2=residual_block(s2,128); s2=residual_block(s2,64)
    merged=layers.Concatenate()([s1,s2]); merged=layers.Dense(256,activation='relu')(merged); merged=layers.BatchNormalization()(merged); merged=layers.Dense(embed_dim,activation='relu')(merged)
    tr_out=sparse_transformer(merged,embed_dim=embed_dim,num_heads=4); embed=layers.Dense(embed_dim,activation='relu',name='embedding')(tr_out); embed=layers.BatchNormalization()(embed)
    x=layers.Dropout(.35)(embed); x=layers.Dense(64,activation='relu')(x); x=layers.Dropout(.25)(x); out=layers.Dense(num_classes,activation='softmax',name='clf')(x)
    return Model(inp,out,name='MSRSpT152'),Model(inp,embed,name='MSRSpT152_embed')

class HierarchicalSelfAttention(keras.Layer):
    def __init__(self,embed_dim,num_heads=4,**kwargs):
        super().__init__(**kwargs); self.embed_dim=embed_dim; kd=max(1,embed_dim//num_heads)
        self.local_attn=layers.MultiHeadAttention(num_heads=num_heads,key_dim=kd,dropout=.2); self.global_attn=layers.MultiHeadAttention(num_heads=num_heads,key_dim=kd,dropout=.2); self.target_attn=layers.MultiHeadAttention(num_heads=num_heads,key_dim=kd)
        self.ln1=layers.LayerNormalization(epsilon=1e-6); self.ln2=layers.LayerNormalization(epsilon=1e-6); self.pool=layers.GlobalAveragePooling1D(keepdims=True); self.target_dense=layers.Dense(embed_dim); self.flat=layers.Flatten()
    def call(self,x,training=False):
        l1=self.local_attn(x,x,training=training); x1=self.ln1(x+l1); g=self.pool(x1); l2=self.global_attn(g,x1,training=training); x2=self.ln2(g+l2); tgt=self.target_dense(self.flat(x2)); tgt=keras.ops.expand_dims(tgt,axis=1); out=self.target_attn(tgt,x1,training=training); return self.flat(out)

def build_conv_hsa(input_dim,num_classes,embed_dim=64):
    inp=Input(shape=(input_dim,),name='conv_hsa_in'); x=layers.Reshape((input_dim,1))(inp); x=layers.Conv1D(64,3,activation='relu',padding='same')(x); x=layers.BatchNormalization()(x); x=layers.MaxPooling1D(2,padding='same')(x); x=layers.Conv1D(128,3,activation='relu',padding='same')(x); x=layers.BatchNormalization()(x); x=layers.MaxPooling1D(2,padding='same')(x); x=layers.Conv1D(embed_dim,3,activation='relu',padding='same')(x); x=layers.BatchNormalization()(x); x=HierarchicalSelfAttention(embed_dim,4,name='hsa')(x); embed=layers.Dense(64,activation='relu',name='s1_embed')(x); out=layers.Dense(num_classes,activation='softmax',name='s1_out')(layers.Dropout(.45)(embed))
    return Model(inp,out,name='Conv_HSA'),Model(inp,embed,name='Conv_HSA_embed')
