# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This file contains the definition of encoders used in https://arxiv.org/pdf/1705.02364.pdf
"""

import numpy as np
import time

import torch
import torch.nn as nn
from transformer.models import Transformer
import math
import torch.nn.functional as F
from copy import deepcopy
"""
BLSTM (max/mean) encoder
"""
cuda = True
device = torch.device("cuda:0" if cuda else "cpu")
tau = 0.1


class InferSent(nn.Module):

    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = 1 if 'version' not in config else config['version']

        self.hid_size = 600
        self.enc_lstm    = nn.LSTM(self.word_emb_dim, self.hid_size, 1, bidirectional=False, dropout=self.dpout_model)
        self.phrase_lstm = nn.LSTM(self.hid_size, self.enc_lstm_dim, 1, bidirectional=False, dropout=self.dpout_model)
        self.multihead_attn = nn.MultiheadAttention(self.enc_lstm_dim, 1)
        self.linear1 = nn.Linear(self.hid_size, self.hid_size)
        self.linear2 = nn.Linear(self.enc_lstm_dim, self.enc_lstm_dim)

        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        sent, sent_len, action, action_pos = sent_tuple
        #print(int(sent.size(0)), len(action), sent_len, action_pos)

        embedding = []
        h = torch.zeros(int(sent.size(1)), 1, 600).cuda()
        c = torch.zeros(int(sent.size(1)), 1, 600).cuda()
        for i in range(int(sent.size(0))):
            self.enc_lstm.flatten_parameters()
            o, (h, c) = self.enc_lstm(sent[i].view(1, 1, self.word_emb_dim), (h,c))
            embedding.append(o)
            h = h *(1.0 - action[i])
            c = c *(1.0 - action[i])
            #print("index: ", i, "\nh = ", h, "\nc = ", c, "\no = ", o)

        embedding = torch.cat(embedding, dim = 0)
        #print(embedding.size(), action, sent_len, action_pos)

        sent_output = embedding
        embedding = torch.index_select(embedding, 0, torch.tensor(action_pos).cuda())
        #print(embedding)
        
        self.phrase_lstm.flatten_parameters()
        embedding   = torch.tanh(self.linear1(embedding))
        sent_output = self.phrase_lstm(embedding)[0].transpose(0,1)
        sent_output = torch.tanh(self.linear2(sent_output))
        sent_output, _ = self.multihead_attn(sent_output, sent_output, sent_output)
        sent_output = sent_output.transpose(0,1)
        return torch.sum(sent_output, dim =0).view(1, self.enc_lstm_dim)
        
        #print(sent_output.size())
        #print(embedding.size())
        return sent_output[-1].view(1, -1)
        # Pooling
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

    
    def getNextHiddenState(self, hc, x):
        hidden = hc[0,0:self.enc_lstm_dim].view(1,1,self.enc_lstm_dim)
        cell = hc[0,self.enc_lstm_dim:].view(1,1,self.enc_lstm_dim)
        input = x.view(1,1,-1) #self.word_embeddings(x).view(1,1,-1)      
        self.enc_lstm.flatten_parameters() 
        out, hidden = self.enc_lstm(input, [hidden, cell])
        hidden = torch.cat([hidden[0], hidden[1]], -1).view(1, -1)
        return out, hidden


class policyNet(nn.Module):
    def __init__(self, hidden_size, embedding_length):
        super(policyNet, self).__init__()
        self.hidden = 2*hidden_size 
        self.W1 = nn.Linear(self.hidden,1, bias = False)
        self.W2 = nn.Linear(embedding_length,1, bias = False)
        self.W3 = nn.Linear(hidden_size,1, bias = True)
        
        '''
        self.W1 = nn.Parameter(torch.cuda.FloatTensor(2*self.hidden, 1).uniform_(-0.5, 0.5)) 
        self.W2 = nn.Parameter(torch.cuda.FloatTensor(embedding_length, 1).uniform_(-0.5, 0.5)) 
        self.W3 = nn.Parameter(torch.cuda.FloatTensor(embedding_length, 1).uniform_(-0.5, 0.5)) 
        self.b = nn.Parameter(torch.cuda.FloatTensor(1, 1).uniform_(-0.5, 0.5))
        '''

    def forward(self, h, x, hF):
        '''
        h_ = torch.matmul(h.view(1,-1), self.W1) # 1x1
        x_ = torch.matmul(x.view(1,-1), self.W2) # 1x1
        hF_ = torch.matmul(hF.view(1,-1), self.W3) # 1x1
        '''
        h_ = self.W1(h.view(1,-1)) # 1x1
        x_ = self.W2(x.view(1,-1)) # 1x1
        hF_ = self.W3(hF.view(1,-1)) # 1x1
        scaled_out = torch.sigmoid(h_ +  x_ + hF_) # 1x1
        scaled_out = torch.clamp(scaled_out, min=1e-5, max=1 - 1e-5)
        scaled_out = torch.cat([1.0 - scaled_out, scaled_out],0)
        return scaled_out

class actor(nn.Module):
    def __init__(self, hidden_size, embedding_length):
        super(actor, self).__init__()
        self.target_policy = policyNet(hidden_size, embedding_length)
        self.active_policy = policyNet(hidden_size, embedding_length)
     
    def get_target_logOutput(self, h, x):
        out = self.target_policy(h, x)
        logOut = torch.log(out)
        return logOut

    def get_target_output(self, h, x, s, scope):
        if scope == "target":
            out = self.target_policy(h, x, s)
        if scope == "active":
            out = self.active_policy(h, x, s)
        return out

    def get_gradient(self, h, x, s, reward, scope):
        if scope == "target":
            out = self.target_policy(h, x, s)
            logout = torch.log(out).view(-1)
            index = reward.index(0)
            index = (index + 1) % 2
            grad = torch.autograd.grad(logout[index].view(-1), self.target_policy.parameters()) # torch.cuda.FloatTensor(reward[index])
            grad[0].data = grad[0].data * reward[index]
            grad[1].data = grad[1].data * reward[index]
            grad[2].data = grad[2].data * reward[index]
            grad[3].data = grad[3].data * reward[index]
            return grad
        if scope == "active":
            out = self.active_policy(h, x)
        return out

    def assign_active_network_gradients(self, grad1, grad2, grad3, grad4):
        params = [grad1, grad2, grad3, grad4]    
        i=0
        for name, x in self.active_policy.named_parameters():
            x.grad = deepcopy(params[i])
            i+=1

    def update_target_network(self):
        params = []
        for name, x in self.active_policy.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_policy.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
            i+=1

    def assign_active_network(self):
        params = []
        for name, x in self.target_policy.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_policy.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

class critic(nn.Module):
    def __init__(self, config):
        super(critic, self).__init__()
        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        self.active_pred = eval(self.encoder_type)(config)
        self.target_pred = eval(self.encoder_type)(config)
        
        self.inputdim = 4*1*self.enc_lstm_dim

        if self.nonlinear_fc:
            self.active_classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
            self.target_classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.active_classifier = nn.Sequential(
                nn.Linear(int(self.inputdim), self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )
            self.target_classifier = nn.Sequential(
                nn.Linear(int(self.inputdim), self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )
   
    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.encoder.pos_emb.parameters()))
        dec_freezed_param_ids = set(map(id, self.encoder.decoder.pos_emb.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)
       
    def assign_target_network(self):
        params = []
        for name, x in self.active_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1
        params = []
        for name, x in self.active_classifier.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_classifier.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

    def update_target_network(self):
        params = []
        for name, x in self.active_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
            i+=1

        params = []
        for name, x in self.active_classifier.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_classifier.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
            i+=1

    def assign_active_network(self):
        params = []
        for name, x in self.target_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1
        params = []
        for name, x in self.target_classifier.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_classifier.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

    def assign_active_network_gradients(self):
        params = []
        for name, x in self.target_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_pred.named_parameters():
            x.grad = deepcopy(params[i].grad)
            i+=1
        for name, x in self.target_pred.named_parameters():
            x.grad = None

        params = []
        for name, x in self.target_classifier.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_classifier.named_parameters():
            x.grad = deepcopy(params[i].grad)
            i+=1
        for name, x in self.target_classifier.named_parameters():
            x.grad = None

    def forward(self, s1, s2, scope):
        # s1 : (s1, s1_len)
        if scope == "target":
            u = self.target_pred(s1)
            v = self.target_pred(s2)
            features = torch.cat((u, v, torch.abs(u-v), u*v), 1)     
            output = self.target_classifier(features)
        
        if scope == "active":
            u = self.active_pred(s1)
            v = self.active_pred(s2)
            features = torch.cat((u, v, torch.abs(u-v), u*v), 1)     
            output = self.active_classifier(features)
        
        return output

    def summary(self, s1):
        emb = self.target_pred(s1)
        return emb
    
    def forward_lstm(self, hc, x, scope):
        if scope == "target":
            out, state = self.target_pred.getNextHiddenState(hc, x)
        if scope == "active":
            out, state = self.active_pred.getNextHiddenState(hc, x)
        return out, state