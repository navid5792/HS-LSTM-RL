# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# small window no delay

from comet_ml import Experiment
import os
import sys
import time
import argparse
from collections import Counter
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models_GRU import critic, actor
#from models_LSTM_attention import critic, actor
from read_data import get_MSRP_data, get_SICK_data, get_AI_data, get_SICK_tree_data, get_QQP_data


logg = "LR: 0.02"
print(logg)

parser = argparse.ArgumentParser(description='NLI training')

#experiment = Experiment(api_key="3Csy9MHwcDspZldyCHP7bkuMe",
                        #project_name="hs-lstm-rl", workspace="navid5792")

samplecnt = 5
epsilon = 0.05
alpha = 0.1
verbose = 0
mode = 2    # 1 added and 2 removed
if verbose:
    f = open("results.txt", "w")
    f.close()


# paths
parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/msrp/msrp1/', help="Output directory")

parser.add_argument("--criticmodelname", type=str, default='model_test.pickle')
parser.add_argument("--actormodelname", type=str, default='model_actor.pickle')
parser.add_argument("--Newcriticmodelname", type=str, default='model_test5.pickle')


parser.add_argument("--word_emb_path", type=str, default="dataset/glove.840B.300d.txt", help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0.05, help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=1, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.23", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=1, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=600, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=2, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()



# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)
random.seed(params.seed)

"""
DATA
"""
#train, valid, test = get_nli(params.nlipath)

train, valid, test = get_MSRP_data()

#print(len(valid['s1'][100].split()), len(valid['a1'][100]))

word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)

'''
for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])
'''
for data_type in ['train', 'valid', 'test']:
    for k in (('s1', 'a1'), ('s2', 'a2')):
        struct_sent = eval(data_type)[k[0]]
        struct_action = eval(data_type)[k[1]]
        for i in range(len(struct_sent)):
            sent = struct_sent[i].split()
            action = struct_action[i]
            temp_action = []
            for j in range(len(sent)):
                if sent[j] in word_vec:
                    temp_action.append(action[j])
            eval(data_type)[k[1]][i] = temp_action


for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([[word for word in sent.split() if word in word_vec] for sent in eval(data_type)[split] ])

for split in [('s1', 'a1'), ('s2', 'a2')]:
    cnt = 0
    for data_type in ['train', 'valid', 'test']:
        sent = eval(data_type)[split[0]]
        action = eval(data_type)[split[1]]
        for i in range(len(sent)):
            assert(len(sent[i]) == len(action[i]))
            cnt += 1
        print(cnt)

        

"""
MODEL
"""
# model config
config_nli_model = {
    'optimizer_LR'   :  params.optimizer      ,
    'n_words'        :  len(word_vec)         ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,
    'RNN type'       :  "GRU"                ,
    'n_epochs'       :  params.n_epochs       ,
    'details'        :  logg,
}

#experiment.log_parameters(config_nli_model)


# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
nli_net = critic(config_nli_model)
actorModel = actor(params.enc_lstm_dim, params.word_emb_dim)
print(nli_net)
print(actorModel)

for name, x in nli_net.named_parameters():
    print(name)

for name, x in actorModel.named_parameters():
    print(name)

#print(nli_net.target_pred.enc_lstm.weight_ih_l0)
#print(nli_net.target_classifier[4].bias)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False


# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
critic_target_optimizer = optim_fn(list(nli_net.target_pred.parameters()) + list(nli_net.target_classifier.parameters()), **optim_params)

optim_fn2, optim_params2 = get_optimizer(params.optimizer)
critic_active_optimizer = optim_fn(list(nli_net.active_pred.parameters()) + list(nli_net.active_classifier.parameters()), **optim_params2)


optim_fn3, optim_params3 = get_optimizer("adam,lr=0.1")
actor_target_optimizer = optim_fn3(actorModel.target_policy.parameters(), **optim_params3)

optim_fn4, optim_params4 = get_optimizer("adam,lr=0.1")
actor_active_optimizer = optim_fn4(actorModel.active_policy.parameters(), **optim_params4)

# cuda by default
nli_net.cuda()
actorModel.cuda()
loss_fn.cuda()


def Sampling_RL(current, summary, length, epsilon, Random = True):
    current_lower_state = torch.zeros(1, params.enc_lstm_dim).cuda()
    current_upper_state = torch.zeros(1, params.enc_lstm_dim).cuda()
    current = current.squeeze(0)
    actions = []
    states = []
    for pos in range(0, length):
        out_d, current_lower_state = nli_net.forward_lstm_lower(current_lower_state, current[pos], scope = "target")
        predicted = actorModel.get_target_output(current_upper_state, current_lower_state, summary, scope = "target")
        states.append([current_lower_state, current_upper_state, summary])
        if mode == 1:
            if Random:
                if random.random() > epsilon:
                    action = (0 if random.random() < float(predicted[0].item()) else 1)
                else:
                    action = (1 if random.random() < float(predicted[0].item()) else 0)
            else:
                action = int(torch.argmax(predicted))
        if mode == 2:
            if Random:
                action = (0 if random.random() < float(predicted[0].item()) else 1)
            else:
                action = int(torch.argmax(predicted))
        actions.append(action)
        if action == 1:
            _, current_upper_state = nli_net.forward_lstm_upper(current_upper_state, out_d, scope = "target")
            current_lower_state = torch.zeros(1, params.enc_lstm_dim).cuda()

    action_pos = []
    actions[-1] = 1
    for (i, j) in enumerate(actions):
        if j == 1:
            action_pos.append(i)
    '''
    if 1 not in actions and len(action_pos) == 0:
        actions[-1] = 1
        action_pos.append(length-1)
    '''
    return actions, states, action_pos

def sampling_random(lenth, p_action = None):
#    print(lenth, p_action)
    if p_action.count(1) == 0:
        p_action[-1] = 1
    actions = []
    actions = np.copy(p_action).tolist()
    actions += [0] * (lenth - len(actions))
    action_pos = []
    for (i, j) in enumerate(actions):
        if j == 1:
            action_pos.append(i) # append the actual index of action that is 1
    if len(action_pos) == 0:       # if none of the action is 1 then manually add 1 at the end and track the index. adding 1 at the end meaning evrything falls under one phrase.
        actions[lenth-1] = 1
        action_pos.append(lenth-1)
    if len(actions) != lenth:
        print (lenth, p_action, "assert")
    return actions, action_pos
"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params2['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch, RL_train = True, LSTM_train = True):
    print('\nTRAINING : Epoch ' + str(epoch))
    
    actorModel.train(False)
    nli_net.train(False)
    if RL_train:
        print("Actor Training")
        #actor_active_optimizer.param_groups[0]['lr'] = actor_active_optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        #and 'sgd' in params.optimizer else actor_active_optimizer.param_groups[0]['lr']
        #actor_active_optimizer.param_groups[0]['lr'] = 0.1
        print('Learning rate : {0}'.format(actor_active_optimizer.param_groups[0]['lr']))
        actorModel.train()
    if LSTM_train:
        print("Crtic Training")
        critic_active_optimizer.param_groups[0]['lr'] = critic_active_optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else critic_active_optimizer.param_groups[0]['lr']
        print('Learning rate : {0}'.format(critic_active_optimizer.param_groups[0]['lr']))
        nli_net.train()
    
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    epoch_loss = []
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]
    a1 = train['a1'][permutation]
    a2 = train['a2'][permutation]


    for stidx in tqdm(range(0, len(s1), params.batch_size)):
        #print(nli_net.target_pred.enc_lstm.bias_hh_l0)
        #print(nli_net.active_pred.enc_lstm.bias_hh_l0)
        # prepare batch
        #if stidx % 100 == 0:
            #print(stidx, "/", len(s1))
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s1_action, s2_action = a1[stidx:stidx + params.batch_size], a2[stidx:stidx + params.batch_size]
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + s1_batch.size(1)])).cuda()

        k = s1_batch.size(1)  # actual batch size
        predict = torch.zeros(s1_batch.size(1), params.n_classes).cuda()
        avgloss = 0.
        totloss = 0.
        nli_net.assign_active_network()
        actorModel.assign_active_network()
        #print("Target Weight: ", actorModel.target_policy.W1.weight.data, "\n\n")
        for kk in range(s1_batch.size(1)):
            
            left = s1_batch.transpose(0,1)[kk].view(-1, 1, 300)
            right = s2_batch.transpose(0,1)[kk].view(-1, 1, 300)
            
            left_len = np.array([s1_len[kk]])
            right_len = np.array([s2_len[kk]])
            
            left_action = s1_action[kk]
            right_action = s2_action[kk]
            
            left_action, left_action_pos = sampling_random(int(s1_len[kk]), left_action)
            right_action, right_action_pos = sampling_random(int(s2_len[kk]), right_action)

            #print(left.size(), left_action_pos, left_action)
            #aaa

            tgt = tgt_batch[kk].view(-1)
            if RL_train:
                leftSummary = nli_net.summary((left, left_len, left_action, left_action_pos))[-1]
                rightSummary = nli_net.summary((right, right_len, right_action, right_action_pos))[-1]
                '''
                actor_active_optimizer.zero_grad()
                actor_target_optimizer.zero_grad()
                #print(leftSummary.size(), rightSummary.size())

                actionlist_left, actionlist_right, statelist_left, statelist_right, losslist = [], [], [], [], []
                aveloss = 0.
                for i in range(samplecnt):
                    actions_left, states_left, Rleft_action_pos     = Sampling_RL(left, rightSummary, int(left_len), epsilon, Random=True)
                    actions_right, states_right, Rright_action_pos  = Sampling_RL(right, leftSummary, int(right_len), epsilon, Random=True)
                   
                    actionlist_left.append(actions_left)
                    statelist_left.append(states_left)
                    actionlist_right.append(actions_right)
                    statelist_right.append(states_right)
                    
                    out = nli_net((left, left_len, actions_left, Rleft_action_pos), (right, right_len, actions_right, Rright_action_pos), scope = "target")
                    loss_ = loss_fn(out, tgt)
                   
                    _x = float(len(Rleft_action_pos)) /  left_len       #L'/L
                    _y = float(len(Rright_action_pos)) /  right_len     #L'/L
                    
                    #lossL = (1 * _x + 0.1 / _x - 0.6)* 0.1 * params.n_classes
                    #lossR = (1 * _y + 0.1 / _y - 0.6)* 0.1 * params.n_classes
                    
                    lossL = (( _x + (1/_x) * 0.25) - 1.0)
                    lossR = (( _y + (1/_y) * 0.25) - 1.0)
                    loss_ = float(float(loss_) + ((lossL + lossR)/2) * 0.1 * params.n_classes)

                    aveloss += loss_
                    losslist.append(loss_)
                
                aveloss /= samplecnt
                totloss += aveloss
                grad1 = None
                grad2 = None
                grad3 = None
                grad4 = None
                flag = 0 
                #print(actionlist_left, losslist)
                
                for i in range(samplecnt): #5
                    for pos in range(len(actionlist_left[i])): #19 --> 13
                        rr = [0, 0]
                        rr[actionlist_left[i][pos]] = ((losslist[i] - aveloss) * alpha)
                        g = actorModel.get_gradient(statelist_left[i][pos][0], statelist_left[i][pos][1], statelist_left[i][pos][2], rr, scope = "target")
                        if flag == 0:
                            grad1 = g[0]
                            grad2 = g[1]
                            grad3 = g[2]
                            grad4 = g[3]
                            flag = 1
                        else:
                            grad1 += g[0]
                            grad2 += g[1]
                            grad3 += g[2]
                            grad4 += g[3]
                    for pos in range(len(actionlist_right[i])): # 25 --> 5
                        rr = [0, 0]
                        rr[actionlist_right[i][pos]] = ((losslist[i] - aveloss) * alpha)
                        g = actorModel.get_gradient(statelist_right[i][pos][0], statelist_right[i][pos][1], statelist_right[i][pos][2], rr, scope = "target")
                        grad1 += g[0]
                        grad2 += g[1]
                        grad3 += g[2]
                        grad4 += g[3]

                actorModel.assign_active_network_gradients(grad1, grad2, grad3, grad4)
                actor_active_optimizer.step()
                actions_left, states_left, Rleft_action_pos     = Sampling_RL(left, rightSummary, int(left_len), epsilon, Random=False)
                actions_right, states_right, Rright_action_pos  = Sampling_RL(right, leftSummary, int(right_len), epsilon, Random=False)               
                output = nli_net((left, left_len, actions_left, Rleft_action_pos), (right, right_len, actions_right, Rright_action_pos), scope = "target")
                predict[kk] = output
                '''
                if LSTM_train:
                    critic_active_optimizer.zero_grad()
                    critic_target_optimizer.zero_grad()
                    actions_left, states_left, Rleft_action_pos     = Sampling_RL(left, rightSummary, int(left_len), epsilon, Random=False)
                    actions_right, states_right, Rright_action_pos  = Sampling_RL(right, leftSummary, int(right_len), epsilon, Random=False)
                    output = nli_net((left, left_len, actions_left, Rleft_action_pos), (right, right_len, actions_right, Rright_action_pos), scope = "target")
                    predict[kk] = output
                    loss = loss_fn(output, tgt)
                    avgloss += loss.item()
                    loss.backward()
                    nli_net.assign_active_network_gradients()
                    shrink_factor = 1
                    total_norm = 0
                    for p in nli_net.active_pred.parameters():
                        if p.requires_grad:
                            p.grad.data.div_(k ** 2)  # divide by the actual batch size
                            total_norm += p.grad.data.norm() ** 2
                    for p in nli_net.active_classifier.parameters():
                        if p.requires_grad:
                            p.grad.data.div_(k ** 2)  # divide by the actual batch size
                            total_norm += p.grad.data.norm() ** 2
                    total_norm = np.sqrt(total_norm.cpu())
                    if total_norm > params.max_norm:
                        shrink_factor = params.max_norm / total_norm
                    current_lr = critic_active_optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
                    critic_active_optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update
                    critic_active_optimizer.param_groups[0]['lr'] = current_lr            
                    critic_active_optimizer.step()         
            else:
                critic_active_optimizer.zero_grad()
                critic_target_optimizer.zero_grad()
                output = nli_net((left, left_len, left_action, left_action_pos), (right, right_len, right_action, right_action_pos), "target")
                predict[kk] = output
                loss = loss_fn(output, tgt)
                avgloss += loss.item()
                loss.backward()
                nli_net.assign_active_network_gradients()
                shrink_factor = 1
                total_norm = 0
                for p in nli_net.active_pred.parameters():
                    if p.requires_grad:
                        p.grad.data.div_(k ** 2)  # divide by the actual batch size
                        total_norm += p.grad.data.norm() ** 2
                for p in nli_net.active_classifier.parameters():
                    if p.requires_grad:
                        p.grad.data.div_(k ** 2)  # divide by the actual batch size
                        total_norm += p.grad.data.norm() ** 2
                total_norm = np.sqrt(total_norm.cpu())
                if total_norm > params.max_norm:
                    shrink_factor = params.max_norm / total_norm
                current_lr = critic_active_optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
                critic_active_optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update
                critic_active_optimizer.param_groups[0]['lr'] = current_lr            
                critic_active_optimizer.step()
        
        
        if RL_train:
            #actorModel.update_target_network()         
            if LSTM_train:
                nli_net.update_target_network()
        else:
            nli_net.assign_target_network()
        
        pred = predict.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss

        if LSTM_train:
            all_costs.append(avgloss/params.batch_size)
            epoch_loss.append(avgloss/params.batch_size)
        if RL_train:
            all_costs.append(totloss/params.batch_size)
            epoch_loss.append(totloss/params.batch_size)
        
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        #print(nli_net.classifier[4].bias)

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct.item()/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []

    train_acc = round(100 * correct.item()/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'.format(epoch, train_acc))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    if eval_type == "train":
        s1  = train['s1']
        s2  = train['s2']
        a1  = train['a1']
        a2  = train['a2']
        target  = train['label']
    if eval_type == "test":
        s1  = test['s1']
        s2  = test['s2']
        a1  = test['a1']
        a2  = test['a2']
        target  = test['label']
    if eval_type == "valid":
        s1  = valid['s1']
        s2  = valid['s2']
        a1  = valid['a1']
        a2  = valid['a2']
        target  = valid['label']

    for i in tqdm(range(0, len(s1), 1)):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + 1], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + 1], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + 1])).cuda()

        s1_action, s2_action = a1[i:i + 1], a2[i:i + 1]

        left_action = s1_action[0]
        right_action = s2_action[0]
        
        left_action, left_action_pos = sampling_random(int(s1_batch.size(0)), left_action)
        right_action, right_action_pos = sampling_random(int(s2_batch.size(0)), right_action)

        #print(s1_batch.size(), s1_len, s1_action, s2_action, left_action, left_action_pos)
        #aaaa

        # model forward
        output = nli_net((s1_batch, s1_len, left_action, left_action_pos), (s2_batch, s2_len, right_action, right_action_pos), "target")

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct.item() / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc >= val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir, params.criticmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                critic_active_optimizer.param_groups[0]['lr'] = critic_active_optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              critic_active_optimizer.param_groups[0]['lr']))
                if critic_active_optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc

def evaluate_RL(epoch, typ = 0, eval_type='valid', final_eval=False):
    nli_net.eval()
    actorModel.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    if eval_type == "train":
        s1  = train['s1']
        s2  = train['s2']
        a1  = train['a1']
        a2  = train['a2']
        target  = train['label']
    if eval_type == "test":
        s1  = test['s1']
        s2  = test['s2']
        a1  = test['a1']
        a2  = test['a2']
        target  = test['label']
    if eval_type == "valid":
        s1  = valid['s1']
        s2  = valid['s2']
        a1  = valid['a1']
        a2  = valid['a2']
        target  = valid['label']


    ll, rl, ll_, rl_ = 0, 0, 0, 0
    deleteCount = dict()
    wordCount = dict()
    for i in tqdm(range(0, len(s1))):
        
        left, left_len = get_batch(s1[i:i + 1], word_vec, params.word_emb_dim)
        right, right_len = get_batch(s2[i:i + 1], word_vec, params.word_emb_dim)
        left, right = Variable(left.cuda()), Variable(right.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + 1])).cuda()

        s1_action, s2_action = a1[i:i + 1], a2[i:i + 1]

        left_action = s1_action[0]
        right_action = s2_action[0]
        
        left_action, left_action_pos = sampling_random(left_len, left_action)
        right_action, right_action_pos = sampling_random(right_len, right_action)

        leftSummary = nli_net.summary((left, left_len, left_action, left_action_pos))[-1]
        rightSummary = nli_net.summary((right, right_len, right_action, right_action_pos))[-1]

        actions_left, states_left, Rleft_action_pos     = Sampling_RL(left, rightSummary, int(left_len), epsilon, Random=False)
        actions_right, states_right, Rright_action_pos  = Sampling_RL(right, leftSummary, int(right_len), epsilon, Random=False)
        
        #print(s1_batch.size(), actions_left, Rinput_left.size(), s2_batch.size(), actions_right, Rinput_right.size(), "\n\n")
        output = nli_net((left, left_len, actions_left, Rleft_action_pos), (right, right_len, actions_right, Rright_action_pos), scope = "target")
        '''
        if (len(left_action_pos) != left_len) and len(Rleft_action_pos) != 1:
            print(left_action_pos, "--------", Rleft_action_pos)
        '''
        pred = output.data.max(1)[1]
       
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
    
    '''
    with open("results.txt", "a") as f:
        f.write("Average left: " + str(ll/len(s1)) + "\nAverage left new: " + str(ll_/len(s1)) + "\nAverage right: " + str(rl/len(s1)) + "\nAverage right new: " + str(rl_/len(s1)))
        #f.write(deleteCount + "\n\n\n" + wordCount)
        for key, value in sorted(deleteCount.items(), key=lambda item: item[1]):
            f.write(str(key) + ":" + str(value) + "\n")
        f.write("\n\n\n")
        for key, value in sorted(wordCount.items(), key=lambda item: item[1]):
            f.write(str(key) + ":" + str(value) + "\n")
    '''

    # save model
    eval_acc = round(100 * correct.item() / len(s1), 2)
    print(eval_type, " accuracy: ", eval_acc)
    
    '''
    if final_eval:
        params.criticmodelname = bothCritic
        params.actormodelname = bothActor
    ''' 
    
    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc >= val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            if typ == 1:
                torch.save(actorModel.state_dict(), os.path.join(params.outputdir, params.actormodelname))
            if typ == 2:
                torch.save(nli_net.state_dict(), os.path.join(params.outputdir, params.Newcriticmodelname))
                #torch.save(actorModel.state_dict(), os.path.join(params.outputdir, params.Newactormodelname))
            val_acc_best = eval_acc 
        else:
            if final_eval:
                if 'sgd' in params.optimizer:
                    critic_active_optimizer.param_groups[0]['lr'] = critic_active_optimizer.param_groups[0]['lr'] / params.lrshrink
                    print('Shrinking lr by : {0}. New lr = {1}'
                          .format(params.lrshrink,
                                  critic_active_optimizer.param_groups[0]['lr']))
                    if critic_active_optimizer.param_groups[0]['lr'] < params.minlr:
                        stop_training = True
                if 'adam' in params.optimizer:
                    # early stopping (at 2nd decrease in accuracy)
                    stop_training = adam_stop
                    adam_stop = True
    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 10000

#nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.criticmodelname)))
#print("\nCritic Loaded")
print("Just Critic Training......")

while not stop_training and epoch <= params.n_epochs:
    with experiment.train():
        train_accc, train_losss = trainepoch(epoch, RL_train = False)
        experiment.log_metric("Train Accuracy", train_accc, step = epoch)
        experiment.log_metric("Train Loss", sum(train_losss) / len(train_losss) , step = epoch)
    with experiment.test():
        eval_accc = evaluate(epoch, 'valid')
        experiment.log_metric("Validation Accuracy", eval_accc, step = epoch)
    epoch += 1

# Run best model on test set.
print("Just Critic Training......Done!")
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.criticmodelname)))
print("\nCritic Loaded")

#print(evaluate(epoch, 'train'))
#print(evaluate(epoch, 'valid', final_eval = True))
#print(evaluate(epoch, 'test', final_eval = True)) 


#actorModel.load_state_dict(torch.load(os.path.join(params.outputdir, params.actormodelname)))
#print("\nActor Loaded")

epoch = 10000
print("Actor Training......")

while not stop_training and epoch <= params.n_epochs:
    print(trainepoch(epoch, LSTM_train = False))
    eval_acc = evaluate_RL(epoch, 1, 'valid')
    epoch += 1

print("Actor Training......Done!")
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.criticmodelname)))
print("\nCritic Loaded")
actorModel.load_state_dict(torch.load(os.path.join(params.outputdir, params.actormodelname)))
print("\nActor Loaded")

#print(evaluate_RL(epoch, 0, 'test', final_eval = True)) 
#print(evaluate_RL(epoch, 0, 'valid', final_eval = True)) 
#print(evaluate_RL(epoch, 0, 'test', final_eval = True)) 


epoch = 1
print("Again Critic Training......")
while not stop_training and epoch <=params.n_epochs:
    print(trainepoch(epoch))
    eval_acc = evaluate_RL(epoch, 2, 'valid')
    print(eval_acc)
    #test_acc = evaluate_RL(epoch, 2, 'test')
    #print(test_acc)
    epoch += 1

print("Again Critic Training......Done!")
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.Newcriticmodelname)))
print("\nCritic Loaded")
actorModel.load_state_dict(torch.load(os.path.join(params.outputdir, params.actormodelname)))
print("\nActor Loaded")

#print(evaluate_RL(epoch, 0, 'train', final_eval = True))
print(evaluate_RL(epoch, 0, 'valid', final_eval = True))
print(evaluate_RL(epoch, 0, 'test', final_eval = True))

'''
print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)
'''

# Save encoder instead of full model
#torch.save(nli_net.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))