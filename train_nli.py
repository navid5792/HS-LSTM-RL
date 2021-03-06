# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import time
import argparse
import pickle
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import critic, actor
from read_data import get_MSRP_data, get_SICK_data, get_AI_data, get_SICK_tree_data, get_QQP_data

filename = "Results/inferSent_uni_SICKE_RL_actor"
samplecnt = 5
epsilon = 0.05
alpha = 0.1
print(filename)
parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--criticmodelname", type=str, default='modelBS5.pickle')
parser.add_argument("--outputmodelname", type=str, default='modelBS5actor.pickle')
parser.add_argument("--word_emb_path", type=str, default="dataset/glove.840B.300d.txt", help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--dpout_model", type=float, default=0.1, help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0.1, help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=5, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=1, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=600, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=150, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
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

"""
DATA
"""
#train, test, valid = get_nli(params.nlipath)

train, valid, test = get_SICK_data()


word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)


for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] + [word for word in sent.split() if word in word_vec] +
                                                                     ['</s>'] for sent in eval(data_type)[split]])

parser = argparse.ArgumentParser(description='Training Hyperparams')
# data loading params
parser.add_argument('-data_path', default = "data")

# network params
parser.add_argument('-d_model', type=int, default=300)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)
parser.add_argument('-d_ff', type=int, default=2048)
parser.add_argument('-n_heads', type=int, default=8)
parser.add_argument('-n_layers', type=int, default=2)
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-share_proj_weight', action='store_true')
parser.add_argument('-share_embs_weight', action='store_true')
parser.add_argument('-weighted_model', action='store_true') 

# training params
parser.add_argument('-lr', type=float, default=0.0002)
parser.add_argument('-max_epochs', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-max_src_seq_len', type=int, default=300)
parser.add_argument('-max_tgt_seq_len', type=int, default=300)
parser.add_argument('-max_grad_norm', type=float, default=None)
parser.add_argument('-n_warmup_steps', type=int, default=4000)
parser.add_argument('-display_freq', type=int, default=100)
parser.add_argument('-src_vocab_size', type=int, default=len(word_vec))
parser.add_argument('-tgt_vocab_size', type=int, default=len(word_vec))
parser.add_argument('-log', default=None)
parser.add_argument('-model_path', type=str, default = "")

transformer_opt = parser.parse_args()

"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
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

}

# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
criticModel = critic(config_nli_model, transformer_opt)
actorModel = actor(params.enc_lstm_dim, params.word_emb_dim)
print(criticModel)
print(actorModel)

#nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn1, optim_params1 = get_optimizer(params.optimizer)
critic_target_optimizer = optim_fn1(list(criticModel.target_pred.parameters()) + list(criticModel.target_classifier.parameters()), **optim_params1)
optim_fn2, optim_params2 = get_optimizer(params.optimizer)
critic_active_optimizer = optim_fn2(list(criticModel.active_pred.parameters()) + list(criticModel.active_classifier.parameters()), **optim_params2)

optim_fn3, optim_params3 = get_optimizer("adam,lr=0.1")
actor_target_optimizer = optim_fn3(actorModel.target_policy.parameters(), **optim_params3)
optim_fn4, optim_params4 = get_optimizer("adam,lr=0.1")
actor_active_optimizer = optim_fn4(actorModel.active_policy.parameters(), **optim_params4)

# cuda by default
criticModel.cuda()
actorModel.cuda()
loss_fn.cuda()

for name, x in criticModel.named_parameters():
    print(name)

for name, x in actorModel.named_parameters():
    print(name)

def Sampling_RL(current, summary, epsilon, Random = True):
    current_lower_state = torch.zeros(1, 2*params.enc_lstm_dim).cuda()
    current = current.squeeze(0)
    actions = []
    states = []
    length = int(current.size(0))
    for pos in range(length):
        predicted = actorModel.get_target_output(current_lower_state, current[pos], summary, scope = "target")
        states.append([current_lower_state, current[pos], summary])
        if Random:
            if random.random() > epsilon:
                action = (0 if random.random() < float(predicted[0].item()) else 1)
            else:
                action = (1 if random.random() < float(predicted[0].item()) else 0)
        else:
            action = np.argmax(predicted).item()
        actions.append(action)
        if action == 1:
            out_d, current_lower_state = criticModel.forward_lstm(current_lower_state, current[pos], scope = "target")
    Rinput = []
    for (i, a) in enumerate(actions):
        if a == 1:
            #Rinput.append(int(inputs[0][i].item())) ####
            Rinput.append(current[i])
    Rlength = len(Rinput)
    if Rlength == 0:
        print("problem")
    Rinput = torch.stack(Rinput).unsqueeze(0)
    
    return actions, states, Rinput, Rlength

    if Rlength == 0:
        actions[length-2] = 1
        Rinput.append(inputs[0][length-2])
        Rlength = 1
    Rinput += [1] * (maxlength - Rlength)

    Rinput = torch.tensor(Rinput).view(1,-1).cuda()
    
    return actions, states, Rinput, Rlength

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
    criticModel.train(False)
    if RL_train:
        actorModel.train()
    if LSTM_train:
        criticModel.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]


    critic_active_optimizer.param_groups[0]['lr'] = critic_active_optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else critic_active_optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(critic_active_optimizer.param_groups[0]['lr']))

    critic_target_optimizer.param_groups[0]['lr'] = critic_target_optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else critic_target_optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(critic_target_optimizer.param_groups[0]['lr']))

    #print(criticModel.target_pred.enc_lstm.bias_ih_l0)
    #print(criticModel.active_pred.enc_lstm.bias_ih_l0)
    criticModel.assign_active_network()
    actorModel.assign_active_network()
    #print(criticModel.target_pred.enc_lstm.bias_ih_l0)
    #print(criticModel.active_pred.enc_lstm.bias_ih_l0) 

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        totloss = 0.
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size], word_vec, params.word_emb_dim) 
        print(s1_batch.size(), s1_len) 
        asasas     
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        predict = torch.zeros(params.batch_size, params.n_classes).cuda()
        k = s1_batch.size(1)  # actual batch size
        s1_batch = s1_batch.transpose(0,1)
        s2_batch = s2_batch.transpose(0,1)
        position_a = torch.zeros(s1_batch.size(0), s1_batch.size(1)) # 16 21
        mask_a = torch.zeros(s1_batch.size(0), s1_batch.size(1))
        for i in range(s1_batch.size(0)): # 16
            j = 1
            torch.fill_(position_a[i], int(s1_len[i]))
            for k in range(s1_len[i]): #21
                position_a[i][k] = j
                j += 1
                mask_a[i][k] = 1
        position_a = position_a.long()
        position_b = torch.zeros(s2_batch.size(0), s2_batch.size(1)) # 16 21
        mask_b = torch.zeros(s2_batch.size(0), s2_batch.size(1))
        for i in range(s2_batch.size(0)): # 16
            j = 1
            torch.fill_(position_b[i], int(s2_len[i]))
            for k in range(s2_len[i]): #21
                position_b[i][k] = j
                j += 1
                mask_b[i][k] = 1
        position_b = position_b.long()
        position_a, position_b = Variable(position_a.cuda()), Variable(position_b.cuda())
        

        # model forward
        #output = criticModel((s1_batch, s1_len, position_a, mask_a.cuda()), (s2_batch, s2_len, position_b, mask_b.cuda()), "target")
        avgloss = 0
        aveloss = 0.
        total_norm = 0
        for kk in range(params.batch_size):
            left = s1_batch[kk].view(1,-1, 300)
            right = s2_batch[kk].view(1,-1, 300)
            left_len = s1_len[kk]
            right_len = s2_len[kk]
            left_position = position_a[kk].view(1,-1)
            right_position = position_b[kk].view(1,-1)
            left_mask = mask_a[kk].view(1,-1)
            right_mask = mask_b[kk].view(1,-1)
            tgt = tgt_batch[kk].view(-1)
            if RL_train:
                leftSummary = criticModel.summary(left)[-1]
                rightSummary = criticModel.summary(right)[-1]
                actionlist_left, actionlist_right, statelist_left, statelist_right, losslist = [], [], [], [], []
                aveLoss = 0.
                for i in range(samplecnt):
                    actions_left, states_left, Rinput_left, Rlength_left = Sampling_RL(left, rightSummary, epsilon, Random=True)
                    actions_right, states_right, Rinput_right, Rlength_right = Sampling_RL(right, leftSummary, epsilon, Random=True)
                    actionlist_left.append(actions_left)
                    statelist_left.append(states_left)
                    actionlist_right.append(actions_right)
                    statelist_right.append(states_right)
                    L = (Rinput_left, int(Rinput_left.size(1)), None, None)
                    R = (Rinput_right, int(Rinput_right.size(1)), None, None)
                    out = criticModel(L, R, scope = "target")
                    loss_ = loss_fn(out, tgt)
                    loss_ += (float(Rlength_left) / int(left.size(1))) **2 *0.15
                    loss_ += (float(Rlength_right) / int(right.size(1))) **2 *0.15
                    aveloss += loss_
                    losslist.append(loss_)
                aveloss /= samplecnt
                totloss += aveloss
                grad1 = None
                grad2 = None
                grad3 = None
                grad4 = None
                flag = 0 
                for i in range(samplecnt): #5
                    for pos in range(len(actionlist_left[i])): #19 --> 13
                        rr = [0, 0]
                        rr[actionlist_left[i][pos]] = ((losslist[i] - aveloss) * alpha).cpu().item()
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
                            grad3 += g[3]
                    for pos in range(len(actionlist_right[i])): # 25 --> 5
                        rr = [0, 0]
                        rr[actionlist_right[i][pos]] = ((losslist[i] - aveloss) * alpha).cpu().item()
                        g = actorModel.get_gradient(statelist_right[i][pos][0], statelist_right[i][pos][1], statelist_right[i][pos][2], rr, scope = "target")
                        grad1 += g[0]
                        grad2 += g[1]
                        grad3 += g[2]
                        grad3 += g[3]
                actor_target_optimizer.zero_grad()
                actor_active_optimizer.zero_grad()
                actorModel.assign_active_network_gradients(grad1, grad2, grad3, grad4)
                actor_active_optimizer.step()
            else:
                critic_active_optimizer.zero_grad()
                critic_target_optimizer.zero_grad()
                output = criticModel((left, left_len, left_position, left_mask.cuda()), (right, right_len, right_position , right_mask.cuda()), "target")
                predict[kk] = output
                loss = loss_fn(output, tgt)
                avgloss += loss.item()
                loss.backward()
                criticModel.assign_active_network_gradients()
                critic_active_optimizer.step()    
        if RL_train:
            actorModel.update_target_network()
        else:
            for name,p in criticModel.active_pred.named_parameters():
                if p.requires_grad:
                    p.grad.data.div_(params.batch_size)  # divide by the actual batch size
                    total_norm += p.grad.data.norm().item() ** 2
            for name,p in criticModel.active_classifier.named_parameters():
                if p.requires_grad:
                    p.grad.data.div_(params.batch_size)  # divide by the actual batch size
                    total_norm += p.grad.data.norm().item() ** 2
            total_norm = np.sqrt(total_norm)
            shrink_factor = 1
            if total_norm > params.max_norm:
                print("shrinking.............................")
                shrink_factor = params.max_norm / total_norm
            current_lr = critic_active_optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
            critic_active_optimizer.param_groups[0]['lr'] = current_lr * (shrink_factor) # just for update
            
            pred = predict.data.max(1)[1]
            correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
            assert len(pred) == len(s1[stidx:stidx + params.batch_size])
            all_costs.append(float(avgloss/params.batch_size))
            words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim
            # optimizer step
            criticModel.assign_target_network()
            if len(all_costs) == 100:
                logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                                stidx, round(np.mean(all_costs), 2),
                                int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                                int(words_count * 1.0 / (time.time() - last_time)),
                                round(100.* int(correct.data)/(stidx+k), 2)))
                print(logs[-1])
                last_time = time.time()
                words_count = 0
                all_costs = []
    
    if LSTM_train:
        train_acc = round(100 * int(correct.data)/len(s1), 2)
        print('results : epoch {0} ; mean accuracy train : {1}'.format(epoch, train_acc))
        return train_acc
    else:
        return None


def evaluate(epoch, eval_type='valid', final_eval=False):
    criticModel.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch

        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()
        
        s1_batch = s1_batch.transpose(0,1)
        s2_batch = s2_batch.transpose(0,1)

        position_a = torch.zeros(s1_batch.size(0), s1_batch.size(1)) # 16 21
        mask_a = torch.zeros(s1_batch.size(0), s1_batch.size(1))
        for i in range(s1_batch.size(0)): # 16
            j = 1
            torch.fill_(position_a[i], int(s1_len[i]))
            for k in range(s1_len[i]): #21
                position_a[i][k] = j
                j += 1
                mask_a[i][k] = 1
        position_a = position_a.long()

        position_b = torch.zeros(s2_batch.size(0), s2_batch.size(1)) # 16 21
        mask_b = torch.zeros(s2_batch.size(0), s2_batch.size(1))
        for i in range(s2_batch.size(0)): # 16
            j = 1
            torch.fill_(position_b[i], int(s2_len[i]))
            for k in range(s2_len[i]): #21
                position_b[i][k] = j
                j += 1
                mask_b[i][k] = 1
        position_b = position_b.long()
        
        position_a, position_b = Variable(position_a.cuda()), Variable(position_b.cuda())
        # model forward
        output = criticModel((s1_batch, s1_len, position_a, mask_a.cuda()), (s2_batch, s2_len, position_b, mask_b.cuda()), "target")

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * int(correct.data) / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))
        with open(filename, "a") as f:
            f.write(str(epoch)+ " " + str(eval_acc) + "\n")

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(criticModel.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
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


"""
Train model on Natural Language Inference task
"""
epoch = 10000
while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1
criticModel.load_state_dict(torch.load(os.path.join(params.outputdir, params.criticmodelname)))
print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)
print("Critic Load Done.......")

epoch = 1
RL_epoch = 10
while not stop_training and epoch <= RL_epoch:
    train_acc = trainepoch(epoch)
    asasas
    eval_acc = evaluate_RL(epoch, 'valid')
    epoch += 1

# Run best model on test set.
criticModel.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)

# Save encoder instead of full model
torch.save(criticModel.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))
