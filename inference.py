from GPTXRL.tokenizer import Tokenizer
from GPTXRL.model import Agent
from GPTXRL.enviroment import TextEnv
from multiprocessing import Process
from tqdm import tqdm
import numpy as np
import argparse
import random
import pickle
import torch

class Trainer:
    def __init__(self, train_path:str='train.txt', context_size=32, hidden_size=512, epoch_set=4, epochs=4096, steps=512, reruns=2):
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.epoch_set = epoch_set
        self.epochs = epochs
        self.steps = steps
        self.reruns = reruns

        self.tokenizer = Tokenizer()
        self.tokenizer.input_size = self.context_size
        self.tokenizer.load_from_file('vocab.json')

        self.n_actions = len(self.tokenizer.vocab)
        self.observ = self.context_size
        self.agent = Agent(gamma=0.99, epsilon=1.0, batch_size=256, n_actions=self.n_actions, eps_min=0.01, input_dims=[self.observ], lr=0.001, hidden_size=self.hidden_size)
        
        self.pytorch_total_params = sum(p.numel() for p in self.agent.Q_eval.parameters())
        self.pytorch_total_params_trainable = sum(p.numel() for p in self.agent.Q_eval.parameters() if p.requires_grad)
        
        self.train_path = train_path
        self.train_data = None

    def format_params(self):
        if self.pytorch_total_params > 1000000000000:
            return str(int(self.pytorch_total_params/1000000000000))+'M'
        elif self.pytorch_total_params > 1000000000:
            return str(int(self.pytorch_total_params/1000000000))+'M'
        elif self.pytorch_total_params > 1000000:
            return str(int(self.pytorch_total_params/1000000))+'M'
        elif self.pytorch_total_params > 1000:
            return str(int(self.pytorch_total_params/1000))+'K'
        return str(self.pytorch_total_params)

    def print_model_stats(self):
        print('==== GPTX-RL Training Script ====')
        print()
        print(' Model Parameters')
        print('   * Total:', self.pytorch_total_params)
        print('   * Train:', self.pytorch_total_params_trainable)
        print('   * Forma:', self.format_params())
        print('   * Hiddn:', self.hidden_size)
        print('   * Contx:', self.context_size)
        print('   * Acton:', self.n_actions)
        print(' Dataset Details')
        print('   * Vocab:', len(self.tokenizer.vocab))
        print('   * Train:', len(self.train_data))
        print(' Train Settings')
        print('   * Epoch:', self.epochs)
        print('   * Rerun:', self.reruns)
        print('   *  Step:', self.steps)
        print('   * Total:', self.steps*self.epochs)
        print('   *   Set:', self.epoch_set)
        print()
        print('=========== ======== ===========')
        print()
    
    def load(self, model:str):
        print('Loading model.')
        with open(model, 'rb') as f:
            self.agent = pickle.loads(f.read())

    def inference(self):
        while True:
            print('==== Running Inference ====')
            text_block = self.tokenizer.tokenize(input(' >> '), should_continue=True)
            env = TextEnv(self.context_size)
            score= 0
            print(text_block)
            observation = env.reset(text_block)
            action = self.agent.choose_action(observation)
            observation_, reward, done, won, n_games = env.step(action, text_block[-1:][0], inference=True)
            score += reward
            self.agent.store_transition(observation, action, reward, observation_, done)
            loss = self.agent.learn()
            observation = observation_
            genned = np.append(text_block, action)
            print('Genned     ', genned)
            print('Detokenized', self.tokenizer.detokenize(genned))
            print('==== Inference Complete ====')

def main():
    parser = argparse.ArgumentParser(description="GPTX-RL Inference Runner v1.0")
    parser.add_argument("-i", "--input", default=None, help = "Input model file")
    args = parser.parse_args()
    trainer = Trainer()
    trainer.load(args.input)
    trainer.print_model_stats()
    trainer.train()

if __name__ == '__main__':
    main()
