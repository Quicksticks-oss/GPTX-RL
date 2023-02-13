from GPTXRL.tokenizer import Tokenizer
from GPTXRL.model import Agent
from GPTXRL.enviroment import TextEnv
from multiprocessing import Process
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import random
import pickle
import torch

class Trainer:
    def __init__(self, train_path:str='train.txt', context_size=32, hidden_size=2048, epoch_set=4, epochs=512, steps=128, reruns=1):
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
        self.agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=self.n_actions, eps_min=0.001, input_dims=[self.observ], lr=0.0001, hidden_size=self.hidden_size)
        
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
    
    def load(self):
        with open(self.train_path, 'r') as f:
            self.train_data = self.tokenizer.tokenize(f.read())
        self.train_data = np.array_split(self.train_data, np.ceil(self.train_data.shape[0]/self.context_size))

    def get_block(self, specific_index=-1, location=-1):
        # Gets specific index.
        specific_chunk = random.choice(self.train_data)
        # Gets second chunk.
        choice = random.randint(2, self.context_size-1)
        end = np.ceil(len(specific_chunk/choice))
        split_chunk = random.choice(np.array_split(specific_chunk, choice))
        return split_chunk

    def train(self):
        env = TextEnv(self.context_size)
        scores = []
        total_steps = 0
        for epoch in range(self.epochs+1):
            tdq = tqdm(range(self.steps), unit='step', desc='Step',ncols=122)
            for s in tdq:
                score = 0
                done = False
                text_block = self.get_block()
                if len(text_block[:-1] < 2):
                    text_block = self.get_block()
                for _ in range(self.reruns):
                    observation = env.reset(text_block[:-1])
                    while not done:
                        action = self.agent.choose_action(observation)
                        observation_, reward, done, won, n_games = env.step(action, text_block[-1:])
                        score += reward
                        self.agent.store_transition(observation, action, reward, observation_, done)
                        loss = self.agent.learn()
                        observation = observation_
                        total_steps += 1
                    scores.append(score)
                    avg_score = np.mean(scores[-100:])
                tdq.set_description('Epoch ('+str(epoch)+'/'+str(self.epochs)+') Step')
                tdq.set_postfix({"Epsilon": str(round(self.agent.epsilon, 4)), 'Average Score': round(avg_score), 'Won':won, 'Total': total_steps})
            if epoch % self.epoch_set == 0:
                print('Epoch Set Complete! - Last Score:', score, 'Average:', round(avg_score), 'Epsilon', round(self.agent.epsilon, 4), 'Won', won, 'N_Games', n_games, 'Loss', loss if loss == None else round(loss, 4), 'Total:', total_steps)
                print('Saving Model...')
                self.save_model()
                self.test_run()
        self.inference()

    def save_model(self):
        pickled = pickle.dumps(self.agent)
        with open('GPTX-RL-'+self.format_params()+'.pkl', 'wb+') as f:
            f.write(pickled)

    def test_run(self):
        print('==== Running Test Run ====')
        text_block = self.get_block()
        env = TextEnv(self.context_size)
        score= 0
        observation = env.reset(text_block[:-1])

        action = self.agent.choose_action(observation)
        observation_, reward, done, won, n_games = env.step(action, text_block[-1:][0], inference=True)
        score += reward

        genned = np.append(text_block[:-1], action)

        print('Should:', self.tokenizer.detokenize(text_block))
        print('Genned:', self.tokenizer.detokenize(genned))
        print('Gen on:', self.tokenizer.detokenize(text_block[:-1]))
        print('Test Run', 'Score', score, 'Epsilon', self.agent.epsilon, 'Won', won)
        print('==== Test Complete ====')

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
    trainer = Trainer()
    trainer.load()
    trainer.print_model_stats()
    trainer.train()

if __name__ == '__main__':
    main()
