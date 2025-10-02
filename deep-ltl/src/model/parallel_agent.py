import numpy as np
import torch

import preprocessing
from model.model import Model
from sequence.search import SequenceSearch


class ParallelAgent:
    def __init__(self, model: Model, search: SequenceSearch, propositions: set[str], num_envs: int):
        self.model = model
        self.search = search
        self.propositions = propositions
        self.sequences = [None] * num_envs

    def update_dones(self, dones: np.ndarray):
        for i, done in enumerate(dones):
            if done:
                self.sequences[i] = None

    def get_action(self, obss, infos, deterministic=False) -> np.ndarray:
        for i, (obs, info) in enumerate(zip(obss, infos)):
            if 'ldba_state_changed' in info or self.sequences[i] is None:
                self.sequences[i] = self.search(obs['ldba'], obs['ldba_state'], obs)
            obs['goal'] = self.sequences[i]
        return self.forward(obss, deterministic)

    def forward(self, obss, deterministic=False) -> np.ndarray:
        assert isinstance(obss, list) or isinstance(obss, tuple)
        preprocessed = preprocessing.preprocess_obss(obss, self.propositions)
        with torch.no_grad():
            dist, value = self.model(preprocessed)
            action = dist.mode if deterministic else dist.sample()
        return action.detach().cpu().numpy()
