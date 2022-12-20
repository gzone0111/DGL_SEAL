#!/usr/bin/env python
# coding: utf-8

# In[168]:


# Process Data
import pandas as pd
import numpy as np
import torch as th
import dgl
from dgl.data import DGLDataset
import re
import gluonnlp as nlp
GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
data_df = pd.read_table('./dataset/ml-100k/u.data', header=None)
user_df = pd.read_table('./dataset/ml-100k/u.user',sep='|', header=None)
item_df = pd.read_csv('./dataset/ml-100k/u.item', sep='|', header=None,
                                          names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES_ML_100K,
                                          engine='python',encoding = "ISO-8859-1")


# In[243]:


class ML_100K(DGLDataset):
    def __init__(self):
        super().__init__(name="ML_100K")

    def process(self):
        self.GENRES_ML_100K =\
        ['unknown', 'Action', 'Adventure', 'Animation',
         'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
         'Thriller', 'War', 'Western']
        self.rating_info = pd.read_table('./dataset/ml-100k/u.data', header=None)
        # make user id start from 1
        self.rating_info[0] = self.rating_info[0]-1
        self.rating_info[1] = self.rating_info[1]-1
        
        self.user_info = pd.read_table('./dataset/ml-100k/u.user',sep='|', header=None,
                                    names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python')
        self.movie_info = pd.read_csv('./dataset/ml-100k/u.item', sep='|', header=None,
                                                  names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES_ML_100K,
                                                  engine='python',encoding = "ISO-8859-1")
        self.movie_info["id"] = self.movie_info["id"]-1
        # process user_feature
        # Belows data process reference to original STAR-GCN implementation
        # Copy from https://github.com/jennyzhang0215/STAR-GCN/blob/f975e5a0679cdd78bb7d7d75d49ef8090e7547e8/mxgraph/datasets.py#L93
        ages = self.user_info['age'].values.astype(np.float32)
        
        gender = (self.user_info['gender'] == 'F').values.astype(np.float32)
        
        all_occupations = set(self.user_info['occupation'])
        
        occupation_map = {ele: i for i, ele in enumerate(all_occupations)}
        
        occupation_one_hot = np.zeros(shape=(self.user_info.shape[0], len(all_occupations)),
                                      dtype=np.float32)
        occupation_one_hot[np.arange(self.user_info.shape[0]),
                           np.array([occupation_map[ele] for ele in self.user_info['occupation']])] = 1
        
        user_features = np.concatenate([ages.reshape((self.user_info.shape[0], 1)) / 50.0,
                                        gender.reshape((self.user_info.shape[0], 1)),
                                        occupation_one_hot], axis=1)
        
    
        # occupation_one_hot 943 user, 21 occupation
        # process movie_feature
        _word_embedding = nlp.embedding.GloVe('glove.840B.300d')
        _tokenizer = nlp.data.transforms.SpacyTokenizer()
        title_embedding = np.zeros(shape=(self.movie_info.shape[0], 300), dtype=np.float32)
        release_years = np.zeros(shape=(self.movie_info.shape[0], 1), dtype=np.float32)
        p = re.compile(r'(.+)\s*\((\d+)\)')
        for i, title in enumerate(self.movie_info['title']):
            match_res = p.match(title)
            if match_res is None:
                print('{} cannot be matched, index={}, name={}'.format(title, i, self._name))
                title_context, year = title, 1950
            else:
                title_context, year = match_res.groups()
            # We use average of glove
            title_embedding[i, :] =_word_embedding[_tokenizer(title_context)].asnumpy().mean(axis=0)
            release_years[i] = float(year)
        movie_features = np.concatenate((title_embedding,
                                         (release_years - 1950.0) / 100.0,
                                         self.movie_info[self.GENRES_ML_100K]),
                                        axis=1)
        data_dict = {
            ('user','1','movie'):(th.tensor(self.rating_info[self.rating_info[2]==1][0].values.tolist()),
                                  th.tensor(self.rating_info[self.rating_info[2]==1][1].values.tolist())),
            ('user','2','movie'):(th.tensor(self.rating_info[self.rating_info[2]==2][0].values.tolist()),
                                  th.tensor(self.rating_info[self.rating_info[2]==2][1].values.tolist())),
            ('user','3','movie'):(th.tensor(self.rating_info[self.rating_info[2]==3][0].values.tolist()),
                                  th.tensor(self.rating_info[self.rating_info[2]==3][1].values.tolist())),
            ('user','4','movie'):(th.tensor(self.rating_info[self.rating_info[2]==4][0].values.tolist()),
                                  th.tensor(self.rating_info[self.rating_info[2]==4][1].values.tolist())),
            ('user','5','movie'):(th.tensor(self.rating_info[self.rating_info[2]==5][0].values.tolist()),
                                  th.tensor(self.rating_info[self.rating_info[2]==5][1].values.tolist())),
        }
        # process edges
        n_users=943
        n_movies=1682
        graph = dgl.heterograph(data_dict)
        graph.nodes['user'].data['feature'] = th.tensor(user_features)
        graph.nodes['movie'].data['feature'] = th.tensor(movie_features)
        self.graphs = []
        self.graphs.append(graph)
        
#         self.graph = dgl.graph((torch.tensor(self.df_investments.funded_object_id.values.tolist()), 
#                                 torch.tensor(self.df_investments.investor_object_id.values.tolist()))).to(device)
    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)

