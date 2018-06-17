#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Global settings
n_test_authors = 15
n_training_authors = 35
k = 10
authors = [u'JanLopatka', u'WilliamKazer', u'MarcelMichelson', u'KirstinRidley', u'GrahamEarnshaw', u'MichaelConnor', u'MartinWolk', u'ToddNissen', u'PatriciaCommins', u'KevinMorrison', u'HeatherScoffield', u'BradDorfman', u'DavidLawder', u'KevinDrawbaugh', u'LynnleyBrowning', u'ScottHillis', u'FumikoFujisaki', u'TimFarrand', u'SarahDavison', u'AaronPressman', u'JohnMastrini', u'NickLouth', u'PierreTran', u'AlexanderSmith', u'MatthewBunce', u'KouroshKarimkhany', u'JimGilchrist', u'DarrenSchuettler', u'TanEeLyn', u'JoeOrtiz', u'MureDickie', u'EdnaFernandes', u'JoWinterbottom', u'RogerFillion', u'BenjaminKangLim', u"LynneO'Donnell", u'JonathanBirt', u'BernardHickey', u'RobinSidel', u'AlanCrosby', u'LydiaZajc', u'PeterHumphrey', u'KeithWeir', u'EricAuchard', u'TheresePoletti', u'KarlPenhaul', u'SimonCowell', u'JaneMacartney', u'SamuelPerry', u'MarkBendeich']
test_authors = [u'JanLopatka', u'WilliamKazer', u'MarcelMichelson', u'KirstinRidley', u'GrahamEarnshaw', u'MichaelConnor', u'MartinWolk', u'ToddNissen', u'PatriciaCommins', u'KevinMorrison', u'HeatherScoffield', u'BradDorfman', u'DavidLawder', u'KevinDrawbaugh', u'LynnleyBrowning']
training_authors = [x for x in authors if x not in test_authors]

# Settings
min_length = 165
voc_sizes = {'c1': {'en': 87, 'ar': 1839, 'es': 1805}, 'c2': {'en': 3201, 'ar': 31694, 'es': 30025}}
gender_to_idx = {'female': 0, 'male': 1}
idx_to_gender = {0: 'female', 1: 'male'}
country_to_idx = {'canada': 0, 'australia': 1, 'new zealand': 2, 'ireland': 3, 'great britain': 4, 'united states': 5}
idx_to_country = {0: 'canada', 1: 'australia', 2: 'new zealand', 3: 'ireland', 4: 'great britain', 5: 'united states'}

# Glove settings
glove_embedding_dim = 300

# CGFS settings
cgfs_epoch = 400
cgfs_input_dim = glove_embedding_dim
cgfs_mean = -4.56512329954
cgfs_std = 0.911449706065
cgfs_output_dim = {'c1': 30, 'c2': 60, 'c3': 90}
cgfs_lr = 0.001
cgfs_momentum = 0.9

# CCSAA Settings
ccsaa_epoch = 150
# ccsaa_epoch = 20
ccsaa_text_length = 20
ccsaa_voc_size = 86
ccsaa_output_dim = 150
ccsaa_mean = -4.56512329954
ccsaa_std = 0.911449706065
ccsaa_embedding_dim = 50
ccsaa_lr = 0.001
ccsaa_momentum = 0.9

