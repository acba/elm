#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    test_xor
    --------

"""


import elm


def test_save_load_elmk():

    try:
        elmk = elm.ELMRandom()

        file_name = "reg.tmp"
        elmk.save_model(file_name)
        elmk.load_model(file_name)
        
    except:
        ERROR = 1
    else:
        ERROR = 0

    assert (ERROR == 0)


def test_save_load_elmr():

    try:
        elmr = elm.ELMRandom()

        file_name = "reg.tmp"
        elmr.save_model(file_name)
        elmr.load_model(file_name)

    except:
        ERROR = 1
    else:
        ERROR = 0

    assert (ERROR == 0)
