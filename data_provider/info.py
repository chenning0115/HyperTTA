import numpy as np
import pandas as pd
import os, sys

def get_spetral_info(data_sign):
    if data_sign == "Indian":
        spe_start, spe_end, band_width = 400, 2500, 10
        return spe_start, spe_end, band_width
    elif data_sign == "Pavia":
        spe_start, spe_end, band_width = 430, 860, 4
        return spe_start, spe_end, band_width
    elif data_sign == "Salinas":
        spe_start, spe_end, band_width = 400, 2500, 10
        return spe_start, spe_end, band_width
    elif data_sign == "Honghu" or data_sign == "WH":
        spe_start, spe_end, band_width = 400, 1000, 2
        return spe_start, spe_end, band_width
    else:
        assert Exception("not implement")

def get_rgb_index(spe_start, spe_end, band_width):
    red_spe = (620 + 750) // 2
    green_spe = (495 + 570) // 2
    blue_spe = (450 + 495) // 2
    r = (red_spe - spe_start) // band_width
    g = (green_spe - spe_start) // band_width
    b = (blue_spe - spe_start) // band_width

    return r, g, b


def get_rgb_index_by_sign(data_sign):
    spe_start, spe_end, band_width = get_spetral_info(data_sign)
    r, g, b = get_rgb_index(spe_start, spe_end, band_width)
    return r, g, b

noise_type_list = ['jpeg', 'zmguass', 'additive', 'poisson', 'salt_pepper', 'stripes', 'deadlines', 'kernal', 'thin_fog', 'thick_fog']


Indian_kvs_params = {
    "jpeg" : {
            'compression_ratios' : [10]
        },

    "zmguass": {
            "sigma": 0.3
    },

    "additive": {
            "noise_level": 0.3 
    },

    "poisson": {
            "snr_db" : 7
    },

    "salt_pepper": {
            "amount": 0.15 
    },

    "stripes": {
            "start": 3,
            "end" : 5
    },

    "deadlines":{
        "start": 3,
        "end" : 5
    },

    "kernal": {
            # "kernal_size": 10 # PU
            "kernal_size": 2
    },

    "thin_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.3,
            'alpha': 0.01,
            'beta': 1.0
    },

    "thick_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.5,
            'alpha': 0.01,
            'beta': 1.0
    }
}

Pavia_kvs_params = {
    "jpeg" : {
            'compression_ratios' : [10]
        },

    "zmguass": {
            "sigma": 0.3
    },

    "additive": {
        #     "noise_level": 0.3 
            "noise_level": 0.2
    },

    "poisson": {
            "snr_db" : 13 
    },

    "salt_pepper": {
        #     "amount": 0.15 
            "amount": 0.10 
    },

    "stripes": {
            "start": 30,
            "end" : 35
    },

    "deadlines":{
        "start": 30,
        "end" : 35
    },

    "kernal": {
            "kernal_size": 10 # PU
    },

    "thin_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.2,
            'alpha': 0.01,
            'beta': 1.0
    },

    "thick_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.3,
            'alpha': 0.01,
            'beta': 1.0
    }
}

WH_kvs_params = {
    "jpeg" : {
            'compression_ratios' : [10]
        },

    "zmguass": {
            "sigma": 0.3
    },

    "additive": {
            "noise_level": 0.3 
    },

    "poisson": {
            "snr_db" : 9 
    },

    "salt_pepper": {
            "amount": 0.15 
    },

    "stripes": {
            "start": 30,
            "end" : 35
    },

    "deadlines":{
        "start": 30,
        "end" : 35
    },

    "kernal": {
            "kernal_size": 5 # PU
    },

    "thin_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.3,
            'alpha': 0.01,
            'beta': 1.0
    },

    "thick_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.5,
            'alpha': 0.01,
            'beta': 1.0
    }
}