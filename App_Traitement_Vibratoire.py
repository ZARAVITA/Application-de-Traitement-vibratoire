# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:19:47 2025

@author: ZARAVITA Haydar
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# Fonction pour charger les données
def load_data(file):
    df = pd.read_csv(file, skiprows=1)  # Suppression de la première ligne
    df.columns = ["time", "amplitude"]
    df["time"] = df["time"] / 1000  # Conversion en secondes
    return df

# Fonctions de filtrage
def filtre_pass_haut(data, freq_coupure, fs):
    freq_nyquist = 0.5 * fs
    freq_normalisee = freq_coupure / freq_nyquist
    b, a = butter(4, freq_normalisee, btype='high', analog=False)
    return filtfilt(b, a, data)

def filtre_passe_bas(data, freq_coupure, fs):
    freq_nyquist = 0.5 * fs
    freq_normalisee = freq_coupure / freq_nyquist
    b, a = butter(4, freq_normalisee, btype='low', analog=False)
    return filtfilt(b, a, data)

def redressement(data):
    return np.abs(data)

# Interface Streamlit
st.title("Analyse Vibratoire - Angelico & ZARAVITA")
st.write("Application interactive pour l'analyse vibratoire d'un signal")

# Upload du fichier CSV
uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    
    # Fréquence d'échantillonnage
    dt = np.diff(df["time"])  
    fs = 1 / np.mean(dt)  
    
    # Affichage des premières lignes du dataset
    if st.checkbox("Afficher les 5 premières lignes du dataset"):
        st.write(df.head())
    
    # Affichage du signal original
    if st.checkbox("Afficher le signal original"):
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["amplitude"], label='Signal Original')
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)
    
    # Paramètres utilisateur
    freq_coupure_haut = st.slider("Fréquence de coupure du filtre passe-haut", 10, 1000, 500)
    freq_coupure_bas = st.slider("Fréquence de coupure du filtre passe-bas", 10, 1000, 200)
    
    # Traitements
    signal_filtre_haut = filtre_pass_haut(df["amplitude"], freq_coupure_haut, fs)
    signal_redresse = redressement(signal_filtre_haut)
    signal_filtre_bas = filtre_passe_bas(signal_redresse, freq_coupure_bas, fs)
    
    # Affichage du signal après traitement
    if st.checkbox("Afficher le signal après traitement"):
        fig, ax = plt.subplots()
        ax.plot(df["time"], signal_filtre_bas, label='Signal après traitement', color='red')
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)
    
    # Spectre FFT
    n = len(signal_filtre_bas)
    valeur_fft = fft(signal_filtre_bas)
    frequencies = fftfreq(n, d=1/fs)[:n//8]
    fft_magnitudes = np.abs(valeur_fft)[:n//8]
    
    if st.checkbox("Afficher le spectre FFT"):
        fig, ax = plt.subplots()
        ax.plot(frequencies, fft_magnitudes, label='Spectre FFT', color='green')
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)

st.write("Projet réalisé par Angelico & ZARAVITA")
