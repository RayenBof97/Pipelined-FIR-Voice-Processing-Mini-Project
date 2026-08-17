import numpy as np
import pyaudio 
import time 

BLOCK_SIZE = 256                #   On traite les données par blocs de 256 échantillons
SAMPLING_RATE = 44100           #   Fréquence d'échantillonnage (Hz)
CHANNELS = 2                    #   Nombre de canaux (Certains microphones sont stéréo, c'est à dire ils ont 2 capsules, et donc 2 canaux de données: gauche et droite)

#----------------------------------------------------------------------------------
#- @brief : Calcule les coefficients sinc du filtre idéal
#- @param k : indices des coefficients
#- @param fcr : fréquence de coupure réduite (fc / fe)
#- @return : tableau des coefficients g(k) = 2·fcr·sinc(2·fcr·k)
#----------------------------------------------------------------------------------
def sinc_coeff(k, fcr):
    """FFT-1 (G(f)) = g(k) = 2·fcr·sinc(2·fcr·k) avec sinc(x) = sin(πx)/(πx) et sinc(0)=1."""
    return 2 * fcr * np.sinc(2 * fcr * k)

#----------------------------------------------------------------------------------
#- @brief : Construit les coefficients h(n) d'un filtre FIR causal de longueur N
#- @param N : ordre du filtre (doit être impair)
#- @param fcr : fréquence de coupure réduite (fc / fe)
#- @param window : type de fenêtre ('rect' ou 'hamming')
#- @return h : coefficients causaux du filtre
#----------------------------------------------------------------------------------
def make_fir(N, fcr, window):
    """Construit h(n) causal de longueur N (N impair).
    Retourne h (coefficients causaux), gw (réponse non-causale), k_nc (indices non-causaux)."""
    assert N % 2 == 1, 'N doit être impair'
    M  = (N - 1) // 2
    k  = np.arange(-M, M + 1)       # indices non-causaux
    g  = sinc_coeff(k, fcr)          # réponse idéale
    if window == 'hamming':
        w = 0.54 + 0.46 * np.cos(2 * np.pi * k / (N - 1)) # fenêtre de Hamming
    else:                             # rectangulaire
        w = np.ones(N)
    gw = g * w                        # troncature
    h  = gw.copy()                    # h(n) causal = gw décalé de M (même tableau)
    return h, gw, k

#----------------------------------------------------------------------------------
#- @brief : Construit un filtre passe-bande FIR par différence de deux passe-bas
#- @param N : ordre du filtre (doit être impair)
#- @param f_low : fréquence basse de la bande (Hz)
#- @param f_high : fréquence haute de la bande (Hz)
#- @param window : type de fenêtre ('rect' ou 'hamming')
#- @return h_bp : coefficients du filtre passe-bande
#----------------------------------------------------------------------------------
def make_bandpass_fir(N, f_low, f_high, window):
    fcr_low  = f_low  / SAMPLING_RATE       # Fréquence réduite basse
    fcr_high = f_high / SAMPLING_RATE       # Fréquence réduite haute
    h_low,  _, _ = make_fir(N, fcr_low,  window=window)
    h_high, _, _ = make_fir(N, fcr_high, window=window)
    return h_high - h_low               # Différence des deux passe-bas => passe-bande

#----------------------------------------------------------------------------------
#- @brief : Applique un filtre FIR sur un bloc audio avec gestion de l'overlap entre blocs
#- @param block : tableau numpy (BLOCK_SIZE, CHANNELS) du bloc audio courant
#- @param h : coefficients du filtre FIR (précalculés une seule fois)
#- @param overlap_buf : buffer d'overlap entre blocs de forme (N-1, CHANNELS),
#-                      nécessaire pour la continuité de la convolution entre blocs
#- @return filtered_block : bloc filtré de forme (BLOCK_SIZE, CHANNELS)
#- @return overlap_buf : buffer d'overlap mis à jour pour le prochain bloc
#----------------------------------------------------------------------------------
def apply_fir(block, h, overlap_buf):
    N_fir = len(h)
    filtered_block = np.zeros_like(block)

    # Concaténer l'overlap du bloc précédent avec le bloc courant
    # pour assurer la continuité de la convolution aux bords des blocs
    extended = np.concatenate([overlap_buf, block], axis=0)   # forme : (N-1 + BLOCK_SIZE, CHANNELS)

    #Convolution pour chaque channel 
    for channel in range(CHANNELS):
        for n in range(BLOCK_SIZE):
            for k in range(N_fir):
                filtered_block[n, channel] += h[k] * extended[n + (N_fir - 1) - k, channel]

    # Mettre à jour l'overlap avec les N-1 derniers échantillons du bloc courant
    overlap_buf = block[-(N_fir - 1):, :]

    return filtered_block, overlap_buf

#----------------------------------------------------------------------------------
#- @brief : Détecteur d'enveloppe — redressement demi-onde + filtre passe-bas
#- @param block : tableau numpy (BLOCK_SIZE, CHANNELS) du bloc audio courant
#- @param overlap_buf : buffer d'overlap pour le filtre passe-bas de lissage
#- @return envelope : enveloppe du signal (BLOCK_SIZE, CHANNELS)
#- @return overlap_buf : buffer d'overlap mis à jour
#----------------------------------------------------------------------------------
def envelope_detector(block, h_env, overlap_buf_env):
    # Redressement demi-onde : on ne garde que les valeurs positives
    half_wave = np.maximum(block, 0.0)
    # Lissage par filtre passe-bas pour extraire l'enveloppe
    envelope, overlap_buf_env = apply_fir(half_wave, h_env, overlap_buf_env)
    return envelope, overlap_buf_env

#----------------------------------------------------------------------------------
#- @brief : Générateur de signal carré — produit un bloc de signal carré de fréquence fp et d'amplitude A0
#- @param block_index : index du bloc courant (pour calculer la phase absolue et éviter les discontinuités)
#- @param fp : fréquence du signal carré (Hz)
#- @param A0 : amplitude du signal carré
#- @return square_block : tableau numpy (BLOCK_SIZE, CHANNELS) contenant le signal carré
#----------------------------------------------------------------------------------
def square_wave_generator(block_index, fp, A0):
    # Calculer les indices temporels absolus pour ce bloc (pour éviter les discontinuités entre blocs)
    n_start = block_index * BLOCK_SIZE
    n = np.arange(n_start, n_start + BLOCK_SIZE)
    t = n / SAMPLING_RATE                               # Temps absolu en secondes
    # Signal carré via le signe d'un sinus
    square = A0 * np.sign(np.sin(2 * np.pi * fp * t)).astype(np.float32)
    # Répliquer sur les deux canaux
    square_block = np.column_stack([square, square])    # forme : (BLOCK_SIZE, CHANNELS)
    return square_block

#----------------------------------------------------------------------------------
#- @brief : Fonction pour convertir les données brutes (Raw data) en un tableau numpy de forme (Canal : R , Canal : L)
#- @param raw_data : données brutes (Raw data) reçues du microphone, de type bytes
#- @return usable_array : tableau numpy de forme (Canal : R , Canal : L) contenant les données des 2 canaux séparées
#----------------------------------------------------------------------------------
def get_usable_array(raw_data):
    # Make input buffer compatible with numpy
    audio_data_1 = np.frombuffer(raw_data, dtype=np.float32)                # Convertir les données brutes des échantillons en un tableau numpy de type float32
    usable_array = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)       # Créer un tableau numpy vide de la taille (256, 2) pour stocker les données des 2 canaux de manière séparée
    #Iterate for each channel and fill the usable array
    for channel in range(CHANNELS):
        usable_array[:, channel] = audio_data_1[channel::CHANNELS]          # Les canaux sont intercalés LRLRLRL... donc on va mettre chaque canal dans une colonne séparée du tableau  
    
    return usable_array

#----------------------------------------------------------------------------------
#- @brief : Fonction de callback pour le stream audio, appelée à chaque fois que le microphone reçoit un bloc de données (256 échantillons)
#- @param in_data : données brutes (Raw data) reçues du microphone, de type bytes
#- @param frame_count : nombre de trames à traiter
#- @param time_info : informations sur le temps
#- @param status : statut du stream
#- @return : tuple (out_data, flag) où out_data est les données de sortie à envoyer à la carte son (après traitement) et flag indique si le stream doit continuer ou s'arrêter
#----------------------------------------------------------------------------------
def callback_process(in_data, frame_count, time_info, status):
    global overlap_bp_in, overlap_env, overlap_bp_out, block_index

    input_data = get_usable_array(in_data)   # Convertir les données brutes en un tableau numpy de forme (Canal : R , Canal : L)

    # Signal processing routine BEGINS HERE
    # ── Étape 1 : Filtre passe-bande d'entrée [fb1, fh1] ──────────────────────
    bp_in, overlap_bp_in = apply_fir(input_data, h_bp_in, overlap_bp_in)

    # ── Étape 2 : Détecteur d'enveloppe (redressement + lissage passe-bas) ────
    envelope, overlap_env = envelope_detector(bp_in, h_env, overlap_env)

    # ── Étape 3 : Génération du signal carré (fp, A0) et modulation ───────────
    square = square_wave_generator(block_index, FP, A0)
    modulated = envelope * square                            # Multiplication : enveloppe × signal carré

    # ── Étape 4 : Filtre passe-bande de sortie [fb2, fh2] ────────────────────
    bp_out, overlap_bp_out = apply_fir(modulated, h_bp_out, overlap_bp_out)

    # ── Étape 5 : Re-mélange avec le signal original ──────────────────────────
    out_data = (input_data + bp_out).astype(np.float32)

    block_index += 1

    return out_data.tobytes(), pyaudio.paContinue   # Convertir le tableau numpy de sortie en bytes pour pouvoir les renvoyer à la carte son (paContinue pour que ça continue en repeating)

#----------------------------------------------------------------------------------
# - Main function 
# - Create a Bride between python and PortAudio library (pyaudio) 
# - Get sound devices information 
# - Open an input audio stream with the callback function defined above
# - Open and output audio stream to send the processed data to the speakers
# - Close the streams and terminate pyaudio when done
# ----------------------------------------------------------------------------------
def main():
    global h_bp_in, h_bp_out, h_env
    global overlap_bp_in, overlap_env, overlap_bp_out
    global block_index
    global FP, A0

    # ── Paramètres du pipeline (valeurs de test du PDF) ───────────────────────
    N_fir = 21          # Ordre des filtres FIR (impair)
    FB1   = 100         # Fréquence basse filtre d'entrée (Hz)
    FH1   = 300         # Fréquence haute filtre d'entrée (Hz)
    FB2   = 100         # Fréquence basse filtre de sortie (Hz)
    FH2   = 800         # Fréquence haute filtre de sortie (Hz)
    FP    = 500         # Fréquence du signal carré modulant (Hz)
    A0    = 2.0         # Amplitude du signal carré modulant

    # ── Conception des filtres FIR ────────────────────────────────────────────
    h_bp_in  = make_bandpass_fir(N_fir, FB1, FH1, window='hamming')    # Filtre passe-bande d'entrée  [fb1, fh1]
    h_bp_out = make_bandpass_fir(N_fir, FB2, FH2, window='hamming')    # Filtre passe-bande de sortie [fb2, fh2]
    h_env    = make_fir(N_fir, 50 / SAMPLING_RATE, window='hamming')[0] # Filtre passe-bas de lissage pour le détecteur d'enveloppe (coupure 50 Hz)

    # ── Initialisation des buffers d'overlap à zéro ───────────────────────────
    overlap_bp_in  = np.zeros((N_fir - 1, CHANNELS), dtype=np.float32)
    overlap_env    = np.zeros((N_fir - 1, CHANNELS), dtype=np.float32)
    overlap_bp_out = np.zeros((N_fir - 1, CHANNELS), dtype=np.float32)
    block_index    = 0  # Index du bloc courant pour le générateur de signal carré

    p = pyaudio.PyAudio()   # Create a PyAudio instance to interface with PortAudio

    #Get default input and output devices information and print them
    info_in = p.get_default_input_device_info()   # Get default input device information
    info_out = p.get_default_output_device_info() # Get default output device information
    name_in = info_in['name']   # Get the name of the default input device
    idx_in = info_in['index']   # Get the index of the default input device
    name_out = info_out['name'] # Get the name of the default output device
    idx_out = info_out['index'] # Get the index of the default output device
    print(f"Default Input Device: {name_in} (Index: {idx_in})")   # Print the name and index of the default input device
    print(f"Default Output Device: {name_out} (Index: {idx_out})") # Print the name and index of the default output device

    print("\nAvailable microphones:\n")

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"Index {i} : {info['name']}")

    idx_in = int(input("\nSelect microphone index: "))
   
    # Open an input stream with the callback function defined above
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=SAMPLING_RATE,
        frames_per_buffer=BLOCK_SIZE,
        input=True,
        output=True,
        input_device_index=idx_in,   # Use the default input device
        output_device_index=idx_out, # Use the default output device
        stream_callback=callback_process
        )
    print('Audio stream opened ! Press Ctrl+C to stop.')
    #Keep the stream active and processing until interrupted by the user (Ctrl+C)
    try:
        while stream.is_active():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    stream.close() 
    print('Audio stream closed. Terminating PyAudio.')
    p.terminate()
    
if __name__ == "__main__":
    main()
