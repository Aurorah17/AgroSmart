% ==============================================================================
% AGRO-SMART ADVISOR: KNOWLEDGE BASE (Prolog)
% Autori: [Aurora Pesare & Fabrizio Stimolo]
% Descrizione: Base di conoscenza per la validazione logica
% ==============================================================================

% --- DICHIARAZIONE FATTI DINAMICI ---
% Struttura: dati_lotto(ID, N, P, K, PH, Rainfall, Temperature).
:- dynamic dati_lotto/7.

% --- 1. ONTOLOGIA DELLE COLTURE ---
categoria(rice, cereale).
categoria(maize, cereale).
categoria(chickpea, leguminosa).
categoria(kidneybeans, leguminosa).
categoria(pigeonpeas, leguminosa).
categoria(mothbeans, leguminosa).
categoria(mungbean, leguminosa).
categoria(blackgram, leguminosa).
categoria(lentil, leguminosa).
categoria(pomegranate, frutto).
categoria(banana, frutto_tropicale).
categoria(mango, frutto_tropicale).
categoria(grapes, frutto).
categoria(watermelon, frutto_estivo).
categoria(muskmelon, frutto_estivo).
categoria(apple, frutto_temperato).
categoria(orange, agrumi).
categoria(papaya, frutto_tropicale).
categoria(coconut, frutto_tropicale).
categoria(cotton, industriale).
categoria(jute, industriale).
categoria(coffee, industriale).

% --- 2. REGOLE DI DOMINIO (Background Knowledge) ---

% Regola: Terreno ACIDO (pH < 5.5)
terreno_acido(ID) :-
    dati_lotto(ID, _, _, _, PH, _, _),
    PH < 5.5.

% Regola: Terreno ALCALINO (pH > 7.5)
terreno_alcalino(ID) :-
    dati_lotto(ID, _, _, _, PH, _, _),
    PH > 7.5.

% Regola: Terreno ARIDO (Pioggia < 400mm)
terreno_arido(ID) :-
    dati_lotto(ID, _, _, _, _, Rain, _),
    Rain < 400.

% Regola: Terreno POVERO DI AZOTO (N < 20)
carenza_azoto(ID) :-
    dati_lotto(ID, N, _, _, _, _, _),
    N < 20.

% Regola: CLIMA FREDDO (Temp < 15°C)
clima_freddo(ID) :-
    dati_lotto(ID, _, _, _, _, _, Temp),
    Temp < 15.

% --- 3. REGOLE DI VALIDAZIONE (Constraint Checking) ---

% VINCOLO 1: Riso e Juta richiedono molta acqua.
incompatibile(ID, rice) :- terreno_arido(ID).
incompatibile(ID, jute) :- terreno_arido(ID).

% VINCOLO 2: Legumi e Agrumi (Orange) soffrono la acidità.
incompatibile(ID, Pianta) :- categoria(Pianta, leguminosa), terreno_acido(ID).
incompatibile(ID, orange) :- terreno_acido(ID).

% VINCOLO 3: Il Caffè odia i terreni alcalini.
incompatibile(ID, coffee) :- terreno_alcalino(ID).

% VINCOLO 4: Mais e Cotone richiedono molto Azoto.
incompatibile(ID, maize)  :- carenza_azoto(ID).
incompatibile(ID, cotton) :- carenza_azoto(ID).

% VINCOLO 5: Le piante tropicali soffrono il freddo.
incompatibile(ID, Pianta) :- 
    categoria(Pianta, frutto_tropicale), 
    clima_freddo(ID).

% --- 4. INTERFACCIA DI VALUTAZIONE ---
valida_raccomandazione(ID, Pianta) :-
    \+ incompatibile(ID, Pianta).

% --- VINCOLI SPAZIALI (Per il Pruning di A*) ---
% Simuliamo una zona dove il drone non può passare
no_fly_zone('Lotto_B'). 
no_fly_zone('Lotto_C').

% Un nodo è attraversabile solo se NON è in una no_fly_zone
attraversabile(Nodo) :- \+ no_fly_zone(Nodo).