% ==============================================================================
% AGRO-SMART ADVISOR: KNOWLEDGE BASE (Prolog)
% ==============================================================================

% --- DICHIARAZIONE FATTI DINAMICI ---
:- dynamic dati_lotto/7.

% --- 1. ONTOLOGIA DELLE COLTURE ---
% Gerarchia: categoria(NomeColtura, Famiglia).
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
terreno_acido(ID) :- dati_lotto(ID, _, _, _, PH, _, _), PH < 5.5.
terreno_alcalino(ID) :- dati_lotto(ID, _, _, _, PH, _, _), PH > 7.5.
terreno_arido(ID) :- dati_lotto(ID, _, _, _, _, Rain, _), Rain < 400.
carenza_azoto(ID) :- dati_lotto(ID, N, _, _, _, _, _), N < 20.
clima_freddo(ID) :- dati_lotto(ID, _, _, _, _, _, Temp), Temp < 15.

% --- 3. REGOLE DI INCOMPATIBILITÀ (Constraint Checking) ---
incompatibile(ID, rice) :- terreno_arido(ID).
incompatibile(ID, jute) :- terreno_arido(ID).
incompatibile(ID, Pianta) :- categoria(Pianta, leguminosa), terreno_acido(ID).
incompatibile(ID, orange) :- terreno_acido(ID).
incompatibile(ID, coffee) :- terreno_alcalino(ID).
incompatibile(ID, maize)  :- carenza_azoto(ID).
incompatibile(ID, cotton) :- carenza_azoto(ID).
incompatibile(ID, Pianta) :- categoria(Pianta, frutto_tropicale), clima_freddo(ID).

% --- 4. INTERFACCE DI RAGIONAMENTO ---

% A. Validazione Semplice
valida_raccomandazione(ID, Pianta) :-
    \+ incompatibile(ID, Pianta).

% B. Semantic Recovery (Suggerimento Alternativo)
% Se la pianta P suggerita è incompatibile, trova una pianta A che:
% 1. Appartiene alla stessa categoria di P (es. entrambi cereali)
% 2. NON è incompatibile con il terreno
% 3. È diversa da P
suggerisci_alternativa(ID, PiantaScartata, Alternativa) :-
    incompatibile(ID, PiantaScartata),          % Assicuriamoci che la prima sia davvero vietata
    categoria(PiantaScartata, Categoria),       % Prendi la categoria (es. cereale)
    categoria(Alternativa, Categoria),          % Trova un'altra pianta della stessa categoria
    PiantaScartata \= Alternativa,              % Che non sia la stessa
    \+ incompatibile(ID, Alternativa).          % E che sia valida per questo terreno

% --- VINCOLI SPAZIALI (Per A*) ---
no_fly_zone('Lotto_B').
attraversabile(Nodo) :- \+ no_fly_zone(Nodo).