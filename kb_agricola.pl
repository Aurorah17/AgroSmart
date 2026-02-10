% ==============================================================================
% AGRO-SMART ADVISOR: KNOWLEDGE BASE (Prolog)
% Autori: [Aurora Pesare & Fabrizio Stimolo]
% Descrizione: Base di conoscenza per la validazione logica delle predizioni ML.
% ==============================================================================

% --- DICHIARAZIONE FATTI DINAMICI ---
% Questi predicati verranno asseriti (iniettati) da Python a runtime.
% Struttura: dati_lotto(ID, N, P, K, PH, Rainfall, Temperature).
:- dynamic dati_lotto/7.

% --- 1. ONTOLOGIA DELLE COLTURE (Tassonomia) ---
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
categoria(banana, frutto).
categoria(mango, frutto).
categoria(grapes, frutto).
categoria(watermelon, frutto).
categoria(muskmelon, frutto).
categoria(apple, frutto).
categoria(orange, frutto).
categoria(papaya, frutto).
categoria(coconut, frutto).
categoria(cotton, industriale).
categoria(jute, industriale).
categoria(coffee, industriale).

% --- 2. REGOLE DI DOMINIO (Background Knowledge) ---
% Regole esperte basate sulle proprietà chimico-fisiche del terreno.

% Regola: Un terreno è considerato 'acido' se il pH è inferiore a 5.5
terreno_acido(ID) :-
    dati_lotto(ID, _, _, _, PH, _, _),
    PH < 5.5.

% Regola: Un terreno è 'arido' se le precipitazioni sono inferiori a 400mm
terreno_arido(ID) :-
    dati_lotto(ID, _, _, _, _, Rain, _),
    Rain < 400.

% --- 3. REGOLE DI VALIDAZIONE (Constraint Checking) ---
% Il sistema logico controlla se la pianta suggerita dal ML è compatibile
% con i vincoli "hard" che il ML potrebbe aver trascurato.

% VINCOLO 1: Il Riso necessita di molta acqua.
% Se il ML suggerisce 'rice' ma il terreno è arido -> Validazione Fallita.
incompatibile(ID, rice) :-
    terreno_arido(ID).

% VINCOLO 2: I legumi soffrono i terreni fortemente acidi.
% Se il ML suggerisce una leguminosa su terreno acido -> Validazione Fallita.
incompatibile(ID, Pianta) :-
    categoria(Pianta, leguminosa),
    terreno_acido(ID).

% --- 4. INTERFACCIA DI VALUTAZIONE ---
% Predicato principale chiamato da Python.
% Restituisce 'true' solo se non ci sono incompatibilità logiche.
valida_raccomandazione(ID, Pianta) :-
    \+ incompatibile(ID, Pianta). % '\+' significa NOT in Prolog