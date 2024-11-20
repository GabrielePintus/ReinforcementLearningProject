Limit order book: tiene traccia delle richieste d'acquisto e di vendita di asset (azioni). Quindi un ordine specifica il prezzo e la direzione:
- Bid: compra
- Ask: vende
Un LOB ha fixato il numero di livelli, che indica quanto in profondità possono andare le richieste. Esistono due tipologie di ordini:

- Market Order: voglio comprare/vendere al prezzo disponibile nel LOB. Viene gestito da una queue. Hanno sempre precedenza i prezzi più vantaggiosi
- Limit order: specifico il prezzo a cui voglio vendere/comprare e aspetto fino a quando non trovo qualcuno che vuole vendere/comprare esattamente a quel prezzo. L'idea è che qui entri il market maker che ha comprato/venduto ad un prezzo vantaggioso e poi entra su questa richiesta guadagnandoci.

Le azioni che può svolgere l'agente possono variare rispetto al mid_price cambiando la varianza (0-4) o il bias (5-8)

Controllare se le azioni unbiased hanno senso. 

$Matched_{a/b}(t_i)$ è quanto ha venduto/comprato. Va messo uno dei due prezzi.
