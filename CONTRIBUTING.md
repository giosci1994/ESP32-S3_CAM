# Contributing Guidelines

Grazie per voler contribuire! ✨

## Requisiti
- Docker installato e funzionante
- Python 3.11 (solo per sviluppare lato server, non obbligatorio per usare il container)
- Git

## Flusso di lavoro
1. **Fork** del repo e crea un branch:
   ```bash
   git checkout -b feat/descrizione-breve
   ```
2. Fai modifiche piccole e mirate. Mantieni i commit chiari.
3. Aggiorna la documentazione se serve (README/CHANGELOG).
4. Esegui test manuali veloci:
   - `docker compose up -d` nella cartella `server`
   - verifica `http://localhost:12345/` e `/status`
5. Apri una **Pull Request** con descrizione, log e screenshot (se UI).

## Stile dei commit
Usa uno di questi prefissi:
- `feat:` nuova funzionalità
- `fix:` bugfix
- `docs:` documentazione
- `chore:` manutenzione, refactor non funzionale
- `perf:` performance
- `test:` test

Esempi:
- `feat: pinch → publish eventi MQTT su soglia`
- `fix: gestione reconnect MQTT`

## Linee guida codice
- Niente segreti in chiaro. Usa `.env` / variabili d'ambiente.
- Mantieni i default **neutri** con placeholder.
- Evita dipendenze pesanti se non strettamente necessarie.
- Segui la struttura delle varianti esistenti.

## Segnalazione bug
Apri una issue usando il template "Bug report" e includi:
- passaggi per riprodurre
- log rilevanti (container)
- info sistema (OS/architettura, versione Docker)
- screenshot se utile

## Funzionalità/idee
Usa il template "Feature request". Spiega perché è utile e come immagini l’uso.

Buon hacking! 💚
