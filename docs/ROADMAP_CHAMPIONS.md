# Champion roadmap (saved 2026-09-23 from Dani's message; do not re-fetch)

Top-lane pick list, all elo brackets, in order of pick rate. Columns as given:
rank, role, champion, tier, win rate, pick rate, ban rate, games.

| # | champion | tier | win | pick | ban | games |
|---|---|---|---|---|---|---|
| 1 | Yone | S | 50.05% | 10.4% | 16.5% | 35,747 |
| 2 | Darius | D | 48.85% | 7.8% | 12.9% | 26,595 |
| 3 | Garen | S+ | 51.43% | 7.8% | 5.3% | 26,551 |
| 4 | Nasus | S+ | 51.51% | 7.6% | 28.4% | 26,086 |
| 5 | Sett | S | 51.30% | 7.1% | 3.2% | 24,448 |
| 6 | Malphite | S+ | 51.58% | 7.1% | 15.0% | 24,389 |
| 7 | Teemo | S+ | 51.58% | 6.8% | 15.1% | 23,412 |
| 8 | Mordekaiser | S+ | 50.65% | 6.5% | 11.3% | 22,123 |
| 9 | Aatrox | S | 50.21% | 6.3% | 8.9% | 21,717 |
| 10 | Jax | S | 50.23% | 6.0% | 7.8% | 20,422 |
| 11 | Renekton | A | 50.27% | 5.1% | 3.6% | 17,329 |
| 12 | Jayce | D | 47.84% | 4.6% | 4.6% | 15,838 |
| 13 | Yasuo | D | 48.83% | 4.1% | 23.5% | 14,115 |
| 14 | Gangplank | S | 50.84% | 4.0% | 6.6% | 13,824 |
| 15 | Irelia | B | 49.67% | 4.0% | 11.4% | 13,814 |
| 16 | Tryndamere | S | 50.79% | 3.9% | 3.3% | 13,408 |
| 17 | Yorick | S+ | 50.76% | 3.8% | 8.6% | 13,174 |
| 18 | Volibear | A | 50.26% | 3.8% | 1.6% | 12,880 |
| 19 | Illaoi | S+ | 51.88% | 3.8% | 8.6% | 12,848 |
| 20 | Fiora | A | 50.16% | 3.5% | 2.8% | 12,064 |
| 21 | Cho'Gath | C | 49.08% | 3.4% | 4.7% | 11,743 |
| 22 | Dr. Mundo | S | 50.96% | 3.4% | 3.3% | 11,671 |
| 23 | K'Sante | D | 46.08% | 3.2% | 2.0% | 10,974 |
| 24 | Ambessa | B | 49.56% | 2.8% | 3.0% | 9,718 |
| 25 | Camille | B | 49.49% | 2.8% | 2.2% | 9,494 |

## The plan this list serves (Dani, 2026-09-23)

1. Prove the RL loop on Garen-vs-Garen, 10-minute trials, in the C# port (now).
2. Then change the sim to model MODERN League, not the C# port; verify it against
   memory reads from high-elo Garen `.rofl` replays. Tower logic etc. is checked
   once, for the modern sim only.
3. Those replays are also the imitation prior: a KL anchor that DIMINISHES over
   training so RL can leave the human playstyle after a point.
4. Port champions from the C# server's roster, starting with this top-20.
   One policy conditioned on champion identity (OpenAI Five's hero embedding),
   not one model per champion and probably not even ranged-vs-melee; KL only
   on a handful of champions with replays and let the model generalise.
5. Scale: full games, then 5v5.
6. Single autoregressive action space, not separate heads. Memory in the core.
   Everything here is open to revision after the architecture literature
   review (`docs/ARCH_LIT_REVIEW.md`).
