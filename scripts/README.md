# Skripte za pokretanje benchmark-a

Sve skripte se pokrecu iz root-a repozitorijuma.

## Report skripte (OMP / CUDA)

Dve glavne skripte generisu kompletan izvestaj (accuracy + timing + markdown):

| Skripta | Gde se pokrece | Izlaz |
|---------|----------------|-------|
| `./scripts/benchmark-omp.sh` | lokalno (Docker) | `results/benchmark-omp-report.md` |
| `./scripts/benchmark-cuda.sh` | remote masina sa `nvcc` | `results/benchmark-cuda-report.md` |

**Upotreba:** pokreni skriptu iz root-a repo-a. OMP ide kroz Docker; CUDA mora na masini sa NVIDIA GPU (prethodno sync repo-a, npr. `rsync`).

**Dodavanje nedostajucih referentnih resenja:** kad se pojavi nova referenca (npr. Moldyn CUDA, Feynman-Kac):

1. Stavi izvorni kod u `problems/<problem>/reference/{omp,cuda}/`.
2. Po potrebi dodaj `scripts/<problem>/run-*-reference.sh` po uzoru na postojece.
3. U odgovarajucoj `benchmark-*.sh` skripti: ukljuci `reference` u listu modela za taj problem, dodaj putanju do fajla u validate + benchmark funkcijama, i u generisanju reporta (tabela / labels).
4. Ponovo pokreni benchmark skriptu da se referenca pojavi u reportu.

Trenutno nedostaju npr. **Moldyn CUDA** i **Feynman-Kac** reference (OMP/CUDA); ostale reference su vec ukljucene gde postoje.

## Struktura

```
scripts/
├── benchmark-omp.sh        # Pun OMP benchmark + report
├── benchmark-cuda.sh       # Pun CUDA benchmark + report
├── sgemm/
│   ├── run-omp.sh          # OpenMP verzija (Docker + Parboil)
│   ├── run-seq.sh          # Sekvencijalna verzija (Docker + Parboil)
│   ├── run-cuda.sh         # CUDA verzija (remote masina)
│   └── run-seq-remote.sh   # Sekvencijalna verzija (remote, bez Dockera)
├── hotspot/
│   ├── run-omp.sh          # OpenMP verzija (Docker)
│   ├── run-seq.sh          # Sekvencijalna verzija (Docker)
│   ├── run-cuda.sh         # CUDA verzija (remote masina)
│   └── run-seq-remote.sh   # Sekvencijalna verzija (remote, bez Dockera)
├── mandelbrot/
│   ├── run-omp.sh          # OpenMP verzija (Docker)
│   ├── run-seq.sh          # Sekvencijalna verzija (Docker)
│   ├── run-cuda.sh         # CUDA verzija (remote masina)
│   └── run-seq-remote.sh   # Sekvencijalna verzija (remote, bez Dockera)
├── moldyn/
│   ├── run-omp.sh          # OpenMP verzija (Docker)
│   ├── run-seq.sh          # Sekvencijalna verzija (Docker)
│   ├── run-cuda.sh         # CUDA verzija (remote masina)
│   └── run-seq-remote.sh   # Sekvencijalna verzija (remote, bez Dockera)
└── feynman-kac/
    ├── run-omp.sh          # OpenMP verzija (Docker)
    ├── run-seq.sh          # Sekvencijalna verzija (Docker)
    ├── run-cuda.sh         # CUDA verzija (remote masina)
    └── run-seq-remote.sh   # Sekvencijalna verzija (remote, bez Dockera)
```

## Preduslovi

### Za OpenMP i sekvencijalne (Docker) skripte
- Docker Desktop instaliran i pokrenut
- Za SGEMM: `~/Desktop/parboil` direktorijum sa Parboil benchmark suite-om
- Za Hotspot: data fajlovi u `problems/hotspot/data/` (temp_512, power_512, temp_1024, power_1024)

### Za CUDA i remote skripte
- Pristup udaljenoj masini sa NVIDIA GPU i `nvcc`
- Repo iskopiran na tu masinu (npr. `rsync -avz ./ user@host:~/repo/`)
- `gcc`/`g++` dostupan na masini

## Upotreba

### OpenMP (lokalno, u Docker-u)

```bash
# SGEMM — model moze biti: codex, opus, gemini
./scripts/sgemm/run-omp.sh codex 10

# Hotspot — argumenti: model, niti, grid (512/1024), iteracije
./scripts/hotspot/run-omp.sh codex 4 512 10000

# Mandelbrot — argumenti: model, niti
./scripts/mandelbrot/run-omp.sh codex 4

# Moldyn — argumenti: model, cestice, niti
./scripts/moldyn/run-omp.sh codex 20 4

# Feynman-Kac — argumenti: model, niti
./scripts/feynman-kac/run-omp.sh codex 4
```

### Sekvencijalno (lokalno, u Docker-u)

```bash
./scripts/sgemm/run-seq.sh
./scripts/hotspot/run-seq.sh 512 10000
./scripts/mandelbrot/run-seq.sh
./scripts/moldyn/run-seq.sh 20
./scripts/feynman-kac/run-seq.sh
```

### CUDA (na udaljenoj masini)

```bash
# Kopiraj repo na masinu
rsync -avz ./ user@gpu-server:~/repo/

# SSH na masinu
ssh user@gpu-server
cd ~/repo

# Pokreni CUDA benchmark-e
./scripts/sgemm/run-cuda.sh codex
./scripts/hotspot/run-cuda.sh codex 512 10000
./scripts/mandelbrot/run-cuda.sh codex
./scripts/moldyn/run-cuda.sh codex 20
./scripts/feynman-kac/run-cuda.sh codex
```

### Sekvencijalno (na udaljenoj masini, bez Docker-a)

```bash
./scripts/sgemm/run-seq-remote.sh
./scripts/hotspot/run-seq-remote.sh 512 10000
./scripts/mandelbrot/run-seq-remote.sh
./scripts/moldyn/run-seq-remote.sh 20
./scripts/feynman-kac/run-seq-remote.sh
```

## Merenje vremena

Sve skripte koriste `time` komandu. Relevantna metrika je **real** (wall-clock time):

```
real    0m4.173s   <-- ovo je vreme izvrsavanja
user    0m4.150s
sys     0m0.020s
```

## Napomene

- SGEMM OMP/SEQ koriste Parboil framework i zahtevaju Python 2 (automatski se instalira u kontejneru)
- Svi ostali problemi se kompajliraju direktno sa `gcc`/`g++`
- Za Feynman-Kac CUDA: opus i gemini verzije koriste `curand` biblioteku (automatski se linkuje)
