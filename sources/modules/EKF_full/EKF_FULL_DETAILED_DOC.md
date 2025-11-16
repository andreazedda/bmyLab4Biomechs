# EKF_full – Documentazione dettagliata

## 1. Obiettivi e campo di applicazione
EKF_full è il modulo più completo del toolkit **bmyLab4Biomechs** per ricostruire la cinematica del torace durante esercizi di squat utilizzando esclusivamente i sensori IMU dello smartphone: accelerometro, giroscopio (con bias), magnetometro, quaternioni relativi ed Euler angles già calcolati dal sistema operativo Android.  
Il modulo fornisce:

- un **EKF a 15 stati** per la stima di posizione, velocità, orientamento e bias;
- un **Complementary Filter** orientato ai pattern biomeccanici (approccio consigliato);
- strumenti di confronto, plotting avanzato, analisi per ripetizione e reportistica sui test.

> ⚠️ **Limite fisico**: senza un riferimento assoluto (UWB, visione, barometro) la posizione integrata da accelerazioni è soggetta a deriva. Il modulo privilegia quindi stime *relative* e metriche per ripetizione, concentrandosi su orientamento e profondità dello squat.

---

## 2. Architettura del modulo

```
EKF_full/
├── sources/
│   ├── config.py                  # Dataclass di configurazione pipeline
│   ├── data_loading.py            # Parsing & allineamento temporale sensori
│   ├── ekf_model.py               # Dinamica EKF 15D + aggiornamenti
│   ├── complementary_filter.py    # Filtro complementare biomeccanico
│   ├── run_ekf.py                 # Pipeline legacy EKF con plot opzionali
│   ├── compare_filters.py         # Confronto grafico EKF vs Complementary
│   ├── plot_squat_analysis.py     # Visualizzazione multi-pannello squat
│   ├── relative_tracking.py       # Reset per ripetizione e metriche
│   ├── plotting.py                # Helper per grafici standard
│   └── math_utils.py              # Utility matematiche (rotazioni, jacobiani)
├── data/
│   ├── inputs/txts/               # File grezzi Android (timestamp + sensori)
│   └── outputs/                   # CSV elaborati e figure
├── TESTING.md                     # Stato validazione e problemi aperti
├── SENSOR_FUSION_REPORT.md        # Report tecnico sul sensore fusion
└── (nuovo) EKF_FULL_DETAILED_DOC.md
```

### Ruoli principali dei file

| File | Ruolo |
| --- | --- |
| `config.py` | Definisce `PipelineConfig`, `NoiseParams`, `DetectionParams` e la funzione `ensure_output_dir()` utilizzata da tutti gli script. |
| `data_loading.py` | Gestisce la discovery delle sessioni, il caricamento dei TXT e l'allineamento `merge_asof` con interpolazione limitata (senza interpolare gli Euler angles). |
| `ekf_model.py` | Implementa la classe `InertialPositionEKF`, la dinamica non lineare, i jacobiani numerici, la logica ZUPT e i limiti sulla covarianza. |
| `complementary_filter.py` | Realizza un filtro complementare con modello biomeccanico, rilevazione delle fasi e stima stabile di posizione relativa. |
| `relative_tracking.py` | Estrae ripetizioni successive a partire da fasi stazionarie e resetta l’origine per contenere la deriva. |
| `plot_squat_analysis.py` / `plotting.py` | Generano grafici standard e layout avanzati per una sessione. |
| `compare_filters.py` | Produce un confronto visivo EKF vs Complementary con statistiche di range per ogni asse. |

---

## 3. Pipeline dati end-to-end

1. **Acquisizione** (`data/inputs/txts/FILE_*<sessione>.txt`): tutti i sensori sono salvati con timestamp in millisecondi.
2. **Selezione sessione** (`list_sessions()`): individua gli ID disponibili analizzando i prefissi dei file ACC.
3. **Caricamento** (`load_session()`): costruisce `SessionData` con DataFrame individuali per acc, gyro, magn, Euler, quaternioni relativi.
4. **Allineamento temporale** (`build_time_aligned_frame()`):
   - usa l’accelerometro come base e aggiunge gli altri sensori via `merge_asof`;
   - applica interpolazione numerica limitata (esclusi gli Euler angles) e calcola `time_s`.
5. **Stima orientamento iniziale**: dalla prima riga delle colonne `euler_*`.
6. **Esecuzione filtro (EKF o Complementary)**:
   - EKF integra dinamica + aggiornamenti multipli (Euler, magnetometro, ZUPT);
   - il Complementary Filter lavora nel dominio della frequenza basandosi su orientamento + modello biomeccanico.
7. **Post-processing**:
   - salvataggio CSV in `data/outputs/<sessione>_*_results.csv`;
   - generazione di grafici (`plotting.py`, `plot_squat_analysis.py`);
   - confronto o analisi per ripetizione (`compare_filters.py`, `relative_tracking.py`).

---

## 4. Configurazione del pipeline (`sources/modules/EKF_full/sources/config.py`)

### 4.1 `PipelineConfig`
```python
@dataclass(frozen=True)
class PipelineConfig:
    sensor_sampling_hz: float = 100.0
    noise: NoiseParams = NoiseParams()
    detection: DetectionParams = DetectionParams()
```

### 4.2 `NoiseParams`

| Campo | Default | Effetto |
| --- | --- | --- |
| `pos` | `5e-3` | Rumore di processo sulla posizione: valori più alti disaccoppiano la posizione dalla dinamica integrata, utile per attenuare la deriva durante clamp o reset. |
| `vel` | `5e-2` | Rumore di processo sulla velocità: incrementarlo permette al filtro di assorbire meglio bias residui. |
| `angles` | `1e-3` | Rumore sulle componenti roll/pitch/yaw. Valori bassi indicano alta fiducia nel modello; qui restano bassi perché gli aggiornamenti Euler sono molto precisi. |
| `accel_bias` | `5e-4` | Modella la deriva lenta del bias accelerometrico (random walk). |
| `gyro_bias` | `1e-6` | Idem per il bias del giroscopio. |
| `euler_meas_deg` | `0.5` | Rumore di misura (°) per gli Euler angles provenienti da Android; 0.5° indica altissima fiducia. |
| `mag_meas_uT` | `30.0` | Rumore sul magnetometro, volutamente alto per limitare gli aggiornamenti in ambienti indoor. |
| `zero_vel` | `1e-3` | Varianza utilizzata durante le ZUPT: più piccolo = vincolo di velocità quasi rigido. |

### 4.3 `DetectionParams`

| Campo | Default | Ruolo |
| --- | --- | --- |
| `gravity` | `9.80665` | Modulo del vettore gravità utilizzato per normalizzare le misure accelerometriche. |
| `zero_vel_acc_window` | `0.3` | Tolleranza in m/s² tra norma accelerometro e gravità per considerare la posa stazionaria. |
| `zero_vel_gyro_window` | `0.08` | Soglia rad/s sul giroscopio per rilevare quiete. |
| `zero_vel_mag_window` | `0.5` | Soglia opzionale sul magnetometro. |
| `max_interp_gap_ms` | `15` | Massimo gap temporale per l’interpolazione durante l’allineamento. |
| `max_position_m` | `2.0` | Limite di sicurezza sulla distanza dal riferimento; oltre questo valore l’EKF applica clamp. |
| `position_reset_interval_s` | `2.0` | Frequenza con cui verificare la necessità di resettare/ammorbidire la posizione per evitare deriva. |

### 4.4 Consigli di tuning
- **Squat lenti**: ridurre `zero_vel_acc_window` a 0.2 per ZUPT più frequenti.
- **Sessioni rumorose**: aumentare `vel` e `pos` di un ordine di grandezza per permettere correzioni più aggressive.
- **Magnetometro inutilizzabile**: settare `mag_meas_uT` a valori >100 per disattivare di fatto l’aggiornamento.

---

## 5. Teoria ed implementazione dell’EKF

### 5.1 Stato e dinamica (`ekf_model.py`)
- Stato a 15 componenti: `[p(3), v(3), euler(3), bias_acc(3), bias_gyro(3)]`.
- Aggiornamento predittivo:
  - conversione velocità angolari → derivate Euler tramite matrice `J`;
  - integrazione e wrap degli angoli;
  - trasformazione accelerazioni nel world frame (`euler_to_rotation_matrix`) e sottrazione della gravità;
  - integrazione numerica di velocità e posizione (semi-implicit integrator).
- Jacobiano calcolato numericamente (`numerical_jacobian`) per mantenere la generalità del modello.
- Covarianza limitata (`_limit_covariance`) per evitare divergenze numeriche dopo update rumorosi.

### 5.1.1 Equazioni dinamiche
Il modello continuo è assimilabile a un classico strapdown INS:
```
dot(p) = v
dot(v) = R(roll, pitch, yaw) * (a_meas - b_a - n_a) - g
dot(theta) = T(theta) * (w_meas - b_g - n_g)
dot(b_a) = n_ba
dot(b_g) = n_bg
```
dove:
- `R` è la matrice di rotazione ZYX;
- `T(theta)` è la matrice di trasformazione tra velocità angolari e derivate Euler;
- `n_*` rappresentano rumor bianco gaussiano.

La discretizzazione con passo `dt` usata nella pipeline è di tipo Euler esplicito:
```
x_k = f(x_{k-1}, u_k) = x_{k-1} + dot(x_{k-1}) * dt + O(dt^2)
```
e la covarianza si propaga secondo
```
P_k^- = F_k P_{k-1} F_k^T + Q_k
```
con `F_k = ∂f/∂x` calcolata via jacobiano numerico e `Q_k` costruita da `NoiseParams`.

### 5.2 Funzioni di misura
| Sensore | Funzione | Note |
| --- | --- | --- |
| Euler angles (`_euler_measurement_fn`) | Misura diretta dello stato `IDX_ANG`. Usata con residuo avvolto (`_angle_residual`). |
| Magnetometro (`_mag_measurement_fn`) | Confronta il campo magnetico previsto (body frame) con quello misurato. Utilizzato solo se il modulo stimato supera 20 µT. |
| ZUPT (`zero_velocity_update`) | Impone la misura `v = 0` lungo i tre assi durante le fasi stazionarie. |

### 5.3 Strategie di controllo deriva
1. **Blend con gli Euler di Android**: ogni 5 campioni, metà dell’errore angolare viene iniettato direttamente nello stato per mantenere l’orientamento fedele (`run_ekf.py`).
2. **Position bounding**: se la distanza dal riferimento supera `max_position_m`, viene applicata una correzione rigida o morbida (70% hard clamp, 20% soft clamp).
3. **Reference update**: i periodi stazionari aggiornano la `reference_position`, usata come ancoraggio locale.

> Nonostante questi accorgimenti, l’EKF puro resta inadatto per posizione assoluta prolungata (vedi `TESTING.md`).

### 5.4 Dettagli della fase di aggiornamento
Per ogni misura valida `z_k = h(x_k) + v_k` con jacobiano `H_k = ∂h/∂x`, l’algoritmo utilizza la forma di Joseph per mantenere la semidefinitività:
```
S_k = H_k P_k^- H_k^T + R_k
K_k = P_k^- H_k^T S_k^{-1}
P_k = (I - K_k H_k) P_k^- (I - K_k H_k)^T + K_k R_k K_k^T
```
Il residuo angolare sfrutta `wrap_angles` per evitare salti di ±2π. In caso di matrice di innovazione singolare, viene aggiunta una regolarizzazione `εI` con `ε ≈ 1e-9`.

### 5.5 Logica di rilevazione stazionarietà (ZUPT)
Lo pseudocodice usato in `_should_apply_zupt` è:
```
acc_norm  = ||a_body||
gyro_norm = ||ω_calibrated||
mag_norm  = ||m||
acc_stable  = |acc_norm - g| < zero_vel_acc_window
gyro_stable = gyro_norm < zero_vel_gyro_window
mag_stable  = (mag dati) ? (mag_norm < zero_vel_mag_window) : True
return acc_stable and gyro_stable and mag_stable
```
Quando `True`, viene applicato un aggiornamento di misura con `z = [0,0,0]^T`, `h(x)=v`, `R=diag(zero_vel)` imponendo velocità nulla e riducendo rapidamente l’incertezza.

---

## 6. Complementary Filter e modello biomeccanico

Il file `complementary_filter.py` introduce un approccio ibrido pensato per squat ciclici:

1. **Separazione in frequenza**: un filtro Butterworth di ordine 4 suddivide accelerazioni (componente ad alta frequenza) e posizione biomeccanica stimata (bassa frequenza) usando un cut-off predefinito (`cutoff_hz = 0.5`).
2. **Orientamento affidabile**: gli Euler angles vengono imputati forward/backward fill per rimuovere i buchi; la rotazione world→body viene ricavata con `euler_to_rotation_matrix`.
3. **Rilevazione fasi** (`detect_squat_phases`):
   - smoothing Savitzky-Golay su pitch;
   - gradienti di pitch e velocità per classificare i campioni in `standing`, `descending`, `bottom`, `ascending`.
4. **Modello biomeccanico** (`biomechanical_position_model`):
   - profondità Z proporzionale all’angolo di pitch (fino a 0.5 m);
   - spostamento Y derivato da `TRUNK_LENGTH * sin(pitch)`;
   - spostamento X limitato (10% del trunk length) legato al roll.
5. **Fusione complementare**:
   ```python
   alpha = dt / (dt + 1/(2π*cutoff))
   pos = alpha * pos_high_freq + (1 - alpha) * biomech_position
   ```
6. **Statistica finale**: range realistici (X ±14 cm, Y ±37 cm, Z ≈34 cm) e percentuale fasi.

Questo approccio è la **pipeline raccomandata** per ottenere misure stabili della profondità dello squat e dell’oscillazione del torace.

### 6.1 Derivazione del filtro complementare
L’idea è fondere un integratore ad alta frequenza (`G_h(s)`), derivato dall’accelerometro corretto dalla gravità, con un modello biomeccanico `G_l(s)` affidabile solo alle basse frequenze:
```
H(s) = α(s) * G_h(s) + (1 - α(s)) * G_l(s)
```
con `α(s) = s / (s + ω_c)` e `ω_c = 2π * cutoff_hz`. Nel dominio discreto si ottiene il coefficiente
```
α_d = dt / (dt + 1/ω_c)
```
usato nella classe `ComplementaryPositionFilter`. Questo rende la risposta unitaria (somma dei contributi = 1) e mantiene stabili le componenti a bassa frequenza.

### 6.2 Modello biomeccanico parametrico
Il modello implementa la mappa:
```
z = -MAX_SQUAT_DEPTH * |pitch| / pitch_max
y = TRUNK_LENGTH * sin(pitch)
x = 0.1 * TRUNK_LENGTH * sin(roll)
```
con `pitch_max ≈ 40°`. Questi valori possono essere adattati per soggetti con diversa antropometria (es. modificando `MAX_SQUAT_DEPTH` o `TRUNK_LENGTH`).

### 6.3 Rilevazione di fase e segmentazione
`detect_squat_phases` applica:
1. smooth Savitzky-Golay (`window_length=21`, `polyorder=3`);
2. derivata numerica `dpitch/dt`;
3. classificazione campione per campione:
   - `standing`: velocità < 0.05 m/s e |dpitch| < 5°/s;
   - `descending`: `dpitch > threshold`;
   - `ascending`: `dpitch < -threshold`;
   - `bottom`: regione residua compresa tra i due rami.

Questa segmentazione alimenta sia il modello biomeccanico (differenziando fasi attive da stazionarie) sia `relative_tracking`.

---

## 7. Strumenti di analisi aggiuntivi

| Script | Funzione |
| --- | --- |
| `run_ekf.py` | Fornisce CLI per processare sessioni con l’EKF legacy (`--session`, `--all`, `--plot`). |
| `compare_filters.py` | Genera figure e statistiche confrontando EKF e Complementary; evidenzia i fattori di miglioramento (da 700× a 4700×). |
| `plot_squat_analysis.py` | Produce un layout 4×1 con posizione, profondità, orientamento e timeline delle fasi. |
| `relative_tracking.py` | Rileva fasi stazionarie tramite soglia sulla velocità, resetta la posizione e calcola metriche per ripetizione (durata, discesa/ascesa, picchi di velocità). |
| `TESTING.md` | Riassume bug risolti/pending e dà linee guida per ulteriori vincoli biomeccanici. |
| `SENSOR_FUSION_REPORT.md` | Documenta i risultati ottenuti, limiti residui e raccomandazioni future. |

---

## 8. Utilizzo pratico

### 8.1 Preparazione
```bash
cd /Volumes/nvme/Github/bmyLab4Biomechs/sources/modules/EKF_full
python -m venv venv && source venv/bin/activate  # (se non già fatto)
pip install -r ../../requirements.txt             # includere pandas, numpy, matplotlib, scipy
```

### 8.2 Esecuzioni comuni
```bash
# Elencare sessioni disponibili
python -m sources.run_ekf --list-sessions

# Complementary filter sulla sessione più recente
python sources/complementary_filter.py

# EKF + plot per sessione specifica
python -m sources.run_ekf --session 2025-10-28-10-30-39 --plot

# Analisi squat e confronto
python sources/plot_squat_analysis.py
python sources/compare_filters.py
```

Output principali in `data/outputs/`:
- `<session>_complementary_results.csv`
- `<session>_complementary_positions.png`, `..._velocities.png`, `..._trajectory3d.png`
- `<session>_complementary_squat_analysis.png`
- `<session>_comparison_ekf_vs_complementary.png`
- `<session>_ekf_results.csv` (solo pipeline legacy)

---

## 9. Come leggere i risultati

### 9.1 CSV complementare
Colonne principali:
`timestamp_ms`, `time_s`, `pos_[xyz]_cm`, `vel_[xyz]_cm_s`, `roll_deg`, `pitch_deg`, `yaw_deg`, `phase`.

### 9.2 Range attesi (telefono sul torace, squat standard)
| Metrica | Range realistico | Interpretazione |
| --- | --- | --- |
| `pos_z_cm` | -60 cm … 0 | Profondità squat (segno negativo = verso il basso). |
| `pos_y_cm` | ±35 cm | Traslazione antero-posteriore durante l’inclinazione del busto. |
| `pos_x_cm` | ±15 cm | Sway laterale; valori maggiori indicano instabilità. |
| `phase` | 0–3 | Controllare proporzioni per verificare tempi sotto/oltre carico. |
| `vel_z_cm_s` | Picchi 100–200 cm/s | Velocità di discesa/risalita. |

Se i range differiscono di ordini di grandezza, verificare: corretto orientamento del telefono, calibratura accelerometri, presenza di saturazione o errori nel modello biomeccanico.

---

## 10. Testing, validazione e limiti

- `TESTING.md` documenta bug risolti (matrice di trasformazione Euler, gestione NaN, stabilità numerica) e problemi ancora aperti sulla deriva delle posizioni.
- I test mostrano che, con l’EKF puro, l’escursione può raggiungere centinaia di metri in 30 s (`sources/modules/EKF_full/TESTING.md`). È quindi fondamentale usare:
  1. Complementary filter per analisi operative.
  2. `relative_tracking.apply_relative_tracking()` per misurazioni a ripetizione.
  3. Resets basati su fasi stazionarie per qualsiasi metrica cumulativa.
- Raccomandazioni future (dal report sensori): integrare barometro, vincolare maggiormente l’asse Z o sfruttare marker esterni per ottenere riferimento assoluto.

---

## 11. Best practice operative

1. **Assicurare la qualità dei dati**: eliminare sessioni con buchi >15 ms o Euler non disponibili.
2. **Fidarsi dell’orientamento del telefono**: l’errore medio 0.5° permette di usare direttamente roll/pitch/yaw per l’analisi biomeccanica senza ulteriori filtri.
3. **Usare sempre il Complementary Filter per insight clinici**; tenere l’EKF solo come baseline o per esperimenti di sensor fusion avanzata.
4. **Segmentare per ripetizione**: resettare la posizione ad ogni fase di stazionamento evita accumulo errori e fornisce metriche direttamente confrontabili (durata, profondità, simmetria).
5. **Documentare i parametri**: ogni modifica a `NoiseParams` o `DetectionParams` va annotata per riproducibilità (consigliato mantenere copie di `config.py` con tag data/attività).

Con questa documentazione il modulo EKF_full è pronto per essere usato come base per analisi biomeccaniche affidabili in assenza di sistemi di motion capture, privilegiando metriche relative, fasi temporali e orientamenti rispetto alla posizione assoluta.
