# Parametri Temporali Dettagliati - Documentazione Completa

## 📊 Panoramica

Il sistema di analisi biomeccanica ora calcola **50 parametri temporali dettagliati** organizzati in **7 categorie principali**. Questi parametri forniscono un'analisi completa delle caratteristiche temporali dell'esecuzione degli esercizi (es. squat).

---

## 🎯 Struttura dei Parametri

### A. Tempi per Singola Ripetizione (8 parametri × N ripetizioni)

Per ogni ripetizione vengono calcolati:

#### Tempi Assoluti (secondi)
1. **`eccentric_time`** - Tempo fase eccentrica (discesa)
2. **`concentric_time`** - Tempo fase concentrica (risalita)
3. **`bottom_hold_time`** - Tempo in buca (posizione bassa)
4. **`top_hold_time`** - Tempo in piedi (posizione alta)

#### Tempi Normalizzati (percentuali)
5. **`eccentric_pct`** - % tempo eccentrica sul totale ripetizione
6. **`concentric_pct`** - % tempo concentrica sul totale ripetizione
7. **`bottom_hold_pct`** - % tempo in buca sul totale ripetizione
8. **`top_hold_pct`** - % tempo in piedi sul totale ripetizione

**Esempio Output (Rep #2)**:
```json
{
  "rep_num": 2,
  "eccentric_time": 1.397,
  "concentric_time": 1.240,
  "bottom_hold_time": 0.0,
  "top_hold_time": 0.785,
  "eccentric_pct": 40.84,
  "concentric_pct": 36.25,
  "bottom_hold_pct": 0.0,
  "top_hold_pct": 22.95,
  "total_time": 3.420
}
```

---

### B. Parametri di Timing Globale (4 parametri)

9. **`mean_rep_duration`** - Durata media di una ripetizione (s)
10. **`mean_cycle_time`** - Tempo medio da inizio rep → inizio rep successiva (s)
11. **`execution_frequency_hz`** - Frequenza esecutiva (Hz) = 1/cycle_time
12. **`mean_density_ratio`** - Rapporto lavoro/totale = (ecc+conc)/durata_rep

**Esempio**:
```json
{
  "mean_rep_duration": 4.337,
  "mean_cycle_time": 4.811,
  "execution_frequency_hz": 0.208,
  "mean_density_ratio": 0.683
}
```

**Interpretazione**:
- Frequenza di 0.208 Hz = ~1 ripetizione ogni 4.8 secondi
- Density ratio 0.683 = 68.3% del tempo è lavoro attivo (ecc+conc), 31.7% è pausa

---

### C. Tempi Totali della Sessione (6 parametri)

13. **`total_eccentric_time`** - Somma di tutti i tempi eccentrici (s)
14. **`total_concentric_time`** - Somma di tutti i tempi concentrici (s)
15. **`total_bottom_hold_time`** - Somma di tutte le pause in buca (s)
16. **`total_top_hold_time`** - Somma di tutte le pause in piedi (s)
17. **`total_work_time`** - Tempo totale di lavoro = ecc + conc (s)
18. **`total_pause_time`** - Tempo totale di pausa = bottom + top hold (s)

**Esempio (9 ripetizioni)**:
```json
{
  "total_eccentric_time": 15.38,
  "total_concentric_time": 11.80,
  "total_bottom_hold_time": 0.39,
  "total_top_hold_time": 10.91,
  "total_work_time": 27.18,
  "total_pause_time": 11.31
}
```

---

### D. Varianze Temporali (16 parametri: 4 fasi × 4 metriche)

Per ogni fase (eccentric, concentric, bottom_hold, top_hold) vengono calcolati:

19-22. **Eccentric**: `var`, `std`, `cv`, `range`
23-26. **Concentric**: `var`, `std`, `cv`, `range`
27-30. **Bottom Hold**: `var`, `std`, `cv`, `range`
31-34. **Top Hold**: `var`, `std`, `cv`, `range`

**Metriche**:
- **`var`** - Varianza (s²)
- **`std`** - Deviazione standard (s)
- **`cv`** - Coefficiente di variazione (std/mean)
- **`range`** - Range (max - min) (s)

**Esempio (Eccentric)**:
```json
{
  "var": 0.178,
  "std": 0.422,
  "cv": 0.192,
  "range": 1.476
}
```

**Interpretazione**:
- CV = 19.2% indica variabilità moderata nei tempi eccentrici
- CV < 10% = molto consistente
- CV 10-20% = moderatamente consistente
- CV > 20% = alta variabilità

---

### E. Parametri di Regolarità e Continuità (6 parametri)

35. **`rep_duration_var`** - Varianza delle durate totali (s²)
36. **`rep_duration_cv`** - CV delle durate totali
37. **`rep_duration_range`** - Range delle durate totali (s)
38. **`cycle_time_var`** - Varianza del cycle time (s²)
39. **`cycle_time_cv`** - CV del cycle time
40. **`cycle_time_range`** - Range del cycle time (s)

**Esempio**:
```json
{
  "rep_duration_var": 2.145,
  "rep_duration_cv": 0.338,
  "rep_duration_range": 4.041,
  "cycle_time_var": 2.148,
  "cycle_time_cv": 0.305,
  "cycle_time_range": 3.976
}
```

**Interpretazione**:
- CV rep_duration 33.8% indica variabilità alta nelle durate
- Possibile indicatore di fatica o mancanza di ritmo costante

---

### F. Outliers Temporali (4 parametri)

41. **`num_slow_eccentric`** - Ripetizioni con eccentrica > 2× mediana
42. **`num_fast_concentric`** - Ripetizioni con concentrica < 0.5× mediana
43. **`num_excessive_bottom_hold`** - Pause in buca > 2.0s
44. **`num_excessive_top_hold`** - Pause in piedi > 2.0s

**Esempio**:
```json
{
  "num_slow_eccentric": 0,
  "num_fast_concentric": 0,
  "num_excessive_bottom_hold": 0,
  "num_excessive_top_hold": 1
}
```

**Interpretazione**:
- 1 ripetizione con pausa eccessiva in piedi (>2s)
- Possibile indicatore di riposo tra ripetizioni

---

### G. Dinamiche Temporali Longitudinali (6 parametri)

Trend lineari che mostrano come cambiano i tempi nel corso della sessione (slope in s/rep):

45. **`eccentric_trend`** - Trend tempi eccentrici (s/rep)
46. **`concentric_trend`** - Trend tempi concentrici (s/rep)
47. **`bottom_hold_trend`** - Trend pause in buca (s/rep)
48. **`top_hold_trend`** - Trend pause in piedi (s/rep)
49. **`rep_duration_trend`** - Trend durata totale (s/rep)
50. **`cycle_time_trend`** - Trend cycle time (s/rep)

**Esempio**:
```json
{
  "eccentric_trend": +0.248,
  "concentric_trend": +0.155,
  "bottom_hold_trend": +0.026,
  "top_hold_trend": -0.310,
  "rep_duration_trend": +0.130,
  "cycle_time_trend": +0.170
}
```

**Interpretazione**:
- **Eccentric trend +0.248 s/rep**: Rallentamento progressivo della discesa (+0.248s ogni ripetizione)
- **Concentric trend +0.155 s/rep**: Rallentamento progressivo della risalita
- **Top hold trend -0.310 s/rep**: Riduzione progressiva delle pause in piedi (meno riposo)
- **Rep duration trend +0.130 s/rep**: Aumento progressivo della durata totale

**Indicatori di Fatica**:
- ✅ Trend positivi in ecc/conc = rallentamento (fatica muscolare)
- ⚠️ Trend negativo in top_hold = meno riposo (possibile compensazione)
- 🔍 Combinazione suggerisce accumulo di fatica con riduzione volontaria dei riposi

---

## 📊 Output Disponibili

### 1. Terminal Output
Visualizzazione formattata durante l'esecuzione dello script:

```
⏱️  DETAILED TEMPORAL PARAMETERS:
┌───────────────────────────────────────────────┐
│ GLOBAL TIMING                                 │
├───────────────────────────────────────────────┤
│   Mean Rep Duration:          4.34 s          │
│   Mean Cycle Time:            4.81 s          │
│   Execution Frequency:       0.208 Hz         │
│   Mean Density Ratio:        68.28%          │
...
```

### 2. JSON Export
File: `FILE_EULER*_report_repetitions.json`

```json
{
  "ang3_filtered": {
    "num_repetitions": 9,
    "temporal_parameters": {
      "per_repetition": [...],
      "global_timing": {...},
      "session_totals": {...},
      "temporal_variances": {...},
      "regularity": {...},
      "outliers": {...},
      "longitudinal_trends": {...}
    }
  }
}
```

### 3. HTML Report
Sezione dedicata con visualizzazione interattiva:

- **Tabelle** per parametri globali, totali sessione, varianze
- **Grafici colore-coded** per trend (verde = positivo, rosso = negativo)
- **Tabella dettagliata** per-ripetizione con tutte le fasi

---

## 🎯 Casi d'Uso

### 1. Valutazione della Tecnica
- **Density ratio alto** (>70%) = esecuzione continua, pochi riposi
- **CV tempi bassi** (<15%) = tecnica molto consistente
- **Outliers minimi** = esecuzione controllata

### 2. Monitoraggio della Fatica
- **Trend eccentric positivo** = rallentamento progressivo (fatica)
- **Trend concentric positivo** = perdita di esplosività
- **Aumento pause in piedi** = necessità crescente di riposo

### 3. Confronto tra Sessioni
- Confrontare `session_totals` per volume totale di lavoro
- Confrontare `temporal_variances` per valutare consistenza
- Confrontare `longitudinal_trends` per strategie di pacing

### 4. Ottimizzazione del Training
- **Alto density ratio + trend stabili** = ottimale per ipertrofia
- **Bassi tempi eccentrici** = focus su forza esplosiva
- **Alti tempi eccentrici** = focus su controllo/ipertrofia

---

## 🔬 Aspetti Tecnici

### Calcolo delle Fasi
1. **Mappatura fasi biomeccaniche**:
   - `ascending` → `concentric` (risalita)
   - `descending` → `eccentric` (discesa)
   - `stable_low` → `bottom_hold` (buca)
   - `stable_high` → `top_hold` (in piedi)

2. **Finestre di ripetizione**: Definite tra i minimi (trough) consecutivi

3. **Overlap detection**: Le fasi che intersecano la finestra vengono incluse proporzionalmente

### Soglie Outliers
- **Slow eccentric**: > 2.0× tempo mediano
- **Fast concentric**: < 0.5× tempo mediano
- **Excessive bottom hold**: > 2.0 secondi
- **Excessive top hold**: > 2.0 secondi

### Trend Calculation
Regressione lineare (`scipy.stats.linregress`) su:
- X = indice ripetizione (0, 1, 2, ...)
- Y = tempo della fase
- Output = slope (s/rep)

---

## 📚 Riferimenti nel Codice

### Funzione Principale
`compute_detailed_temporal_parameters(rep_data, phases, ts, theta)`
- **Posizione**: Line ~430 in `find_time_intervals.py`
- **Input**: Dati ripetizioni, fasi, timestamp, segnale
- **Output**: Dizionario con 7 categorie di parametri

### Integrazione
- **Calcolo**: Line ~1468 (chiamata dopo `analyze_repetition_quality`)
- **Salvataggio JSON**: Incluso in `repetition_stats[angle_col]`
- **Visualizzazione Terminal**: Line ~1694-1755
- **Visualizzazione HTML**: Line ~3105-3240

---

## 🎓 Glossario

| Termine | Definizione |
|---------|-------------|
| **Eccentric** | Fase di allungamento muscolare (discesa nello squat) |
| **Concentric** | Fase di accorciamento muscolare (risalita nello squat) |
| **Bottom Hold** | Pausa nella posizione più bassa |
| **Top Hold** | Pausa nella posizione più alta (in piedi) |
| **Cycle Time** | Tempo da inizio ripetizione a inizio ripetizione successiva |
| **Density Ratio** | Rapporto tra tempo di lavoro attivo e tempo totale |
| **CV (Coefficient of Variation)** | std/mean - misura di variabilità relativa |
| **Trend** | Slope della regressione lineare - cambio per ripetizione |

---

## ✅ Checklist Parametri

- [x] **A1-A8**: Tempi per ripetizione (assoluti + normalizzati)
- [x] **B9-B12**: Parametri timing globale
- [x] **C13-C18**: Tempi totali sessione
- [x] **D19-D34**: Varianze temporali (4 fasi × 4 metriche)
- [x] **E35-E40**: Regolarità e continuità
- [x] **F41-F44**: Outliers temporali
- [x] **G45-G50**: Trend longitudinali

**Totale: 50 parametri temporali implementati! ✨**

---

## 📞 Supporto

Per domande o suggerimenti sui parametri temporali:
1. Consultare il codice sorgente: `find_time_intervals.py`
2. Verificare output JSON: `FILE_EULER*_report_repetitions.json`
3. Aprire HTML report per visualizzazione interattiva
4. Controllare terminal output per riepiloghi testuali

---

**Ultimo aggiornamento**: 17 Novembre 2025
**Versione**: 1.0.0
**Autore**: AI Assistant per bmyLab4Biomechs
