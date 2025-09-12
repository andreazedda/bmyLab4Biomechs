# EKF Enhanced Controls Documentation

## Panoramica
Questo documento descrive le funzionalità di controllo e validazione avanzate aggiunte al sistema EKF per la stima di velocità e posizione da accelerazioni.

## Nuove Funzionalità

### 1. Sistema di Validazione Completo

#### 1.1 Validazione della Configurazione
- **Funzione**: `validate_configuration(config: Dict) -> bool`
- **Scopo**: Valida tutti i parametri di configurazione prima dell'esecuzione
- **Controlli**:
  - Presenza di parametri obbligatori
  - Validità dei range numerici
  - Coerenza tra parametri correlati
  - Verifica percorsi file e directory

#### 1.2 Validazione dei Risultati
- **Funzione**: `validate_results(results: Dict, config: Dict) -> Tuple[bool, List[str]]`
- **Scopo**: Valida i risultati EKF per rilevare anomalie
- **Controlli**:
  - **Bounds checking**: Verifica che velocità, posizione e accelerazione rimangano entro limiti fisici
  - **Physics validation**: Controllo coerenza fisica (derivate, continuità)
  - **Drift detection**: Rilevamento di derive non realistiche
  - **NaN/Inf detection**: Controllo valori numerici non validi

### 2. Sistema di Controllo Esecuzione

#### 2.1 Modalità di Esecuzione
```yaml
execution_control:
  mode: 'auto'  # Opzioni: 'auto', 'interactive', 'batch'
```

- **auto**: Esecuzione automatica senza interruzioni
- **interactive**: Richiede conferme dell'utente per operazioni critiche
- **batch**: Modalità batch con logging esteso

#### 2.2 Controllo Step-by-Step
```yaml
step_by_step:
  enabled: false
  pause_after_preprocessing: false
  pause_after_ekf: false
  pause_after_postprocessing: false
  pause_before_visualization: false
  pause_before_completion: false
```

**Punti di pausa disponibili**:
1. **Dopo preprocessing**: Pausa dopo caricamento e preparazione dati
2. **Dopo EKF**: Pausa dopo ogni esecuzione del filtro di Kalman
3. **Dopo post-processing**: Pausa dopo salvataggio risultati per ogni asse
4. **Prima visualizzazione**: Pausa prima di generare grafici
5. **Prima conclusione**: Pausa finale prima di terminare

### 3. Sistema di Backup e Sicurezza

#### 3.1 Backup Automatico
- **Funzione**: `create_backup_if_needed(file_path: str, config: Dict) -> bool`
- **Comportamento**:
  - Crea backup automatici con timestamp
  - Formato: `filename_backup_YYYYMMDD_HHMMSS.ext`
  - Configurabile tramite `backup_existing_files`

#### 3.2 Conferme Utente
- **Funzione**: `get_user_confirmation(message: str, config: Dict) -> bool`
- **Utilizzo**:
  - Sovrascrittura file esistenti
  - Operazioni potenzialmente pericolose
  - Progressione step-by-step

### 4. Sistema di Reporting

#### 4.1 Report di Validazione
- **Funzione**: `print_validation_report(results: Dict, validation_results: List[str])`
- **Output**: Report formattato con:
  - Statistiche risultati (min, max, media, std)
  - Elenco avvertimenti e errori
  - Raccomandazioni per ottimizzazioni

#### 4.2 Logging Avanzato
- Integrazione con sistema logging Python
- Livelli di log configurabili
- Output colorato per terminale

### 5. Scelte Algoritmiche

#### 5.1 Varianti EKF
```yaml
algorithm_choices:
  ekf_variant: 'standard'  # 'standard', 'extended', 'unscented'
```

#### 5.2 Correzione Drift
```yaml
drift_correction:
  method: 'none'  # 'none', 'linear_detrend', 'high_pass', 'zero_velocity'
```

#### 5.3 Ottimizzazione Parametri
```yaml
optimization:
  algorithm: 'minimize'  # 'minimize', 'differential_evolution', 'genetic'
```

#### 5.4 Smoothing
```yaml
smoothing:
  method: 'none'  # 'none', 'moving_average', 'savgol', 'gaussian'
```

## Configurazione di Esempio

### Configurazione per Debug Interattivo
```yaml
execution_control:
  mode: 'interactive'
  step_by_step:
    enabled: true
    pause_after_preprocessing: true
    pause_after_ekf: true
    pause_before_visualization: true
  user_confirmations:
    ask_before_overwrite: true
    ask_before_backup: true

output_validation:
  enabled: true
  velocity_bounds: [-5.0, 5.0]
  position_bounds: [-10.0, 10.0]
  physics_checks: true
  drift_detection: true
```

### Configurazione per Produzione
```yaml
execution_control:
  mode: 'auto'
  step_by_step:
    enabled: false
  user_confirmations:
    ask_before_overwrite: false
    ask_before_backup: false

output_validation:
  enabled: true
  velocity_bounds: [-10.0, 10.0]
  position_bounds: [-20.0, 20.0]
  physics_checks: true
  drift_detection: true
```

## Utilizzo

### 1. Abilitazione Controlli Base
```python
# Nel file YAML di configurazione
execution_control:
  mode: 'interactive'
output_validation:
  enabled: true
```

### 2. Debug Step-by-Step
```python
# Per debug dettagliato
step_by_step:
  enabled: true
  pause_after_preprocessing: true
  pause_after_ekf: true
```

### 3. Validazione Rigorosa
```python
# Per validazione completa
output_validation:
  enabled: true
  physics_checks: true
  drift_detection: true
  bounds_checking: true
```

## Gestione Errori

### Errori di Validazione
- **Configurazione non valida**: Il sistema si ferma con messaggio di errore dettagliato
- **Risultati anomali**: Vengono generate avvertenze ma l'esecuzione continua
- **Errori critici**: L'esecuzione si interrompe con opzione di ripristino

### Recovery e Backup
- Backup automatici prevengono perdita dati
- Possibilità di ripristino da configurazioni precedenti
- Log dettagliati per debugging

## Note di Performance

### Overhead delle Validazioni
- Validazione configurazione: ~1-5ms
- Validazione risultati: ~10-50ms per asse
- Backup file: dipende dalla dimensione del file

### Raccomandazioni
- Disabilitare validazioni non necessarie in produzione
- Usare modalità step-by-step solo per debug
- Configurare bounds appropriati per il proprio caso d'uso

## Troubleshooting

### Problemi Comuni

1. **Validazione fallisce sempre**
   - Verificare bounds nel file di configurazione
   - Controllare qualità dati di input
   - Disabilitare temporaneamente physics_checks

2. **Troppe pause step-by-step**
   - Disabilitare pause non necessarie
   - Usare mode: 'auto' per esecuzione automatica

3. **Backup non creati**
   - Verificare permessi di scrittura
   - Controllare spazio disco disponibile
   - Verificare configurazione backup_existing_files

## Futuro Sviluppo

### Funzionalità Pianificate
- [ ] Validazione in tempo reale durante esecuzione
- [ ] Export configurazioni ottimali
- [ ] Interfaccia grafica per controlli
- [ ] Profiling automatico performance
- [ ] Test automatici qualità risultati

### Estensioni Possibili
- [ ] Integrazione con database risultati
- [ ] API REST per controllo remoto
- [ ] Machine learning per ottimizzazione automatica
- [ ] Confronto automatico con versioni precedenti
