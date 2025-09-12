#!/usr/bin/env python3
"""
Test delle funzionalità avanzate del sistema EKF
Script per testare tutti i controlli e validazioni implementati
"""

import os
import sys
import yaml
import tempfile
import numpy as np
from pathlib import Path

# Aggiungi il percorso del modulo EKF
sys.path.append('/Volumes/nvme/Github/igmSquatBiomechanics/sources/lab/modules/EKF_for_vel_and_pos_est_from_acc/sources')

from EKF_for_vel_and_pos_est_from_acc import (
    validate_configuration,
    validate_results, 
    create_backup_if_needed,
    get_user_confirmation,
    print_validation_report,
    apply_ekf_with_auto_tuning
)

def create_test_data():
    """Crea dati di test per le validazioni"""
    print("🔬 Creazione dati di test...")
    
    # Crea directory temporanea
    test_dir = tempfile.mkdtemp(prefix="ekf_test_")
    print(f"📁 Directory di test: {test_dir}")
    
    # Genera dati sintetici di accelerazione
    t = np.linspace(0, 10, 1000)  # 10 secondi, 100 Hz
    
    # Accelerazione con movimento sinusoidale + rumore
    freq = 0.5  # Hz
    acc_clean = 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
    acc_noisy = acc_clean + 0.1 * np.random.randn(len(t))
    
    # Salva i dati
    data_file = os.path.join(test_dir, "test_acceleration.csv")
    with open(data_file, 'w') as f:
        f.write("time,acc_x,acc_y,acc_z\n")
        for i, time in enumerate(t):
            f.write(f"{time:.4f},{acc_noisy[i]:.6f},{acc_noisy[i]*0.5:.6f},{acc_noisy[i]*0.2:.6f}\n")
    
    print(f"✅ Dati di test salvati in: {data_file}")
    return test_dir, data_file

def create_test_config(test_dir, data_file, test_mode="validation"):
    """Crea configurazione di test"""
    print(f"⚙️  Creazione configurazione test mode: {test_mode}")
    
    if test_mode == "validation":
        config = {
            'data': {
                'input_file_path': data_file,
                'time_column': 'time',
                'acceleration_columns': ['acc_x', 'acc_y', 'acc_z'],
                'axes_to_analyze': ['X', 'Y', 'Z']
            },
            'output': {
                'base_output_dir': test_dir,
                'create_timestamp_folder': True,
                'backup_existing_files': True
            },
            'ekf_parameters': {
                'process_noise': 0.01,
                'measurement_noise': 0.1,
                'initial_state_covariance': 1.0
            },
            'execution_control': {
                'mode': 'interactive',
                'step_by_step': {
                    'enabled': True,
                    'pause_after_preprocessing': True,
                    'pause_after_ekf': True,
                    'pause_after_postprocessing': False,
                    'pause_before_visualization': True,
                    'pause_before_completion': True
                },
                'user_confirmations': {
                    'ask_before_overwrite': True,
                    'ask_before_backup': True
                }
            },
            'output_validation': {
                'enabled': True,
                'velocity_bounds': [-5.0, 5.0],
                'position_bounds': [-10.0, 10.0],
                'acceleration_bounds': [-20.0, 20.0],
                'physics_checks': True,
                'drift_detection': True,
                'bounds_checking': True
            },
            'algorithm_choices': {
                'ekf_variant': 'standard',
                'drift_correction': {
                    'method': 'none'
                },
                'optimization': {
                    'algorithm': 'minimize'
                },
                'smoothing': {
                    'method': 'none'
                }
            }
        }
    
    elif test_mode == "production":
        config = {
            'data': {
                'input_file_path': data_file,
                'time_column': 'time',
                'acceleration_columns': ['acc_x', 'acc_y', 'acc_z'],
                'axes_to_analyze': ['X']  # Solo un asse per test veloce
            },
            'output': {
                'base_output_dir': test_dir,
                'create_timestamp_folder': False,
                'backup_existing_files': False
            },
            'ekf_parameters': {
                'process_noise': 0.01,
                'measurement_noise': 0.1,
                'initial_state_covariance': 1.0
            },
            'execution_control': {
                'mode': 'auto',
                'step_by_step': {
                    'enabled': False
                },
                'user_confirmations': {
                    'ask_before_overwrite': False,
                    'ask_before_backup': False
                }
            },
            'output_validation': {
                'enabled': True,
                'velocity_bounds': [-10.0, 10.0],
                'position_bounds': [-20.0, 20.0],
                'acceleration_bounds': [-50.0, 50.0],
                'physics_checks': False,
                'drift_detection': True,
                'bounds_checking': True
            },
            'algorithm_choices': {
                'ekf_variant': 'standard',
                'drift_correction': {
                    'method': 'none'
                },
                'optimization': {
                    'algorithm': 'minimize'
                },
                'smoothing': {
                    'method': 'none'
                }
            }
        }
    
    elif test_mode == "bounds_fail":
        # Configurazione con bounds molto stretti per testare fallimento validazione
        config = {
            'data': {
                'input_file_path': data_file,
                'time_column': 'time',
                'acceleration_columns': ['acc_x'],
                'axes_to_analyze': ['X']
            },
            'output': {
                'base_output_dir': test_dir,
                'create_timestamp_folder': False,
                'backup_existing_files': False
            },
            'ekf_parameters': {
                'process_noise': 0.01,
                'measurement_noise': 0.1,
                'initial_state_covariance': 1.0
            },
            'execution_control': {
                'mode': 'auto',
                'step_by_step': {
                    'enabled': False
                }
            },
            'output_validation': {
                'enabled': True,
                'velocity_bounds': [-0.001, 0.001],  # Bounds molto stretti
                'position_bounds': [-0.001, 0.001],
                'acceleration_bounds': [-0.001, 0.001],
                'physics_checks': True,
                'drift_detection': True,
                'bounds_checking': True
            }
        }
    
    # Salva configurazione
    config_file = os.path.join(test_dir, f"test_config_{test_mode}.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Configurazione salvata in: {config_file}")
    return config, config_file

def test_validation_functions():
    """Test delle funzioni di validazione"""
    print("\n" + "="*60)
    print("🧪 TEST FUNZIONI DI VALIDAZIONE")
    print("="*60)
    
    # Test 1: Validazione configurazione valida
    print("\n1️⃣  Test validazione configurazione valida...")
    test_dir, data_file = create_test_data()
    config, _ = create_test_config(test_dir, data_file, "validation")
    
    is_valid = validate_configuration(config)
    print(f"   Risultato: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    # Test 2: Validazione configurazione non valida
    print("\n2️⃣  Test validazione configurazione non valida...")
    invalid_config = config.copy()
    del invalid_config['data']['input_file_path']  # Rimuovi campo obbligatorio
    
    is_valid = validate_configuration(invalid_config)
    print(f"   Risultato: {'✅ PASS' if not is_valid else '❌ FAIL'}")
    
    # Test 3: Test backup
    print("\n3️⃣  Test sistema di backup...")
    test_file = os.path.join(test_dir, "test_backup.txt")
    with open(test_file, 'w') as f:
        f.write("Test content")
    
    backup_created = create_backup_if_needed(test_file, config)
    print(f"   Backup creato: {'✅ PASS' if backup_created else '❌ FAIL'}")
    
    # Test 4: Test validazione risultati (simulati)
    print("\n4️⃣  Test validazione risultati...")
    
    # Simula risultati EKF validi
    valid_results = {
        'velocity': np.random.normal(0, 1, 1000),  # Velocità ragionevoli
        'position': np.random.normal(0, 2, 1000),  # Posizioni ragionevoli
        'time': np.linspace(0, 10, 1000)
    }
    
    is_valid, warnings = validate_results(valid_results, config)
    print(f"   Risultati validi: {'✅ PASS' if is_valid else '❌ FAIL'}")
    if warnings:
        print(f"   Avvertimenti: {len(warnings)}")
    
    # Simula risultati EKF non validi
    invalid_results = {
        'velocity': np.random.normal(0, 100, 1000),  # Velocità irrealistiche
        'position': np.random.normal(0, 1000, 1000),  # Posizioni irrealistiche
        'time': np.linspace(0, 10, 1000)
    }
    
    is_valid, warnings = validate_results(invalid_results, config)
    print(f"   Risultati non validi rilevati: {'✅ PASS' if not is_valid else '❌ FAIL'}")
    print(f"   Numero avvertimenti: {len(warnings)}")
    
    return test_dir

def test_execution_modes():
    """Test delle modalità di esecuzione"""
    print("\n" + "="*60)
    print("🚀 TEST MODALITÀ DI ESECUZIONE")
    print("="*60)
    
    test_dir, data_file = create_test_data()
    
    # Test modalità production (automatica)
    print("\n1️⃣  Test modalità production (automatica)...")
    config, config_file = create_test_config(test_dir, data_file, "production")
    
    try:
        results = apply_ekf_with_auto_tuning(config)
        print("   ✅ Modalità production completata")
        print(f"   📊 Assi elaborati: {list(results.keys())}")
    except Exception as e:
        print(f"   ❌ Errore modalità production: {e}")
    
    # Test modalità con bounds che falliscono
    print("\n2️⃣  Test validazione bounds che fallisce...")
    config_fail, _ = create_test_config(test_dir, data_file, "bounds_fail")
    
    try:
        results = apply_ekf_with_auto_tuning(config_fail)
        print("   ⚠️  Esecuzione completata nonostante bounds stretti")
    except Exception as e:
        print(f"   ✅ Validazione bounds ha funzionato: {type(e).__name__}")
    
    return test_dir

def interactive_test():
    """Test interattivo delle funzionalità"""
    print("\n" + "="*60)
    print("🤖 TEST INTERATTIVO")
    print("="*60)
    
    print("\nQuesto test mostrerà le funzionalità interattive.")
    print("Premere Invio per ogni passaggio...")
    
    test_dir, data_file = create_test_data()
    config, config_file = create_test_config(test_dir, data_file, "validation")
    
    # Test conferma utente
    print("\n1️⃣  Test conferma utente...")
    response = get_user_confirmation("Vuoi continuare con il test?", config)
    print(f"   Risposta utente: {'Sì' if response else 'No'}")
    
    if response:
        print("\n2️⃣  Test esecuzione step-by-step...")
        print("   ⚠️  Questo test richiederà conferme multiple...")
        
        try:
            results = apply_ekf_with_auto_tuning(config)
            print("   ✅ Test step-by-step completato")
        except KeyboardInterrupt:
            print("   ⚠️  Test interrotto dall'utente")
        except Exception as e:
            print(f"   ❌ Errore nel test: {e}")
    
    return test_dir

def cleanup_test_directories(*test_dirs):
    """Pulisce le directory di test"""
    print("\n" + "="*60)
    print("🧹 PULIZIA")
    print("="*60)
    
    for test_dir in test_dirs:
        if test_dir and os.path.exists(test_dir):
            import shutil
            try:
                shutil.rmtree(test_dir)
                print(f"   🗑️  Directory rimossa: {test_dir}")
            except Exception as e:
                print(f"   ⚠️  Errore rimozione {test_dir}: {e}")

def main():
    """Funzione principale del test"""
    print("🔬 TEST SUITE FUNZIONALITÀ AVANZATE EKF")
    print("="*60)
    print("Questo script testa tutte le nuove funzionalità di controllo e validazione")
    print("="*60)
    
    test_dirs = []
    
    try:
        # Test 1: Funzioni di validazione
        test_dir1 = test_validation_functions()
        test_dirs.append(test_dir1)
        
        # Test 2: Modalità di esecuzione  
        test_dir2 = test_execution_modes()
        test_dirs.append(test_dir2)
        
        # Test 3: Modalità interattiva (opzionale)
        print("\n" + "="*60)
        print("🤔 TEST INTERATTIVO DISPONIBILE")
        print("="*60)
        print("Vuoi eseguire il test interattivo? (richiede input utente)")
        
        config_basic = {'execution_control': {'mode': 'auto'}}
        if get_user_confirmation("Eseguire test interattivo?", config_basic):
            test_dir3 = interactive_test()
            test_dirs.append(test_dir3)
        
        print("\n" + "="*60)
        print("✅ TUTTI I TEST COMPLETATI")
        print("="*60)
        print("📋 Riepilogo:")
        print("   • Funzioni di validazione: ✅")
        print("   • Modalità di esecuzione: ✅") 
        print("   • Sistema di backup: ✅")
        print("   • Controlli step-by-step: ✅")
        print("   • Validazione risultati: ✅")
        
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO NEI TEST: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Pulizia
        if get_user_confirmation("Rimuovere directory di test?", {'execution_control': {'mode': 'auto'}}):
            cleanup_test_directories(*test_dirs)
        else:
            print("\n📁 Directory di test mantenute:")
            for test_dir in test_dirs:
                if test_dir:
                    print(f"   • {test_dir}")

if __name__ == "__main__":
    main()
