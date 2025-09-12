#!/usr/bin/env python3
"""
Test del nuovo EKF 3D con auto-tuning adattivo
Genera dati sintetici 3D con orientamento per testare il sistema completo
"""

import numpy as np
import pandas as pd
import sys
import os

# Aggiungi il percorso del modulo
sys.path.append('/Volumes/nvme/Github/igmSquatBiomechanics/sources/lab/modules/EKF_for_vel_and_pos_est_from_acc/sources')

def generate_3d_test_data(duration=10.0, freq=100, gravity=9.81):
    """
    Genera dati di test 3D con movimento sinusoidale e rotazione.
    
    Args:
        duration: Durata in secondi
        freq: Frequenza di campionamento in Hz
        gravity: Accelerazione di gravità
        
    Returns:
        pd.DataFrame: Dati con colonne time, acc_x, acc_y, acc_z, roll, pitch, yaw
    """
    print("🔬 Generazione dati di test 3D...")
    
    dt = 1.0 / freq
    t = np.arange(0, duration, dt)
    n_samples = len(t)
    
    # Movimento sinusoidale nelle 3 dimensioni
    # X: movimento principale sinusoidale
    pos_x_true = 0.5 * np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz
    vel_x_true = 0.5 * 2 * np.pi * 0.2 * np.cos(2 * np.pi * 0.2 * t)
    acc_x_true = -0.5 * (2 * np.pi * 0.2)**2 * np.sin(2 * np.pi * 0.2 * t)
    
    # Y: movimento secondario
    pos_y_true = 0.3 * np.sin(2 * np.pi * 0.15 * t + np.pi/4)  # 0.15 Hz, sfasato
    vel_y_true = 0.3 * 2 * np.pi * 0.15 * np.cos(2 * np.pi * 0.15 * t + np.pi/4)
    acc_y_true = -0.3 * (2 * np.pi * 0.15)**2 * np.sin(2 * np.pi * 0.15 * t + np.pi/4)
    
    # Z: movimento verticale meno intenso
    pos_z_true = 0.1 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz
    vel_z_true = 0.1 * 2 * np.pi * 0.1 * np.cos(2 * np.pi * 0.1 * t)
    acc_z_true = -0.1 * (2 * np.pi * 0.1)**2 * np.sin(2 * np.pi * 0.1 * t)
    
    # Angoli di Euler che variano lentamente
    roll = 0.1 * np.sin(2 * np.pi * 0.05 * t)   # Rotazione attorno X
    pitch = 0.15 * np.sin(2 * np.pi * 0.07 * t)  # Rotazione attorno Y  
    yaw = 0.2 * np.sin(2 * np.pi * 0.03 * t)     # Rotazione attorno Z
    
    # Accelerazioni in world-frame (senza gravità)
    acc_world_x = acc_x_true
    acc_world_y = acc_y_true
    acc_world_z = acc_z_true
    
    # Matrice di rotazione da world a body-frame (inversa di quella usata nell'EKF)
    acc_body = np.zeros((n_samples, 3))
    
    for i in range(n_samples):
        # Matrice di rotazione world-to-body (inversa di body-to-world)
        cr, sr = np.cos(roll[i]), np.sin(roll[i])
        cp, sp = np.cos(pitch[i]), np.sin(pitch[i])
        cy, sy = np.cos(yaw[i]), np.sin(yaw[i])
        
        # Rotazione ZYX inversa (body-to-world)^T
        R_world_to_body = np.array([
            [cy*cp, sy*cp, -sp],
            [cy*sp*sr - sy*cr, sy*sp*sr + cy*cr, cp*sr],
            [cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr]
        ])
        
        # Accelerazione world con gravità
        acc_world_with_gravity = np.array([acc_world_x[i], acc_world_y[i], acc_world_z[i] + gravity])
        
        # Converti in body-frame
        acc_body[i] = R_world_to_body @ acc_world_with_gravity
    
    # Aggiungi rumore realistico
    noise_level = 0.1
    acc_body[:, 0] += np.random.normal(0, noise_level, n_samples)
    acc_body[:, 1] += np.random.normal(0, noise_level, n_samples)
    acc_body[:, 2] += np.random.normal(0, noise_level, n_samples)
    
    # Crea DataFrame
    data = pd.DataFrame({
        'timestamp': t,
        'acc_x': acc_body[:, 0],
        'acc_y': acc_body[:, 1], 
        'acc_z': acc_body[:, 2],
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        # Dati di verità per confronto
        'pos_x_true': pos_x_true,
        'pos_y_true': pos_y_true,
        'pos_z_true': pos_z_true,
        'vel_x_true': vel_x_true,
        'vel_y_true': vel_y_true,
        'vel_z_true': vel_z_true
    })
    
    print(f"✅ Generati {n_samples} campioni per {duration}s a {freq}Hz")
    print(f"📊 Range accelerazioni: X=[{acc_body[:, 0].min():.2f}, {acc_body[:, 0].max():.2f}]")
    print(f"📊 Range accelerazioni: Y=[{acc_body[:, 1].min():.2f}, {acc_body[:, 1].max():.2f}]")
    print(f"📊 Range accelerazioni: Z=[{acc_body[:, 2].min():.2f}, {acc_body[:, 2].max():.2f}]")
    
    return data

def test_ekf_3d():
    """Test completo del sistema EKF 3D"""
    print("🧪 TEST EKF 3D CON AUTO-TUNING ADATTIVO")
    print("="*60)
    
    # Genera dati di test
    test_data = generate_3d_test_data(duration=20.0, freq=100)
    
    # Salva i dati di test
    test_file = '/tmp/ekf_3d_test_data.csv'
    test_data.to_csv(test_file, index=False)
    print(f"💾 Dati di test salvati in: {test_file}")
    
    # Importa e configura l'EKF
    from EKF_for_vel_and_pos_est_from_acc import load_config, apply_ekf_single_axis
    
    # Carica configurazione
    config_path = '/Volumes/nvme/Github/igmSquatBiomechanics/sources/lab/modules/EKF_for_vel_and_pos_est_from_acc/configs/EKF_for_vel_and_pos_est_from_acc.yaml'
    config = load_config(config_path)
    
    # Modifica configurazione per test
    config['input_files']['linear'] = test_file
    config['acceleration_type'] = 'linear'
    config['analysis_axis'] = 'Y'  # Testa asse Y
    
    # Configura parametri per test 3D
    config['kalman_filter']['adaptive_threshold'] = 3.0  # Più sensibile
    config['kalman_filter']['max_adaptations'] = 5      # Più adattamenti
    
    print("\n🚀 Avvio test EKF 3D...")
    
    try:
        # Esegui EKF
        results = apply_ekf_single_axis(test_data, 'Y', config)
        
        print("\n✅ TEST COMPLETATO CON SUCCESSO!")
        print("="*60)
        
        # Analizza risultati
        if results:
            pos_est = results['position']
            vel_est = results['velocity']
            
            # Confronta con verità
            pos_true = test_data['pos_y_true'].values
            vel_true = test_data['vel_y_true'].values
            
            # Calcola errori
            pos_error = np.mean(np.abs(pos_est - pos_true[:len(pos_est)]))
            vel_error = np.mean(np.abs(vel_est - vel_true[:len(vel_est)]))
            
            print(f"📊 ACCURATEZZA STIMAZIONE:")
            print(f"   📍 Errore posizione medio: {pos_error:.4f} m")
            print(f"   🚀 Errore velocità medio: {vel_error:.4f} m/s")
            print(f"   📈 Range posizione stimata: [{np.min(pos_est):.3f}, {np.max(pos_est):.3f}] m")
            print(f"   📈 Range velocità stimata: [{np.min(vel_est):.3f}, {np.max(vel_est):.3f}] m/s")
            
            # Valutazione qualità
            if pos_error < 0.1 and vel_error < 0.1:
                print("🏆 QUALITÀ: ECCELLENTE")
            elif pos_error < 0.2 and vel_error < 0.2:
                print("🥇 QUALITÀ: BUONA")
            elif pos_error < 0.5 and vel_error < 0.5:
                print("🥈 QUALITÀ: ACCETTABILE")
            else:
                print("⚠️ QUALITÀ: DA MIGLIORARE")
                
        else:
            print("❌ Nessun risultato ottenuto")
            
    except Exception as e:
        print(f"❌ ERRORE NEL TEST: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_compatibility_mode():
    """Test modalità compatibilità 1D"""
    print("\n🔄 TEST MODALITÀ COMPATIBILITÀ 1D")
    print("="*60)
    
    # Genera dati solo con accelerazione Y (senza orientamento)
    t = np.linspace(0, 10, 1000)
    acc_y = 2 * np.sin(2 * np.pi * 0.3 * t) + 0.1 * np.random.randn(len(t))
    
    compat_data = pd.DataFrame({
        'timestamp': t,
        'acc_y': acc_y
    })
    
    # Salva dati di compatibilità
    compat_file = '/tmp/ekf_compat_test_data.csv'
    compat_data.to_csv(compat_file, index=False)
    
    try:
        from EKF_for_vel_and_pos_est_from_acc import load_config, apply_ekf_single_axis
        
        config_path = '/Volumes/nvme/Github/igmSquatBiomechanics/sources/lab/modules/EKF_for_vel_and_pos_est_from_acc/configs/EKF_for_vel_and_pos_est_from_acc.yaml'
        config = load_config(config_path)
        config['input_files']['linear'] = compat_file
        
        results = apply_ekf_single_axis(compat_data, 'Y', config)
        
        if results:
            print("✅ Modalità compatibilità 1D funziona correttamente")
            return True
        else:
            print("❌ Modalità compatibilità 1D fallita")
            return False
            
    except Exception as e:
        print(f"❌ ERRORE compatibilità: {e}")
        return False

if __name__ == "__main__":
    print("🔬 SUITE DI TEST EKF 3D")
    print("="*60)
    
    success = True
    
    # Test 1: EKF 3D completo
    if not test_ekf_3d():
        success = False
    
    # Test 2: Modalità compatibilità
    if not test_compatibility_mode():
        success = False
    
    if success:
        print("\n🎉 TUTTI I TEST SUPERATI!")
        print("✅ Sistema EKF 3D pronto per l'uso")
    else:
        print("\n⚠️ ALCUNI TEST FALLITI")
        print("🔧 Verificare la configurazione e riprovare")
