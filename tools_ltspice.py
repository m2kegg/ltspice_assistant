import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from langchain_core.tools import tool
from PyLTSpice import SimRunner, SpiceEditor, RawRead
from io import BytesIO
import base64
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional, Any, Union

from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional, Any, Union


@tool
def analyze_frequency(circuit_name: str, frequency_range: tuple, parameters: dict = None) -> dict:
    """Частотный анализ (AC-анализ) схемы."""
    runner = SimRunner(output_folder='./simulations')
    netlist = SpiceEditor(f"{circuit_name}.asc")
    if parameters:
        for param, value in parameters.items():
            netlist.set_parameter(param, value)
    start_freq, stop_freq, points = frequency_range
    netlist.add_instructions(f".ac dec {points} {start_freq} {stop_freq}")
    
    runner.run(netlist)
    
    results = {}
    for raw_file, log_file in runner:
        raw_data = RawRead(raw_file)
        
        frequencies = raw_data.get_trace('frequency').get_wave()
        gain_db = 20 * np.log10(np.abs(raw_data.get_trace('V(out)').get_wave()))
        phase = np.angle(raw_data.get_trace('V(out)').get_wave(), deg=True)
        
        try:
            max_gain = np.max(gain_db)
            bandwidth_indices = np.where(gain_db >= (max_gain - 3))
            if len(bandwidth_indices[0]) > 0:
                min_freq = frequencies[bandwidth_indices[0][0]]
                max_freq = frequencies[bandwidth_indices[0][-1]]
                bandwidth = max_freq - min_freq
            else:
                bandwidth = None
        except:
            bandwidth = None
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=frequencies,
            y=gain_db,
            name='АЧХ',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Амплитудно-частотная характеристика',
            xaxis_title='Частота (Гц)',
            yaxis_title='Усиление (дБ)',
            xaxis_type='log',
            template='plotly_white',
            height=500,
            width=800
        )
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=frequencies,
            y=phase,
            name='ФЧХ',
            line=dict(color='red', width=2)
        ))
        
        fig2.update_layout(
            title='Фазо-частотная характеристика',
            xaxis_title='Частота (Гц)',
            yaxis_title='Фаза (градусы)',
            xaxis_type='log',
            template='plotly_white',
            height=500,
            width=800
        )
        
        # Конвертация графиков в JSON для передачи
        plot_json = json.dumps({
            'ach': fig.to_json(),
            'fch': fig2.to_json()
        })
        
        results = {
            'frequencies': frequencies.tolist(),
            'gain_db': gain_db.tolist(),
            'phase': phase.tolist(),
            'bandwidth': bandwidth,
            'plot_json': plot_json,
            'summary': {
                'max_gain': float(np.max(gain_db)),
                'min_gain': float(np.min(gain_db)),
                'bandwidth': float(bandwidth) if bandwidth else None,
                'unity_gain_frequency': find_unity_gain_frequency(frequencies, gain_db)
            }
        }
        
    return results

@tool
def find_unity_gain_frequency(frequencies: np.ndarray, gain_db: np.ndarray) -> float:
    """Находит частоту единичного усиления"""
    try:
        for i in range(len(gain_db)-1):
            if (gain_db[i] >= 0 and gain_db[i+1] < 0) or (gain_db[i] <= 0 and gain_db[i+1] > 0):
                x1, y1 = frequencies[i], gain_db[i]
                x2, y2 = frequencies[i+1], gain_db[i+1]
                unity_freq = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                return float(unity_freq)
        return None
    except:
        return None
    
@tool
def analyze_monte_carlo(circuit_name: str, param_variations: dict, num_runs: int, analysis_type: str = "tran") -> dict:
    """Анализ Монте-Карло для оценки влияния разброса параметров."""
    results = []
    
    for run in range(num_runs):
        run_params = {}
        for param, variation in param_variations.items():
            base_value = variation['value']
            deviation = variation['deviation']
            if variation['distribution'] == 'normal':
                value = np.random.normal(base_value, base_value * deviation / 100)
            elif variation['distribution'] == 'uniform':
                min_val = base_value * (1 - deviation / 100)
                max_val = base_value * (1 + deviation / 100)
                value = np.random.uniform(min_val, max_val)
            run_params[param] = value
        
        runner = SimRunner(output_folder='./simulations')
        netlist = SpiceEditor(f"{circuit_name}.asc")
        
        for param, value in run_params.items():
            netlist.set_component_value(param, f"{value}")
        
        if analysis_type == "tran":
            netlist.add_instructions(".tran 0 10m 0 0.01m")
        elif analysis_type == "ac":
            netlist.add_instructions(".ac dec 20 1 100k")
        
        runner.run(netlist)
        
        for raw_file, log_file in runner:
            raw_data = RawRead(raw_file)
            
            if analysis_type == "tran":
                time_data = raw_data.get_trace('time').get_wave()
                output_data = raw_data.get_trace('V(out)').get_wave()
                
                rise_time, fall_time, overshoot = calculate_transient_params(time_data, output_data)
                
                run_result = {
                    'run': run + 1,
                    'parameters': run_params,
                    'rise_time': rise_time,
                    'fall_time': fall_time,
                    'overshoot': overshoot
                }
            
            elif analysis_type == "ac":
                frequencies = raw_data.get_trace('frequency').get_wave()
                gain = 20 * np.log10(np.abs(raw_data.get_trace('V(out)').get_wave()))
                max_gain = np.max(gain)
                unity_gain_freq = find_unity_gain_frequency(frequencies, gain)
                
                run_result = {
                    'run': run + 1,
                    'parameters': run_params,
                    'max_gain': float(max_gain),
                    'unity_gain_frequency': unity_gain_freq
                }
            
            results.append(run_result)
    
    stats = calculate_monte_carlo_stats(results, analysis_type)
    
    return {
        'runs': results,
        'statistics': stats
    }

@tool
def calculate_monte_carlo_stats(results: list, analysis_type: str) -> dict:
    """Расчет статистики для результатов Монте-Карло."""
    stats = {}
    
    if analysis_type == "tran":
        rise_times = [r['rise_time'] for r in results if r['rise_time'] is not None]
        fall_times = [r['fall_time'] for r in results if r['fall_time'] is not None]
        overshoots = [r['overshoot'] for r in results if r['overshoot'] is not None]
        
        stats = {
            'rise_time': {
                'mean': np.mean(rise_times) if rise_times else None,
                'std': np.std(rise_times) if rise_times else None,
                'min': np.min(rise_times) if rise_times else None,
                'max': np.max(rise_times) if rise_times else None
            },
            'fall_time': {
                'mean': np.mean(fall_times) if fall_times else None,
                'std': np.std(fall_times) if fall_times else None,
                'min': np.min(fall_times) if fall_times else None,
                'max': np.max(fall_times) if fall_times else None
            },
            'overshoot': {
                'mean': np.mean(overshoots) if overshoots else None,
                'std': np.std(overshoots) if overshoots else None,
                'min': np.min(overshoots) if overshoots else None,
                'max': np.max(overshoots) if overshoots else None
            }
        }
    
    elif analysis_type == "ac":
        max_gains = [r['max_gain'] for r in results if r['max_gain'] is not None]
        unity_freqs = [r['unity_gain_frequency'] for r in results if r['unity_gain_frequency'] is not None]
        
        stats = {
            'max_gain': {
                'mean': np.mean(max_gains) if max_gains else None,
                'std': np.std(max_gains) if max_gains else None,
                'min': np.min(max_gains) if max_gains else None,
                'max': np.max(max_gains) if max_gains else None
            },
            'unity_gain_frequency': {
                'mean': np.mean(unity_freqs) if unity_freqs else None,
                'std': np.std(unity_freqs) if unity_freqs else None,
                'min': np.min(unity_freqs) if unity_freqs else None,
                'max': np.max(unity_freqs) if unity_freqs else None
            }
        }
    
    return stats

def calculate_transient_params(time_data: np.ndarray, output_data: np.ndarray) -> tuple:
    """Расчет параметров переходного процесса."""
    try:
        result = {}
        steady_state = output_data[-1]
        initial_state = output_data[0]
        threshold_10 = initial_state + 0.1 * (steady_state - initial_state)
        threshold_90 = initial_state + 0.9 * (steady_state - initial_state)
        
        t_10_idx = np.where(output_data >= threshold_10)[0][0]
        t_90_idx = np.where(output_data >= threshold_90)[0][0]
        
        rise_time = time_data[t_90_idx] - time_data[t_10_idx]
        fall_time = None
        max_value = np.max(output_data)
        overshoot_percent = ((max_value - steady_state) / steady_state) * 100 if steady_state != 0 else 0
        result["rise_time"] = rise_time
        result["fall_time"] = fall_time
        result["overshoot_percent"] = overshoot_percent
        return result
    except:
        return None, None, None
    

def analyze_transient(circuit_name: str, load_list: list, input_voltage_list: list) -> dict:
    """Анализ переходных процессов при различных нагрузках и входных напряжениях."""
    results = {}
    
    for load in load_list:
        for voltage in input_voltage_list:
            runner = SimRunner(output_folder='./simulations')
            netlist = SpiceEditor(f"{circuit_name}.asc")
            
            netlist.set_component_value('Rload', load)
            netlist.set_component_value('V1', voltage)
            netlist.add_instructions(".tran 0 10m 0 0.01m")
            
            runner.run(netlist)
            
            for raw_file, log_file in runner:
                raw_data = RawRead(raw_file)
                time_data = raw_data.get_trace('time').get_wave()
                output_data = raw_data.get_trace('V(out)').get_wave()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_data,
                    y=output_data,
                    name='Выходное напряжение',
                    line=dict(color='green', width=2)
                ))
                
                fig.update_layout(
                    title=f'Переходный процесс (нагрузка={load}, вход={voltage})',
                    xaxis_title='Время (с)',
                    yaxis_title='Напряжение (В)',
                    template='plotly_white',
                    height=500,
                    width=800
                )
                
                plot_json = json.dumps({
                    'transient': fig.to_json()
                })
                
                transient_data = calculate_transient_params(time_data, output_data)
                transient_data['plot_json'] = plot_json
                results[f"load={load},voltage={voltage}"] = transient_data
    
    return results

@tool
def analyze_noise(circuit_name: str, frequency_range: tuple, output_node: str = "V(out)", input_source: str = "V1") -> dict:
    """
    Шумовой анализ схемы с использованием LTSpice.

    Args:
        circuit_name (str): Имя файла схемы LTSpice (без расширения).
        frequency_range (tuple): Диапазон частот для анализа (start_freq, stop_freq, points_per_decade).
        output_node (str): Узел выхода для анализа шума.
        input_source (str): Источник входного сигнала.

    Returns:
        dict: Результаты анализа шума, включая данные и графики.
    """
    try:
        runner = SimRunner(output_folder='./simulations')
        netlist = SpiceEditor(f"{circuit_name}.asc")

        start_freq, stop_freq, points_per_decade = frequency_range
        netlist.add_instructions(f".noise {output_node} {input_source} dec {points_per_decade} {start_freq} {stop_freq}")

        runner.run(netlist)

        results = {}
        for raw_file, log_file in runner:
            raw_data = RawRead(raw_file)

            frequencies = raw_data.get_trace('frequency').get_wave()
            noise_voltage = raw_data.get_trace('onoise').get_wave()  # Выходной шум
            input_referred_noise = raw_data.get_trace('inoise').get_wave()  # Входной шум

            # Расчет SNR (Signal-to-Noise Ratio)
            snr = 20 * np.log10(np.abs(noise_voltage / input_referred_noise))

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=frequencies,
                y=noise_voltage,
                mode='lines',
                name='Выходной шум',
                line=dict(color='blue', width=2)
            ))
            fig1.update_layout(
                title='Выходной шум',
                xaxis_title='Частота (Гц)',
                yaxis_title='Шумовое напряжение (В/√Гц)',
                xaxis_type='log',
                template='plotly_white'
            )

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=frequencies,
                y=input_referred_noise,
                mode='lines',
                name='Входной шум',
                line=dict(color='red', width=2)
            ))
            fig2.update_layout(
                title='Входной шум',
                xaxis_title='Частота (Гц)',
                yaxis_title='Шумовое напряжение (В/√Гц)',
                xaxis_type='log',
                template='plotly_white'
            )

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=frequencies,
                y=snr,
                mode='lines',
                name='Отношение сигнал/шум (SNR)',
                line=dict(color='green', width=2)
            ))
            fig3.update_layout(
                title='Отношение сигнал/шум (SNR)',
                xaxis_title='Частота (Гц)',
                yaxis_title='SNR (дБ)',
                xaxis_type='log',
                template='plotly_white'
            )

            plot_json = json.dumps({
                'output_noise': fig1.to_json(),
                'input_noise': fig2.to_json(),
                'snr': fig3.to_json()
            })

            results = {
                'frequencies': frequencies.tolist(),
                'output_noise': noise_voltage.tolist(),
                'input_noise': input_referred_noise.tolist(),
                'snr': snr.tolist(),
                'plot_json': plot_json,
                'summary': {
                    'max_output_noise': float(np.max(noise_voltage)),
                    'min_output_noise': float(np.min(noise_voltage)),
                    'max_snr': float(np.max(snr)),
                    'min_snr': float(np.min(snr))
                }
            }
        
        return results

    except Exception as e:
        return {"error": str(e)}

@tool
def generate_report(circuit_name: str, analyses_results: dict) -> dict:
    """Создание комплексного отчета на основе результатов анализа."""
    report = {
        'circuit_name': circuit_name,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'analyses': analyses_results,
        'summary': {}
    }
    # Формирование сводки
    if 'stability' in analyses_results:
        stable_count = sum(1 for result in analyses_results['stability'] if result['stability'] == 'Стабильна')
        total_count = len(analyses_results['stability'])
        report['summary']['stability'] = {
            'stable_count': stable_count,
            'total_count': total_count,
            'stable_percentage': (stable_count / total_count * 100) if total_count > 0 else 0
        }
    
    if 'frequency' in analyses_results:
        report['summary']['frequency'] = {
            'bandwidth': analyses_results['frequency'].get('bandwidth'),
            'max_gain': analyses_results['frequency'].get('summary', {}).get('max_gain')
        }
    
    if 'monte_carlo' in analyses_results:
        if analyses_results['monte_carlo'].get('statistics'):
            report['summary']['monte_carlo'] = analyses_results['monte_carlo']['statistics']
    
    summary_data = []
    
    if 'stability' in analyses_results:
        for result in analyses_results['stability']:
            summary_data.append({
                'Анализ': 'Стабильность',
                'Параметр': f"Запас по фазе: {result.get('phase_margin')}°, Запас по усилению: {result.get('gain_margin')} дБ",
                'Результат': result.get('stability')
            })
    
    if 'frequency' in analyses_results:
        freq_data = analyses_results['frequency']
        if 'summary' in freq_data:
            for key, value in freq_data['summary'].items():
                if key != 'plot':
                    summary_data.append({
                        'Анализ': 'Частотный',
                        'Параметр': key.replace('_', ' ').title(),
                        'Результат': f"{value:.2f}" if isinstance(value, (int, float)) and value is not None else str(value)
                    })
    
    if 'monte_carlo' in analyses_results:
        mc_stats = analyses_results['monte_carlo'].get('statistics', {})
        for param_type, stats in mc_stats.items():
            if stats.get('mean') is not None:
                summary_data.append({
                    'Анализ': 'Монте-Карло',
                    'Параметр': param_type.replace('_', ' ').title(),
                    'Результат': f"Среднее: {stats['mean']:.2e}, Разброс: {stats['std']:.2e}"
                })
    
    df = pd.DataFrame(summary_data)
    if not df.empty:
        report['summary_table'] = df.to_markdown()
    return report
