#!/usr/bin/env python3
"""
Verify that paper.tex matches paper_data.json.
"""

import json
import re
from pathlib import Path

def load_paper_data():
    """Load the source of truth JSON."""
    with open('results/paper_data.json', 'r') as f:
        return json.load(f)

def extract_table1_values(paper_content):
    """Extract values from Table 1."""
    pattern = r'\\begin\{table\}\[t\].*?\\label\{tab:summary\}.*?\\midrule\n(.*?)\\bottomrule'
    match = re.search(pattern, paper_content, re.DOTALL)
    if not match:
        return None
    
    body = match.group(1)
    # Each row is on its own line, ending with \ or \\
    # Split by lines and process each
    lines = body.split('\n')
    values = {}
    current_row = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # If line ends with \ or \\, it's the end of a row
        if line.endswith('\\') or line.endswith('\\\\'):
            # Combine with previous parts if any
            full_row = ' '.join(current_row) + ' ' + line.rstrip('\\')
            current_row = []
            
            if '&' in full_row:
                parts = [p.strip() for p in full_row.split('&')]
                if len(parts) >= 5:
                    condition = parts[0]
                    # Parse: "0.35 (0.20)"
                    rmse_match = re.match(r'([\d.]+)\s+\(([\d.]+)\)', parts[1])
                    mae_match = re.match(r'([\d.]+)\s+\(([\d.]+)\)', parts[2])
                    corr_match = re.match(r'([\d.-]+)\s+\(([\d.]+)\)', parts[3])
                    # n might have trailing \\ or whitespace
                    n_str = parts[4].split()[0] if parts[4].split() else parts[4]
                    try:
                        n = int(n_str)
                    except:
                        continue
                    
                    if rmse_match and mae_match and corr_match:
                        values[condition] = {
                            'mean_rmse': float(rmse_match.group(1)),
                            'std_rmse': float(rmse_match.group(2)),
                            'mean_mae': float(mae_match.group(1)),
                            'std_mae': float(mae_match.group(2)),
                            'mean_corr': float(corr_match.group(1)),
                            'std_corr': float(corr_match.group(2)),
                            'n': n
                        }
        else:
            # Continuation of a row
            current_row.append(line)
    
    return values

def extract_table2_values(paper_content):
    """Extract values from Table 2."""
    pattern = r'\\begin\{table\}\[t\].*?\\label\{tab:framing\}.*?\\midrule\n(.*?)\\midrule.*?Mean'
    match = re.search(pattern, paper_content, re.DOTALL)
    if not match:
        return None
    
    body = match.group(1)
    rows = re.split(r'\\+\\s*\n', body)
    values = {}
    
    for row in rows:
        row = row.strip()
        if '&' not in row or not row:
            continue
        parts = [p.strip() for p in row.split('&')]
        if len(parts) >= 3:
            topic = parts[0]
            nano_str = parts[1] if len(parts) > 1 else '---'
            mini_str = parts[2] if len(parts) > 2 else '---'
            
            nano_val = None if nano_str == '---' else float(nano_str.replace('+', ''))
            mini_val = None if mini_str == '---' else float(mini_str.replace('+', ''))
            
            values[topic] = {'nano': nano_val, 'mini': mini_val}
    
    return values

def extract_appendix_values(paper_content):
    """Extract values from Appendix table."""
    pattern = r'\\begin\{table\*\}\[t\].*?\\label\{tab:full_results\}.*?\\midrule\n(.*?)\\bottomrule'
    match = re.search(pattern, paper_content, re.DOTALL)
    if not match:
        return None
    
    body = match.group(1)
    # Each row is on its own line, ending with \\ or \
    lines = body.split('\n')
    values = {}
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and midrule lines
        if not line or '\\midrule' in line or line == 'midrule':
            continue
        
        # If line ends with \ or \\, it's a complete row
        if line.endswith('\\') or line.endswith('\\\\'):
            # Remove trailing backslashes
            clean_line = line.rstrip('\\').strip()
            
            if '&' in clean_line:
                parts = [p.strip() for p in clean_line.split('&')]
                if len(parts) >= 10:
                    topic = parts[0]
                    model = parts[1]
                    framing = parts[2]
                    
                    # Skip if topic looks wrong (like "midrule")
                    if topic.lower() in ['midrule', 'toprule', 'bottomrule'] or not topic:
                        continue
                    
                    key = (topic, model, framing)
                    
                    try:
                        values[key] = {
                            'rmse': float(parts[3]),
                            'mae': float(parts[4]),
                            'corr': float(parts[5]),
                            'final_diff': float(parts[6]),
                            'early_rmse': float(parts[7]),
                            'late_rmse': float(parts[8]),
                            'pattern': parts[9].rstrip('\\').strip()
                        }
                    except (ValueError, IndexError) as e:
                        continue
    
    return values

def verify_table1(data, paper_values):
    """Verify Table 1."""
    summary = data['summary_stats']
    discrepancies = []
    
    for condition in ['nano, A vs B', 'nano, B vs A', 'mini, A vs B', 'mini, B vs A']:
        if condition not in summary:
            continue
        if condition not in paper_values:
            discrepancies.append(f"Missing {condition} in paper")
            continue
        
        expected = summary[condition]
        actual = paper_values[condition]
        tolerance = 0.01
        
        for metric in ['mean_rmse', 'std_rmse', 'mean_mae', 'std_mae', 'mean_corr', 'std_corr']:
            if abs(expected[metric] - actual[metric]) > tolerance:
                discrepancies.append(
                    f"{condition} {metric}: expected {expected[metric]:.2f}, got {actual[metric]:.2f}"
                )
        if expected['n'] != actual['n']:
            discrepancies.append(f"{condition} n: expected {expected['n']}, got {actual['n']}")
    
    return discrepancies

def verify_table2(data, paper_values):
    """Verify Table 2."""
    effects = data['framing_effects']
    discrepancies = []
    tolerance = 0.01
    
    for topic in effects:
        if topic not in paper_values:
            discrepancies.append(f"Missing {topic} in paper")
            continue
        
        expected = effects[topic]
        actual = paper_values[topic]
        
        if expected['nano'] is not None:
            if actual['nano'] is None:
                discrepancies.append(f"{topic} nano: expected {expected['nano']:.2f}, got None")
            elif abs(expected['nano'] - actual['nano']) > tolerance:
                discrepancies.append(f"{topic} nano: expected {expected['nano']:.2f}, got {actual['nano']:.2f}")
        elif actual['nano'] is not None:
            discrepancies.append(f"{topic} nano: expected None, got {actual['nano']:.2f}")
        
        if expected['mini'] is not None:
            if actual['mini'] is None:
                discrepancies.append(f"{topic} mini: expected {expected['mini']:.2f}, got None")
            elif abs(expected['mini'] - actual['mini']) > tolerance:
                discrepancies.append(f"{topic} mini: expected {expected['mini']:.2f}, got {actual['mini']:.2f}")
        elif actual['mini'] is not None:
            discrepancies.append(f"{topic} mini: expected None, got {actual['mini']:.2f}")
    
    return discrepancies

def verify_appendix(data, paper_values):
    """Verify Appendix table."""
    results = data['all_results']
    discrepancies = []
    tolerance = 0.02
    
    for r in results:
        key = (r['topic'], r['model'], r['framing'])
        
        if key not in paper_values:
            discrepancies.append(f"Missing {key} in paper")
            continue
        
        actual = paper_values[key]
        
        for metric in ['rmse', 'mae', 'corr', 'final_diff', 'early_rmse', 'late_rmse']:
            if abs(r[metric] - actual[metric]) > tolerance:
                discrepancies.append(
                    f"{key} {metric}: expected {r[metric]:.2f}, got {actual[metric]:.2f}"
                )
        
        if r['pattern'] != actual['pattern']:
            discrepancies.append(f"{key} pattern: expected '{r['pattern']}', got '{actual['pattern']}'")
    
    return discrepancies

def main():
    print("="*80)
    print("VERIFYING PAPER AGAINST paper_data.json")
    print("="*80)
    
    # Load data
    print("\nLoading paper_data.json...")
    data = load_paper_data()
    
    # Load paper
    print("Loading paper.tex...")
    with open('paper/paper.tex', 'r') as f:
        paper_content = f.read()
    
    # Extract values
    print("Extracting values from paper...")
    table1_values = extract_table1_values(paper_content)
    table2_values = extract_table2_values(paper_content)
    appendix_values = extract_appendix_values(paper_content)
    
    if not table1_values:
        print("⚠️  Could not extract Table 1 values")
    if not table2_values:
        print("⚠️  Could not extract Table 2 values")
    if not appendix_values:
        print("⚠️  Could not extract Appendix table values")
    
    # Verify
    print("\n" + "="*80)
    print("VERIFYING TABLE 1 (Summary Statistics)")
    print("="*80)
    table1_issues = verify_table1(data, table1_values) if table1_values else ["Could not extract"]
    
    print("\n" + "="*80)
    print("VERIFYING TABLE 2 (Framing Sensitivity)")
    print("="*80)
    table2_issues = verify_table2(data, table2_values) if table2_values else ["Could not extract"]
    
    print("\n" + "="*80)
    print("VERIFYING APPENDIX TABLE (Complete Results)")
    print("="*80)
    appendix_issues = verify_appendix(data, appendix_values) if appendix_values else ["Could not extract"]
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    all_issues = table1_issues + table2_issues + appendix_issues
    
    if not all_issues:
        print("✓ ALL TABLES MATCH paper_data.json!")
    else:
        print(f"⚠️  FOUND {len(all_issues)} DISCREPANCIES:\n")
        for issue in all_issues:
            print(f"  - {issue}")
    
    # Save report
    report_file = "results/verification_report.txt"
    with open(report_file, 'w') as f:
        f.write("PAPER VERIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated from paper_data.json\n\n")
        
        if not all_issues:
            f.write("✓ ALL TABLES MATCH paper_data.json!\n")
        else:
            f.write(f"⚠️  FOUND {len(all_issues)} DISCREPANCIES:\n\n")
            if table1_issues:
                f.write("TABLE 1 ISSUES:\n")
                for issue in table1_issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")
            if table2_issues:
                f.write("TABLE 2 ISSUES:\n")
                for issue in table2_issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")
            if appendix_issues:
                f.write("APPENDIX TABLE ISSUES:\n")
                for issue in appendix_issues:
                    f.write(f"  - {issue}\n")
    
    print(f"\n✓ Report saved to {report_file}")

if __name__ == "__main__":
    main()
