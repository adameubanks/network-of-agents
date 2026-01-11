#!/usr/bin/env python3
"""
Compare paper_data.json with paper.tex and update the paper with correct values.
"""

import json
import re
from pathlib import Path

def load_paper_data():
    """Load the source of truth JSON."""
    with open('results/paper_data.json', 'r') as f:
        return json.load(f)

def update_table1(paper_content, data):
    """Update Table 1 (summary statistics)."""
    summary = data['summary_stats']
    
    # Find table section
    start_marker = r'\\begin\{table\}\[t\].*?\\label\{tab:summary\}'
    end_marker = r'\\end\{table\}'
    
    def find_and_replace_table(content):
        start_match = re.search(start_marker, content, re.DOTALL)
        if not start_match:
            return content
        
        start_pos = start_match.end()
        end_match = re.search(end_marker, content[start_pos:], re.DOTALL)
        if not end_match:
            return content
        
        end_pos = start_pos + end_match.start()
        table_section = content[start_pos:end_pos]
        
        # Replace rows
        rows = []
        for condition in ['nano, A vs B', 'nano, B vs A', 'mini, A vs B', 'mini, B vs A']:
            if condition in summary:
                s = summary[condition]
                rows.append(f"{condition} & {s['mean_rmse']:.2f} ({s['std_rmse']:.2f}) & {s['mean_mae']:.2f} ({s['std_mae']:.2f}) & {s['mean_corr']:.2f} ({s['std_corr']:.2f}) & {s['n']} \\\\")
        
        # Replace body
        body_pattern = r'(\\midrule\n)(.*?)(\\bottomrule)'
        new_body = r'\1' + '\n'.join(rows) + '\n' + r'\3'
        new_table_section = re.sub(body_pattern, new_body, table_section, flags=re.DOTALL)
        
        return content[:start_pos] + new_table_section + content[end_pos:]
    
    return find_and_replace_table(paper_content)

def update_table2(paper_content, data):
    """Update Table 2 (framing sensitivity)."""
    effects = data['framing_effects']
    
    # Find table - simpler approach: find the table section and replace rows
    start_marker = r'\\begin\{table\}\[t\].*?\\label\{tab:framing\}'
    end_marker = r'\\end\{table\}'
    
    def find_and_replace_table(content):
        # Find start
        start_match = re.search(start_marker, content, re.DOTALL)
        if not start_match:
            return content
        
        start_pos = start_match.end()
        
        # Find end
        end_match = re.search(end_marker, content[start_pos:], re.DOTALL)
        if not end_match:
            return content
        
        end_pos = start_pos + end_match.start()
        
        table_section = content[start_pos:end_pos]
        
        # Topic order from paper
        topic_order = [
            'Immigration', 'Restaurant etiquette', 'Child-free weddings',
            'Environment economy', 'Corporate activism', 'Hot dog sandwich',
            'Social media democracy', 'Toilet paper', 'Human cloning', 'Gun safety'
        ]
        
        rows = []
        nano_effects = []
        mini_effects = []
        
        for topic in topic_order:
            if topic in effects:
                e = effects[topic]
                nano_str = f"{e['nano']:+.2f}" if e['nano'] is not None else "---"
                mini_str = f"{e['mini']:+.2f}" if e['mini'] is not None else "---"
                rows.append(f"{topic} & {nano_str} & {mini_str} \\\\")
                
                if e['nano'] is not None:
                    nano_effects.append(e['nano'])
                if e['mini'] is not None:
                    mini_effects.append(e['mini'])
        
        # Update mean row
        mean_nano = np.mean(nano_effects) if nano_effects else None
        std_nano = np.std(nano_effects) if nano_effects else None
        mean_mini = np.mean(mini_effects) if mini_effects else None
        std_mini = np.std(mini_effects) if mini_effects else None
        
        # Find and replace the body (between \midrule after header and \midrule before mean)
        body_pattern = r'(\\midrule\n)(.*?)(\\midrule\n.*?Mean)'
        
        def replace_body(m):
            return m.group(1) + '\n'.join(rows) + '\n' + f'\\midrule\nMean (nano) & {mean_nano:.2f} ({std_nano:.2f}) & --- \\\\\nMean (mini) & --- & {mean_mini:.2f} ({std_mini:.2f}) \\\\'
        
        new_table_section = re.sub(body_pattern, replace_body, table_section, flags=re.DOTALL)
        
        return content[:start_pos] + new_table_section + content[end_pos:]
    
    return find_and_replace_table(paper_content)

def update_appendix_table(paper_content, data):
    """Update Appendix table (complete results)."""
    results = data['all_results']
    
    # Find table section
    start_marker = r'\\begin\{table\*\}\[t\].*?\\label\{tab:full_results\}'
    end_marker = r'\\end\{table\*\}'
    
    def find_and_replace_table(content):
        start_match = re.search(start_marker, content, re.DOTALL)
        if not start_match:
            return content
        
        start_pos = start_match.end()
        end_match = re.search(end_marker, content[start_pos:], re.DOTALL)
        if not end_match:
            return content
        
        end_pos = start_pos + end_match.start()
        table_section = content[start_pos:end_pos]
        
        # Group by topic
        by_topic = {}
        for r in results:
            topic = r['topic']
            if topic not in by_topic:
                by_topic[topic] = []
            by_topic[topic].append(r)
        
        # Topic order
        topic_order = [
            'Hot dog sandwich', 'Human cloning', 'Social media democracy',
            'Gun safety', 'Toilet paper', 'Child-free weddings',
            'Restaurant etiquette', 'Immigration', 'Corporate activism',
            'Environment economy'
        ]
        
        rows = []
        for topic in topic_order:
            if topic in by_topic:
                entries = by_topic[topic]
                # Sort: nano A vs B, nano B vs A, mini A vs B, mini B vs A
                sorted_entries = sorted(entries, key=lambda x: (x['model'], x['framing']))
                for r in sorted_entries:
                    rows.append(
                        f"{r['topic']} & {r['model']} & {r['framing']} & "
                        f"{r['rmse']:.2f} & {r['mae']:.2f} & {r['corr']:.2f} & "
                        f"{r['final_diff']:.2f} & {r['early_rmse']:.2f} & {r['late_rmse']:.2f} & "
                        f"{r['pattern']} \\\\"
                    )
                rows.append("\\midrule")
        
        # Replace body
        body_pattern = r'(\\midrule\n)(.*?)(\\bottomrule)'
        
        def replace_body(m):
            return m.group(1) + '\n'.join(rows) + '\n' + m.group(3)
        
        new_table_section = re.sub(body_pattern, replace_body, table_section, flags=re.DOTALL)
        
        return content[:start_pos] + new_table_section + content[end_pos:]
    
    return find_and_replace_table(paper_content)

def main():
    print("Loading paper data...")
    data = load_paper_data()
    
    print("Reading paper.tex...")
    with open('paper/paper.tex', 'r') as f:
        paper_content = f.read()
    
    print("Updating tables...")
    paper_content = update_table1(paper_content, data)
    paper_content = update_table2(paper_content, data)
    paper_content = update_appendix_table(paper_content, data)
    
    print("Writing updated paper.tex...")
    with open('paper/paper.tex', 'w') as f:
        f.write(paper_content)
    
    print("✓ Paper updated from paper_data.json")

if __name__ == "__main__":
    import numpy as np
    main()
