import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = 'notebooks/emission_photon_energy/add_hydrogens_tuned_select_from_model.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print('Total cells:', len(cells))

for i, cell in enumerate(cells):
    ctype = cell['cell_type']
    src = ''.join(cell['source'])
    outputs = cell.get('outputs', [])
    out_text = ''
    for o in outputs:
        otype = o.get('output_type', '')
        if otype == 'stream':
            out_text += ''.join(o.get('text', []))
        elif otype in ['execute_result', 'display_data']:
            out_text += ''.join(o.get('data', {}).get('text/plain', []))
        elif otype == 'error':
            out_text += 'ERROR: ' + o.get('ename', '') + ': ' + o.get('evalue', '')
    has_image = any('image/png' in o.get('data', {}) for o in outputs)
    print('--- Cell', i, '(' + ctype + ') ---')
    print('  Source preview:', src[:120].replace('\n', ' | '))
    if out_text.strip():
        print('  Output:', out_text[:300].replace('\n', ' | '))
    if has_image:
        print('  [IMAGE OUTPUT PRESENT]')
    print()
