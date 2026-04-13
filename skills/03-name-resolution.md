Name resolution policy:
- If user input contains compound names, normalize or translate to English when needed.
- Use `resolve_smiles_from_name` first to obtain canonical SMILES.
- Perform downstream prediction tools only after SMILES is confirmed.
