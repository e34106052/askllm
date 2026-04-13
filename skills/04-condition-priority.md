Condition prediction priority:
- Use `run_askcos_condition_prediction` when the user asks for reaction condition recommendation and a valid reaction SMILES is available.
- If the query does not yet contain a valid `RCTS>>PRD` reaction SMILES, ask the user for the missing reaction representation first.
