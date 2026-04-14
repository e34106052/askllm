Task routing guide:
- Forward prediction: `run_askcos_forward_prediction`
- Retrosynthesis: `run_askcos_retrosynthesis`
- Multi-step route planning (MCTS): `run_askcos_multistep_retrosynthesis`
- Multi-step (Retro*): `run_askcos_multistep_retrosynthesis_retro_star`
- Compare MCTS vs Retro*: `run_askcos_multistep_retrosynthesis_compare`
- Route recommendation (multi-agent scoring): `run_askcos_route_recommendation`
- Impurity prediction: `run_askcos_impurity_prediction`
- Condition prediction: `run_askcos_condition_prediction`

Call the most specific tool that matches the user intent once valid SMILES or reaction input is available.
