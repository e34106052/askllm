from typing import Any, Dict, List, Tuple


def parse_uds_paths(uds: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(uds, dict):
        return []
    node_dict = uds.get("node_dict", {}) or {}
    uuid2smiles = uds.get("uuid2smiles", {}) or {}
    pathways = uds.get("pathways", []) or []
    pathway_props = uds.get("pathways_properties", []) or []
    routes = []
    for i, edges in enumerate(pathways, start=1):
        props = pathway_props[i - 1] if isinstance(pathway_props, list) and len(pathway_props) >= i else {}
        routes.append(
            {
                "route_id": i,
                "edges": edges if isinstance(edges, list) else [],
                "props": props if isinstance(props, dict) else {},
                "node_dict": node_dict if isinstance(node_dict, dict) else {},
                "uuid2smiles": uuid2smiles if isinstance(uuid2smiles, dict) else {},
            }
        )
    return routes


def _route_nodes(edges: List[Dict[str, Any]], uuid2smiles: Dict[str, str]) -> Tuple[List[str], List[str]]:
    reactions, chemicals = [], []
    seen_r, seen_c = set(), set()
    for e in edges:
        if not isinstance(e, dict):
            continue
        for uid in (e.get("source"), e.get("target")):
            if not uid:
                continue
            smi = uuid2smiles.get(uid, uid)
            if ">>" in str(smi):
                if smi not in seen_r:
                    seen_r.add(smi)
                    reactions.append(smi)
            else:
                if smi not in seen_c:
                    seen_c.add(smi)
                    chemicals.append(smi)
    return reactions, chemicals


def _leaf_chemicals(edges: List[Dict[str, Any]], uuid2smiles: Dict[str, str], node_dict: Dict[str, Dict[str, Any]]) -> List[str]:
    out_count, nodes_seen = {}, set()
    for e in edges:
        if not isinstance(e, dict):
            continue
        src = e.get("source")
        dst = e.get("target")
        if src:
            out_count[src] = out_count.get(src, 0) + 1
            nodes_seen.add(src)
        if dst:
            nodes_seen.add(dst)
    leaves = []
    for uid in nodes_seen:
        if out_count.get(uid, 0) > 0:
            continue
        smi = uuid2smiles.get(uid, uid)
        node = node_dict.get(smi, {})
        if node.get("type") == "chemical" or ">>" not in str(smi):
            leaves.append(smi)
    return leaves


def route_summary(route: Dict[str, Any]) -> Dict[str, Any]:
    edges = route.get("edges", []) or []
    props = route.get("props", {}) or {}
    node_dict = route.get("node_dict", {}) or {}
    uuid2smiles = route.get("uuid2smiles", {}) or {}
    reaction_nodes, chemical_nodes = _route_nodes(edges, uuid2smiles)
    leaves = _leaf_chemicals(edges, uuid2smiles, node_dict)

    plausibility = 1.0
    used = False
    for rxn in reaction_nodes:
        p = node_dict.get(rxn, {}).get("plausibility")
        try:
            p = float(p)
        except Exception:
            continue
        used = True
        plausibility *= p

    return {
        "route_id": route.get("route_id"),
        "depth": props.get("depth"),
        "precursor_cost": props.get("precursor_cost"),
        "score": props.get("score"),
        "cluster_id": props.get("cluster_id"),
        "num_reactions": len(reaction_nodes),
        "num_chemicals": len(chemical_nodes),
        "num_leaf_precursors": len(leaves),
        "reaction_nodes": reaction_nodes,
        "leaf_chemicals": leaves,
        "plausibility_product": plausibility if used else None,
    }
