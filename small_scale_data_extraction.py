import os
import sys
import sumolib
import numpy as np
import joblib


def extract_edge_features(net_file, sumo_cfg):
    """
    Extracts static features from the SUMO network, including internal edges.
    Args:
        net_file (str): Path to the SUMO network file (e.g., small.net.xml).
        sumo_cfg (str): Path to the SUMO configuration file (e.g., small.sumocfg).
    Returns:
        tuple: A dictionary with edge features and a list of all edge IDs.
    """
    # Verify that the network and config files exist.
    if not os.path.exists(net_file):
        raise FileNotFoundError(f"Network file not found: {net_file}")
    if not os.path.exists(sumo_cfg):
        raise FileNotFoundError(f"SUMO config file not found: {sumo_cfg}")

    # Read the SUMO network.
    net = sumolib.net.readNet(net_file)

    # Get all edge IDs (including internal ones).
    edge_ids = [edge.getID() for edge in net.getEdges()]

    # Dictionary to hold features for each edge.
    edge_features_dict = {}

    # Loop through each edge and extract features.
    for edge in net.getEdges():
        eid = edge.getID()
        # Extract basic static features.
        features = {
            'length': edge.getLength(),
            'num_lanes': len(edge.getLanes()),
            'speed_limit': edge.getSpeed(),
            'priority': edge.getPriority(),
            'junction_complexity': len(edge.getFromNode().getOutgoing()) + len(edge.getToNode().getIncoming())
        }
        edge_features_dict[eid] = features

    return edge_features_dict, edge_ids


def main():
    net_file = "small_scale_for_rl.net.xml"
    sumo_cfg = "small_scale_for_rl.sumocfg"

    # Extract edge features.
    edge_features_dict, edge_ids = extract_edge_features(net_file, sumo_cfg)
    print(f"Extracted features for {len(edge_ids)} edges.")

    # Prepare data to be saved.
    extracted_data = {
        'edge_features_dict': edge_features_dict,
        'edge_ids': edge_ids
    }

    # Save the extracted data as a pickle file.
    save_path = "small_scale_extracted_edge_features.pkl"
    joblib.dump(extracted_data, save_path)
    print(f"Extracted data saved to {save_path}")


if __name__ == "__main__":
    main()
