import numpy as np

class RetrosyntheticEngine:
    """
    The Disconnection Engine: Implements the 'Strategic Disconnection' 
    of proteins into precursor synthons (epitopes).
    """
    def deconstruct(self, seq, window=15):
        synthons = []
        # Moving window analysis to simulate bond disconnection
        for i in range(len(seq) - window + 1):
            frag = seq[i:i+window]
            
            # Scoring synthons based on 'Retrosynthetic Priority'
            # (Calculated via Charge and Flexibility indices)
            score = (sum([1 for aa in frag if aa in "RKDHBE"]) / window) * 0.7 + \
                    (sum([1 for aa in frag if aa in "GSP"]) / window) * 0.3
            
            synthons.append({
                "synthon": frag, 
                "priority": score, 
                "index": i
            })
            
        # Returns the 5 highest-priority precursors for synthesis
        return sorted(synthons, key=lambda x: x['priority'], reverse=True)[:5]